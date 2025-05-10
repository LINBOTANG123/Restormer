import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# Number of coil groups (each group corresponds to one coil's paired channels)
NUM_COILS = 32

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##############################
## Layer Norm
##############################
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # Handle both 3-D and 4-D input
        if x.ndim == 3:
            return self.body(x)
        elif x.ndim == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            raise ValueError("Unexpected input dimension for LayerNorm")

##############################
## Gated-Dconv Feed-Forward Network (GDFN)
##############################
class FeedForward(nn.Module):
    def __init__(self, channels, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_channels = int(channels * ffn_expansion_factor)
        if hidden_channels % NUM_COILS != 0:
            hidden_channels = NUM_COILS * ((hidden_channels + NUM_COILS - 1) // NUM_COILS)
        self.project_in = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=bias, groups=NUM_COILS)
        self.dwconv = nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, bias=bias, groups=NUM_COILS)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias, groups=NUM_COILS)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##############################
## Multi-DConv Head Transposed Self-Attention (MDTA)
##############################
class Attention(nn.Module):
    def __init__(self, channels, heads, bias):
        super(Attention, self).__init__()
        self.num_heads = heads  # ideally one head per coil
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias, groups=NUM_COILS)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, bias=bias, groups=NUM_COILS)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias, groups=NUM_COILS)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.split(c, dim=1)
        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = torch.matmul(attn, v)
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out

##############################
## Transformer Block
##############################
class TransformerBlock(nn.Module):
    def __init__(self, channels, heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(channels, LayerNorm_type)
        self.attn = Attention(channels, heads, bias)
        self.norm2 = LayerNorm(channels, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm1(x.view(b, c, -1).transpose(1, 2))
        x_norm = x_norm.transpose(1, 2).view(b, c, h, w)
        x = x + self.attn(x_norm)
        x_norm2 = self.norm2(x.view(b, c, -1).transpose(1, 2))
        x_norm2 = x_norm2.transpose(1, 2).view(b, c, h, w)
        x = x + self.ffn(x_norm2)
        return x

##############################
## Overlap Patch Embedding
##############################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        return self.proj(x)

##############################
## Downsample / Upsample with Grouped Operations
##############################
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False, groups=NUM_COILS)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_unshuffle(x)
        return x

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        # Here, 'channels' is the number of input channels to this upsampling layer.
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False, groups=NUM_COILS)
        self.pixel_shuffle = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

##############################
## coil_concat for Skip Connections (coil-wise)
##############################
def coil_concat(x1, x2, groups=NUM_COILS):
    b, C1, h, w = x1.shape
    b2, C2, h2, w2 = x2.shape
    assert b == b2 and h == h2 and w == w2, "Dimensions must match."
    coil = groups
    assert C1 % coil == 0 and C2 % coil == 0, "Channels must be divisible by number of coils."
    f1 = C1 // coil
    f2 = C2 // coil
    x1_coils = x1.view(b, coil, f1, h, w)
    x2_coils = x2.view(b, coil, f2, h, w)
    x_cat = torch.cat([x1_coils, x2_coils], dim=2)
    return x_cat.view(b, coil * (f1 + f2), h, w)

##############################
## Interleave Channels
##############################
def interleave_channels(x):
    """
    Rearranges the input tensor of shape [B, 64, H, W] (first 32: coil images, next 32: noise maps)
    into [B, 64, H, W] with interleaved order:
    [coil1, noise1, coil2, noise2, ..., coil32, noise32]
    """
    B, C, H, W = x.shape
    assert C == 64, "Expected 64 channels (32 coil images + 32 noise maps)"
    coils = x[:, :32, :, :]
    noise = x[:, 32:, :, :]
    x_int = torch.stack([coils, noise], dim=2)  # [B, 32, 2, H, W]
    x_int = x_int.view(B, 64, H, W)
    return x_int

##############################
## Restormer for Multi-Coil MRI Denoising
##############################
class Restormer(nn.Module):
    def __init__(self, 
                 inp_channels=64,         # 64 input channels: 32 coil images + 32 noise maps
                 out_channels=32,         # 32 outputs (one per coil)
                 dim=48,
                 num_blocks=[4,6,6,8],
                 num_refinement_blocks=4,
                 heads=[32,32,32,32],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='BiasFree',
                 dual_pixel_task=False):
        super(Restormer, self).__init__()
        self.inp_channels = inp_channels
        # Internal channel progression; must be divisible by NUM_COILS.
        channels = [64, 128, 256, 512]
        # Initial embedding: from inp_channels to channels[0], grouped by coil.
        self.embed_conv = nn.Conv2d(inp_channels, channels[0], kernel_size=3, padding=1, bias=False, groups=NUM_COILS)
        # Encoder stages
        self.encoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(ch, heads, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(nb)])
            for ch, heads, nb in zip(channels, heads, num_blocks)
        ])
        # Downsampling layers between encoder stages
        self.downs = nn.ModuleList([DownSample(ch) for ch in channels[:-1]])
        # Decoder stages
        self.decoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(channels[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])]),
            nn.Sequential(*[TransformerBlock(channels[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])]),
            nn.Sequential(*[TransformerBlock(channels[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        ])
        # Upsampling layers between decoder stages.
        # IMPORTANT: We must use the actual channel count from the encoder output.
        # enc4 output has channels[3]=512, so first upsample uses 512.
        # enc3 output has channels[2]=256, so second upsample uses 256.
        # enc2 output has channels[1]=128, so third upsample uses 128.
        self.ups = nn.ModuleList([
            UpSample(channels[3]),  # from 512 -> 512*2/4 = 256 output channels
            UpSample(channels[2]),  # from 256 -> 256*2/4 = 128 output channels
            UpSample(channels[1])   # from 128 -> 128*2/4 = 64 output channels
        ])
        # 1x1 reduction layers for skip connections (grouped by coil)
        self.reduces = nn.ModuleList([
            nn.Conv2d(channels[3], channels[2], kernel_size=1, bias=False, groups=NUM_COILS),
            nn.Conv2d(channels[2], channels[1], kernel_size=1, bias=False, groups=NUM_COILS)
        ])
        # Refinement stage at full resolution
        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type)
                                           for _ in range(num_refinement_blocks)])
        # Output conv to produce 32 denoised coil images, grouped by coil.
        self.output_conv = nn.Conv2d(channels[1], out_channels, kernel_size=3, padding=1, bias=False, groups=NUM_COILS)

    def forward(self, x):
        # x shape: [B, 64, H, W] (first 32: coil images, next 32: noise maps)
        # Interleave channels so that the order becomes:
        # [coil1, noise1, coil2, noise2, ..., coil32, noise32]
        x = interleave_channels(x)
        # Encoder path
        x0 = self.embed_conv(x)  # [B, channels[0], H, W]
        enc1 = self.encoders[0](x0)  # output: [B, 64, H, W]
        enc2 = self.encoders[1](self.downs[0](enc1))  # output: [B, 128, H/2, W/2]
        enc3 = self.encoders[2](self.downs[1](enc2))  # output: [B, 256, H/4, W/4]
        enc4 = self.encoders[3](self.downs[2](enc3))  # output: [B, 512, H/8, W/8]
        # Decoder path with coil-wise skip connections
        up3 = self.ups[0](enc4)  # Upsample: input 512 -> output 256, shape: [B, 256, H/4, W/4]
        comb3 = coil_concat(up3, enc3, groups=NUM_COILS)  # Should have 256+256 = 512 channels, but per-coil it’s 16+16=32 channels? Check per group.
        dec3 = self.decoders[0](self.reduces[0](comb3))  # Reduction: 512 -> 256 channels (grouped by coil)
        up2 = self.ups[1](dec3)  # Upsample: input 256 -> output 128, shape: [B, 128, H/2, W/2]
        comb2 = coil_concat(up2, enc2, groups=NUM_COILS)  # Concatenate, then reduce to 128 channels (grouped)
        dec2 = self.decoders[1](self.reduces[1](comb2))  # Reduction: 128 -> 128 channels (grouped)
        up1 = self.ups[2](dec2)  # Upsample: input 128 -> output 64, shape: [B, 64, H, W]
        comb1 = coil_concat(up1, enc1, groups=NUM_COILS)  # Concatenate, resulting in 64+64=128 channels (but grouped per coil)
        dec1 = self.decoders[2](comb1)  # Process at full resolution.
        ref = self.refinement(dec1)  # Refinement stage.
        out_denoised = self.output_conv(ref)  # [B, 32, H, W] denoised outputs per coil.
        # Residual connection: add original coil images (first 32 channels of interleaved x)
        coil_images = x[:, :32, :, :]
        return out_denoised + coil_images

# Example usage:
if __name__ == "__main__":
    model = Restormer()
    dummy_input = torch.randn(1, 64, 128, 128)
    output = model(dummy_input)
    print(output.shape)  # Expected: [1, 32, 128, 128]
