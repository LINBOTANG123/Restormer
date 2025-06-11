#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import skimage.exposure as ex
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from utils.mat_loader    import load_mri_data
from utils.noise_loader import load_noise_data, replicate_noise_map_with_sampling

NUM_COILS = 32  # adjust if needed

def load_model(model_path: str, inp_channels: int, device: str='cuda') -> torch.nn.Module:
    """
    Load the multi-coil Restormer model with dynamic input channels.
    """
    from basicsr.models.archs.restormer_arch import Restormer

    net = Restormer(
        inp_channels=inp_channels,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt['params'], strict=True)
    net.eval()
    return net

def main():
    p = argparse.ArgumentParser(
        description="Inference for multi-coil MRI denoising."
    )
    p.add_argument('--model_pth',   required=True,
                   help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_mat',     required=True,
                   help='Path to the MRI .mat/.h5 or NIfTI')
    p.add_argument('--mri_key',     default='image',
                   help='Dataset key inside .mat/HDF5 (ignored for simulate)')
    p.add_argument('--mri_format',
                   choices=['Hwihun_phantom','b1000','C','gslider', 'gslider_2', 'simulate'],
                   default='b1000',
                   help="MRI data format (use 'simulate' for NIfTI input)")
    p.add_argument('--noise_mat',   required=False,
                   help='Path to noise .mat/.h5 or NIfTI (only if using noise)')
    p.add_argument('--noise_key',   default='k_gc',
                   help='Key inside noise .mat/HDF5 (ignored for simulate)')
    p.add_argument('--noise_format',
                   choices=['Hwihun_phantom','b1000','C','gslider', 'gslider_2','simulate'],
                   default='b1000',
                   help="Noise data format (only if --use_noise)")
    p.add_argument('--use_noise', action='store_true',
                   help="If set, load and feed a noise‐map as second channel")
    p.add_argument('--output_folder', default='./results_infer',
                   help='Where to save NIfTI outputs')
    p.add_argument('--num_samples', type=int, default=2,
                   help='How many DWI channels to process (for MAT formats)')
    p.add_argument('--dwi_index', type=int, default=None,
               help='Specify which DWI sample index to denoise (0-based). If None, denoise all samples.')

    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # --- 1) Load MRI (always) ---
    mri_img = load_mri_data(
        args.mri_mat,
        key=args.mri_key,
        data_format=args.mri_format,
        num_samples_to_load=args.num_samples
    )
    H, W, _, mri_slices, num_samples = mri_img.shape
    print("Loaded MRI:", mri_img.shape)
    print("MRI min: ", mri_img.min(), "max: ", mri_img.max())

    # Compute global min/max for normalization
    eps = 1e-12
    gmin, gmax = float(mri_img.min()), float(mri_img.max())
    mri_norm = (mri_img - gmin) / (gmax - gmin + eps)
    mri_norm = np.clip(mri_norm, 0, 1)

    # --- 2) Optionally load Noise (2nd channel) ---
    if args.use_noise:
        if not args.noise_mat:
            raise ValueError("`--noise_mat` must be provided when `--use_noise`.")
        noise_img = load_noise_data(
            args.noise_mat,
            key=args.noise_key,
            data_format=args.noise_format
        )
        print("Loaded noise:", noise_img.shape)
        # --- Adjust noise slices if fewer than MRI slices ---
        if args.noise_format == 'gslider':
            # noise_maps = replicate_noise_map_with_sampling(noise_img, H, W, mri_slices)
            _, _, _, noise_slices = noise_img.shape
            if noise_slices < mri_slices:
                print(f"Expanding noise slices: {noise_slices} -> {mri_slices} by repeating each slice 5 times")
                noise_img_expanded = np.zeros((H, W, NUM_COILS, mri_slices), dtype=noise_img.dtype)
                for z in range(mri_slices):
                    src_z = min(z // 5, noise_slices - 1)  # avoid overflow
                    noise_img_expanded[:, :, :, z] = noise_img[:, :, :, src_z]
                    
                # flip around the midpoint so max→min and min→max
                mn, mx = noise_img_expanded.min(), noise_img_expanded.max()
                noise_img_expanded = (mn + mx) - noise_img_expanded

                noise_maps = noise_img_expanded

        if args.noise_format == 'b1000':
            noise_maps = replicate_noise_map_with_sampling(noise_img, H, W, mri_slices)

        # collapse to 2D per coil
        noise_map2D = noise_maps.mean(axis=-1)  # (H, W, coils)
        print("noise min: ", noise_map2D.min(), "max: ", noise_map2D.max())
        noise_norm = (noise_map2D - gmin) / (gmax - gmin + eps)
        noise_norm = np.clip(noise_norm, 0, 1)

        # save the per-coil noise map for inspection
        nib.save(nib.Nifti1Image(noise_norm.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'noise_map.nii'))

    # --- 3) Load model with dynamic input channels ---
    inp_ch = 2 if args.use_noise else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, inp_channels=inp_ch, device=device)

    # --- 4) Denoising Loop ---
    all_deno  = []
    all_res   = []
    all_orig  = []

    sample_indices = [args.dwi_index] if args.dwi_index is not None else list(range(num_samples))

    for sid in sample_indices:
        print(f"Sample {sid+1}/{num_samples}")
        vol = mri_norm[..., sid]         # (H, W, coils, slices)
        orig = np.sqrt((vol**2).sum(axis=2))
        all_orig.append(orig)

        deno_slices = []
        res_slices  = []

        for z in range(mri_slices):
            print(f"  Denoising slice {z+1}/{mri_slices} for sample {sid+1}/{num_samples}")
            deno_coils = []
            res_coils  = []
            for c in range(NUM_COILS):
                img_c = vol[:, :, c, z]
                if args.use_noise:
                    noise_c = noise_norm[:, :, c]
                    arr = np.stack([img_c, noise_c], axis=0)
                else:
                    arr = img_c[np.newaxis, ...]  # single‐channel

                inp = torch.tensor(arr,
                                   dtype=torch.float32,
                                   device=device).unsqueeze(0)  # (1, C, H, W)

                # pad to multiples of 8
                _, _, h, w = inp.shape
                ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
                if ph or pw:
                    inp = F.pad(inp, (0, pw, 0, ph), mode='reflect')

                with torch.no_grad():
                    out = model(inp)
                out    = out[..., :h, :w]
                target = inp[:, 0:1, :h, :w]
                res    = out - target

                deno_coils.append(out.cpu().squeeze().numpy())
                res_coils.append(res.cpu().squeeze().numpy())

            deno_slices.append(np.stack(deno_coils, axis=2))
            res_slices.append(np.stack(res_coils, axis=2))

        deno_vol = np.stack(deno_slices, axis=2)
        res_vol  = np.stack(res_slices, axis=2)
        # Save per-coil original and denoised results (H, W, Coils, Slices)
        orig_path = os.path.join(args.output_folder, f"original_percoil_sample{sid+1}.nii")
        deno_path = os.path.join(args.output_folder, f"denoised_percoil_sample{sid+1}.nii")
        vol_transposed = np.transpose(vol, (0, 1, 3, 2))
        # denormalize per‐coil original and denoised volumes
        vol_denorm      = (vol_transposed         * (gmax - gmin + eps)) + gmin            # vol is mri_norm[...,sid]
        deno_vol_denorm = (deno_vol    * (gmax - gmin + eps)) + gmin            # deno_vol from normalized input
        # then transpose and save
        vol_t = np.transpose(vol_denorm,      (0,1,3,2))
        deno_t= np.transpose(deno_vol_denorm, (0,1,3,2))
        nib.save(nib.Nifti1Image(vol_t.astype(np.float32), np.eye(4)), orig_path)
        nib.save(nib.Nifti1Image(deno_t.astype(np.float32), np.eye(4)), deno_path)

        all_deno.append(np.sqrt((deno_vol**2).sum(axis=3)))
        all_res.append(np.sqrt((res_vol**2).sum(axis=3)))

    # --- 5) Assemble & Save ---
    deno4d = np.stack(all_deno, axis=-1)
    res4d  = np.stack(all_res, axis=-1)
    orig4d = np.stack(all_orig, axis=-1)

    deno4d_denorm = deno4d * (gmax - gmin + eps) + gmin
    res4d_denorm  = res4d  * (gmax - gmin + eps) + gmin
    orig4d_denorm = orig4d * (gmax - gmin + eps) + gmin

    # — compute log10(|residual|) histogram —
    res_vals = res4d_denorm.ravel()
    # 1) compute squared residuals
    res_sq = res_vals**2

    # 2) histogram with density=True
    counts, bin_edges = np.histogram(res_sq, bins=100, density=True)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # 3) log of the density
    eps = 1e-12
    log_counts = np.log(counts + eps)

    # 4) plot
    plt.figure()
    plt.plot(bin_centers, log_counts, '.')
    plt.title('ln PDF vs. residual² (Ours)')
    plt.xlabel('residual²')
    plt.ylabel('ln density')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder,
                            'ours_residual_logpdf_vs_sq.png'))
    plt.close()

    # — compute kurtosis —
    res_kurt = kurtosis(res_vals, fisher=True, bias=False)
    print(f"[Ours] Residual kurtosis: {res_kurt:.6f}")

    # — save kurtosis to text file —
    kurt_file = os.path.join(args.output_folder, 'ours_residual_kurtosis.txt')
    with open(kurt_file, 'w') as f:
        f.write(f"Residual kurtosis: {res_kurt:.6f}\n")

    nib.save(nib.Nifti1Image(deno4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'combined_denoised_all.nii'))
    nib.save(nib.Nifti1Image(res4d_denorm.astype(np.float32),  np.eye(4)), 
            os.path.join(args.output_folder, 'combined_residual_all.nii'))
    nib.save(nib.Nifti1Image(orig4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'original_coilcombined_all.nii'))

    print("All outputs saved in", args.output_folder)


if __name__ == "__main__":
    main()
