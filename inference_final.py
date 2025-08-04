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
import pdb
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
    p.add_argument('--mask_nifti', required=False,
                help='Optional brain mask NIfTI (.nii or .nii.gz) to restrict denoising to brain tissue')


    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # --- 1) Load MRI (always) ---
    mri_img = load_mri_data(
        args.mri_mat,
        key=args.mri_key,
        data_format=args.mri_format,
        num_samples_to_load=args.num_samples
    )
    H, W, C, S, N = mri_img.shape
    print("Loaded MRI:", mri_img.shape)
    print("MRI min: ", mri_img.min(), "max: ", mri_img.max(), "mean: ", np.mean(mri_img), "std: ", np.std(mri_img))

    # --- 2) Optional brain mask --------------------------------------------------
    if args.mask_nifti:
        mask = nib.load(args.mask_nifti).get_fdata().astype(bool)          # (H,W,S)
        if mask.shape != (H, W, S):
            raise ValueError(f"Mask shape {mask.shape} != {(H,W,S)}")
        # broadcast & apply to MRI  (H,W,S,1,1) -> (H,W,C,S,N)
        mri_img *= mask[:, :, None, :, None]
        mask_full = np.broadcast_to(mask[:, :, None, :, None], mri_img.shape)
        brain_voxels = mri_img[mask_full]
        p_cap = np.percentile(brain_voxels, 99.8)

    else:
        mask = None
        p_cap = np.percentile(mri_img, 99.8)

    # scale_factor = 0.99 / (p_cap + 1e-12)
    # scale_factor = 3e6
    scale_factor = 1.0
    print(f"Dynamic scale factor: {scale_factor:.3e}")

    mri_scaled = np.clip(mri_img * scale_factor, 0, 1)


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
            _, _, _, noise_slices = noise_img.shape
            mn, mx = noise_img.min(), noise_img.max()
            noise_maps = (mn + mx) - noise_img
            # Assuming noise_maps has shape (H, W, C, 34)
            repeats = [5] * 34  # Repeat each slice 5 times
            noise_maps = np.repeat(noise_maps, repeats, axis=3)  # Expand along last dimension
            print("Expanded noise maps shape:", noise_maps.shape)  # Should be (290, 290, 32, 170)
        if args.noise_format == 'b1000':
            noise_maps = replicate_noise_map_with_sampling(noise_img, H, W, S)

        # collapse to 2D per coil
        print("NOISE MAP min: ", noise_maps.min(), "max: ", noise_maps.max(),  "mean: ", np.mean(noise_maps), "std", np.std(noise_maps))

        # apply mask to noise
        if mask is not None:
            noise_maps *= mask[:, :, None, :]
            
        # --- 2b) Brain-masked raw stats ---------------------------------------------
        if mask is not None:
            # MRI stats
            mask_full = np.broadcast_to(mask[:, :, None, :, None], mri_img.shape)  # (H,W,C,S,N)
            valid_mri = mri_img[mask_full] 
            print(f"[RAW brain]  MRI   min={valid_mri.min():.4g} "
                f"max={valid_mri.max():.4g} "
                f"mean={valid_mri.mean():.4g} "
                f"std={valid_mri.std():.4g} "
                f"p10={np.percentile(valid_mri,10):.4g} "
                f"p90={np.percentile(valid_mri,90):.4g}")

            if args.use_noise:
                mask_noise = np.broadcast_to(mask[:, :, None, :], noise_maps.shape)    # (H,W,C,S)
                valid_noise = noise_maps[mask_noise]
                print(f"[RAW brain]  Noise min={valid_noise.min():.4g} "
                    f"max={valid_noise.max():.4g} "
                    f"mean={valid_noise.mean():.4g} "
                    f"std={valid_noise.std():.4g} "
                    f"p10={np.percentile(valid_noise,10):.4g} "
                    f"p90={np.percentile(valid_noise,90):.4g}")

        noise_norm = noise_maps.mean(axis=-1)  # (H, W, coils)

        noise_norm = noise_norm * scale_factor
        noise_norm = np.clip(noise_norm, 0, 1)

        # noise_lin = noise_map2D * scale_factor
        # noise_lin = noise_map2D
        # # percentile-based gain
        # p99   = np.percentile(noise_lin, 99)
        # gamma = 0.095 / (p99 + 1e-12)          # use 0.08 if you want the 99-th % at 0.08
        # noise_norm = np.clip(noise_lin * gamma, 0, 0.095)

        # print("noise   mean :", noise_norm.mean())
        # print("noise  95th% :", np.percentile(noise_norm, 95))
        # print("noise   max  :", noise_norm.max())
        # ------------------------------------------------------------

        # save for inspection (optional)
        nib.save(nib.Nifti1Image(noise_norm.astype(np.float32), np.eye(4)),
                os.path.join(args.output_folder, 'noise_map_scaled.nii'))
        nib.save(nib.Nifti1Image(noise_maps.astype(np.float32), np.eye(4)),
                os.path.join(args.output_folder, 'noise_map_raw.nii'))


    # --- 3) Load model with dynamic input channels ---
    inp_ch = 2 if args.use_noise else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, inp_channels=inp_ch, device=device)

    # --- 4) Denoising Loop ---
    all_deno  = []
    all_res   = []
    all_orig  = []

    sample_indices = [args.dwi_index] if args.dwi_index is not None else list(range(N))

    for sid in sample_indices:
        print(f"Sample {sid+1}/{S}")
        vol = mri_scaled[..., sid]         # (H, W, coils, slices)
        orig = np.sqrt((vol**2).sum(axis=2))
        all_orig.append(orig)

        deno_slices = []
        res_slices  = []

        for z in range(S):
            print(f"  Denoising slice {z+1}/{S} for sample {sid+1}/{S}")
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
                # --- new: debias ---
                bias = (out - inp[:, 0:1, :h, :w]).mean()     # scalar
                out = out - bias                      # add back the mean

                target = inp[:, 0:1, :h, :w]
                res    = out - target

                deno_coils.append(out.cpu().squeeze().numpy())
                res_coils.append(res.cpu().squeeze().numpy())

            deno_slices.append(np.stack(deno_coils, axis=2))
            res_slices.append(np.stack(res_coils, axis=2))

        deno_vol = np.stack(deno_slices, axis=2)
        res_vol  = np.stack(res_slices, axis=2)
        # Save per-coil original and denoised results (H, W, Coils, Slices)
        # --- Save per-coil volumes (brain-masked, correct axis order) -------------
        deno_vol_t = deno_vol
        if mask is not None:
            brain_mask_4d = mask[..., None]                        # (H,W,S,1)
            vol_denorm  = (np.transpose(vol, (0, 1, 3, 2)) / scale_factor) * brain_mask_4d
            deno_denorm = (deno_vol_t / scale_factor) * brain_mask_4d
        else:
            vol_denorm  = np.transpose(vol, (0, 1, 3, 2)) / scale_factor
            deno_denorm = deno_vol_t / scale_factor

        orig_path = os.path.join(args.output_folder, f"original_percoil_sample{sid+1}.nii")
        deno_path = os.path.join(args.output_folder, f"denoised_percoil_sample{sid+1}.nii")

        # nib.save(nib.Nifti1Image(vol_denorm.astype(np.float32),  np.eye(4)), orig_path)
        # nib.save(nib.Nifti1Image(deno_denorm.astype(np.float32), np.eye(4)), deno_path)

        all_deno.append(np.sqrt((deno_vol**2).sum(axis=3)))
        all_res.append(np.sqrt((res_vol**2).sum(axis=3)))

    if mask is not None:
        brain_mask_3d = mask.astype(np.float32)       # (H,W,S)
        for i in range(len(all_orig)):
            all_orig[i] *= brain_mask_3d
            all_deno[i] *= brain_mask_3d
            all_res[i]  *= brain_mask_3d
    # --- 5) Assemble & Save ---
    deno4d = np.stack(all_deno, axis=-1)
    res4d  = np.stack(all_res, axis=-1)
    orig4d = np.stack(all_orig, axis=-1)

    deno4d_denorm = deno4d / scale_factor
    res4d_denorm  = res4d / scale_factor
    orig4d_denorm = orig4d / scale_factor

    nib.save(nib.Nifti1Image(deno4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'combined_denoised_all.nii'))
    nib.save(nib.Nifti1Image(res4d_denorm.astype(np.float32),  np.eye(4)), 
            os.path.join(args.output_folder, 'combined_residual_all.nii'))
    nib.save(nib.Nifti1Image(orig4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'original_coilcombined_all.nii'))

    print("All outputs saved in", args.output_folder)


if __name__ == "__main__":
    main()
