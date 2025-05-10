#!/usr/bin/env python
import os
import argparse
import numpy as np
import scipy.io as sio
import nibabel as nib
import torch
import torch.nn.functional as F
import h5py
# Import the MRI loader util.
from utils.mat_loader import load_mri_data
from utils.noise_loader  import load_noise_data, replicate_noise_map_with_sampling, process_noise_kspace_to_img

NUM_COILS = 32  # Total number of coils

def load_model(model_path, device='cuda'):
    """
    Load the multi-coil Restormer model.
    Configured for a 2->1 mapping:
      - inp_channels=2 (coil image and its noise map)
      - out_channels=1 (denoised coil image)
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=2,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    )
    net_g.to(device)
    ckpt = torch.load(model_path, map_location=device)
    net_g.load_state_dict(ckpt['params'], strict=True)
    net_g.eval()
    return net_g

def main():
    parser = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising (2->1 mapping).")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_pth', type=str, required=True, help='Path to the MRI file (.mat or .h5)')
    parser.add_argument('--noise_mat', type=str, required=True, help='Path to the pure noise file (.mat or .h5)')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--data_scale_factor', type=float, default=5e4, help='Scaling factor for data (if not normalizing)')
    parser.add_argument(
    "--noise_format",
        choices=["b1000","Hwihun_phantom","C","gslider"],
        default="b1000",
        help="Which pure-noise MAT format to load (gslider→key='image')."
    )
    parser.add_argument(
        "--mri_format",
        choices=["Hwihun_phantom","b1000","C","gslider"],
        default="b1000",
        help="Which MRI MAT format to load (use 'gslider' for your gSlider_kspace_coil.mat)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Which MRI MAT format to load (use 'gslider' for your gSlider_kspace_coil.mat)"
    )


    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Load all MRI samples using the separate utility. 
    mri_key = "k_coil" if args.mri_format == "gslider" else "image"

    mri_img = load_mri_data(
        args.mri_pth,
        key=mri_key,
        data_format=args.mri_format,
        num_samples_to_load=args.samples 
    )
    H, W, num_coils, mri_slices, num_samples = mri_img.shape
    print("MRI image shape:", mri_img.shape)

    # Load noise data (pure noise)
    noise_key = "image" if args.noise_format=="gslider" else "k_gc"
    noise_imgspace = load_noise_data(
        args.noise_mat,
        key=noise_key,
        data_format=args.noise_format
    )
    H_noise, W_noise, _, noise_slices = noise_imgspace.shape
    print("Noise image space shape:", noise_imgspace.shape)

    # Replicate noise maps to match MRI spatial dimensions and mri_slices.
    noise_maps = replicate_noise_map_with_sampling(noise_imgspace, H, W, mri_slices)

    # Apply data scaling.
    scale = args.data_scale_factor
    mri_img_scaled = mri_img * scale   # Shape: (H, W, num_coils, mri_slices, num_samples)
    # Compute per-coil 2D noise map by averaging along the slice dimension.
    noise_map_2D = np.mean(noise_maps, axis=-1)  # Shape: (H, W, num_coils)
    noise_map_2D_scaled = noise_map_2D * scale * 0.75

    # Save the per-coil noise map as a 3D NIfTI file.
    noise_map_filename = os.path.join(args.output_folder, "noise_map.nii")
    nib.save(nib.Nifti1Image(noise_map_2D.astype(np.float32), affine=np.eye(4)), noise_map_filename)
    print(f"Saved per-coil noise map as {noise_map_filename}")

    # Prepare lists for concatenated outputs from all samples.
    all_combined_denoised = []
    all_combined_residual = []
    all_original_coil_combined = []

    # Load the model.
    model = load_model(args.model_pth, device='cuda')

    # HARDCODE TO PROCESS FIRST 5 SAMPLES
    for sample in range(args.samples):
        print(f"Processing sample {sample+1}/{num_samples}...")
        # For the original coil-combined input image, use the scaled MRI input for the sample.
        mri_sample = mri_img_scaled[..., sample]  # shape: (H, W, num_coils, mri_slices)
        combined_original = np.sqrt(np.sum(mri_sample**2, axis=2)) / scale  # (H, W, mri_slices)
        all_original_coil_combined.append(combined_original)

        # Now perform denoising on a per-coil basis.
        denoised_slices = []
        residual_slices = []
        for s in range(mri_slices):
            denoised_coils = []
            residual_coils = []
            for c in range(num_coils):
                coil_img = mri_sample[:, :, c, s]
                coil_noise = noise_map_2D_scaled[:, :, c]  # same for all slices/samples
                input_array = np.stack([coil_img, coil_noise], axis=0)  # (2, H, W)
                input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().cuda()  # (1,2,H,W)

                _, _, H_in, W_in = input_tensor.shape
                pad_h = (8 - H_in % 8) % 8
                pad_w = (8 - W_in % 8) % 8
                if pad_h or pad_w:
                    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                with torch.no_grad():
                    pred = model(input_tensor)  # (1,1,H_out,W_out)
                pred = pred[:, :, :H_in, :W_in]  # Crop back if needed
                target_tensor = input_tensor[:, 0:1, :H_in, :W_in]
                residual = pred - target_tensor
                denoised_coils.append(pred.cpu().squeeze(0).squeeze(0))   # (H, W)
                residual_coils.append(residual.cpu().squeeze(0).squeeze(0))
            denoised_slice = np.stack(denoised_coils, axis=2)   # (H, W, num_coils)
            residual_slice = np.stack(residual_coils, axis=2)   # (H, W, num_coils)
            denoised_slices.append(denoised_slice)
            residual_slices.append(residual_slice)
            print(f"  Processed slice {s+1}/{mri_slices}")
        # Stack slices for this sample (pre coil-combination)
        denoised_vol_sample = np.stack(denoised_slices, axis=2)   # (H, W, mri_slices, num_coils)
        residual_vol_sample = np.stack(residual_slices, axis=2)   # (H, W, mri_slices, num_coils)
        # Coil combine using root-sum-of-squares over coils to produce a 3D volume: (H, W, mri_slices)
        combined_denoised = np.sqrt(np.sum(denoised_vol_sample**2, axis=3))
        combined_residual = np.sqrt(np.sum(residual_vol_sample**2, axis=3))
        
        # Revert scaling (divide by scale)
        final_denoised_pre = denoised_vol_sample / scale
        final_residual_pre = residual_vol_sample / scale
        final_combined_denoised = combined_denoised / scale
        final_combined_residual = combined_residual / scale

        # Append the concatenated coil-combined results to the lists.
        all_combined_denoised.append(final_combined_denoised)
        all_combined_residual.append(final_combined_residual)

    # After processing all samples, concatenate the coil-combined results to a 4D tensor:
    # New tensor shape: (H, W, mri_slices, num_samples)
    combined_deno_all = np.stack(all_combined_denoised, axis=-1)
    combined_res_all = np.stack(all_combined_residual, axis=-1)
    original_combined_all = np.stack(all_original_coil_combined, axis=-1)
    
    # Save the concatenated 4D tensors.
    combined_deno_all_filename = os.path.join(args.output_folder, "combined_denoised_all.nii")
    combined_res_all_filename = os.path.join(args.output_folder, "combined_residual_all.nii")
    original_combined_all_filename = os.path.join(args.output_folder, "original_coilcombined_all.nii")
    
    nib.save(nib.Nifti1Image(combined_deno_all.astype(np.float32), affine=np.eye(4)), combined_deno_all_filename)
    nib.save(nib.Nifti1Image(combined_res_all.astype(np.float32), affine=np.eye(4)), combined_res_all_filename)
    nib.save(nib.Nifti1Image(original_combined_all.astype(np.float32), affine=np.eye(4)), original_combined_all_filename)
    
    print(f"Saved concatenated coil-combined denoised tensor as {combined_deno_all_filename}")
    print(f"Saved concatenated coil-combined residual tensor as {combined_res_all_filename}")
    print(f"Saved concatenated original input coil-combined tensor as {original_combined_all_filename}")

if __name__ == "__main__":
    main()
