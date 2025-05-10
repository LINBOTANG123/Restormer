#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.io as sio
import cv2
import scipy.ndimage as ndimage

###########################
# MRI & Noise Loading Functions
###########################

def load_mri_kspace_and_transform_raw(file_path, key_kspace="image"):
    """
    Load MRI k-space data from a MATLAB .mat file using scipy.io.loadmat.
    Expected shape in the file: (num_slices, num_samples, num_coils, H, W)
    This function transposes the data to shape (H, W, num_coils, num_slices, num_samples)
    and returns the magnitude image for each k-space slice.
    (Here we assume the data are already reconstructed, so no FFT is applied.)
    """
    data = sio.loadmat(file_path)
    print("Available keys in MRI .mat file:", list(data.keys()))
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    mri_kspace = data[key_kspace]
    # Expected shape: (num_slices, num_samples, num_coils, H, W)
    if mri_kspace.ndim != 5:
        raise ValueError("Expected MRI k-space shape: (num_slices, num_samples, num_coils, H, W). Got shape: " + str(mri_kspace.shape))
    # Transpose to (H, W, num_coils, num_slices, num_samples)
    mri_kspace = np.transpose(mri_kspace, (3, 4, 2, 0, 1))
    H, W, num_coils, num_slices, num_samples = mri_kspace.shape
    print(f"Loaded MRI k-space with shape: {mri_kspace.shape} (H, W, num_coils, num_slices, num_samples)")
    
    # Here we assume that the k-space data is already in image space,
    # so we simply take the absolute value.
    mri_imgspace = np.abs(mri_kspace).astype(np.float32)
    return mri_imgspace

def load_pure_noise_kspace_and_transform_raw(file_path, key_kspace="k_gc"):
    """
    Load pure noise k-space data from a .mat file using scipy.io.loadmat.
    Expected shape: (H, W, num_coils, num_slices)
    Returns a numpy array of magnitude images with shape (H, W, num_coils, num_slices).
    """
    data = sio.loadmat(file_path)
    print("Available keys in pure noise .mat file:", list(data.keys()))
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    noise_kspace = data[key_kspace]
    if noise_kspace.ndim != 4:
        raise ValueError("Expected noise k-space shape: (H, W, num_coils, num_slices). Got shape: " + str(noise_kspace.shape))
    H, W, num_coils, num_slices = noise_kspace.shape
    print(f"Loaded pure noise k-space with shape: {noise_kspace.shape}")
    noise_imgspace = np.abs(noise_kspace).astype(np.float32)
    return noise_imgspace

###########################
# Noise Map Expansion
###########################

def replicate_noise_map(noise_imgspace, target_H, target_W, target_slices):
    """
    Given noise_imgspace of shape (H_noise, W_noise, num_coils, noise_slices),
    for each coil:
      - Pad the noise map (for each available slice) with edge-padding to expand from (H_noise, W_noise)
        to (target_H, target_W).
      - For slices beyond the available noise_slices, fill with the average noise map.
    Returns an array of shape (target_H, target_W, num_coils, target_slices).
    """
    H_noise, W_noise, num_coils, noise_slices = noise_imgspace.shape
    expanded_noise = np.zeros((target_H, target_W, num_coils, target_slices), dtype=noise_imgspace.dtype)
    
    for coil in range(num_coils):
        coil_maps = []
        # Process each available noise slice.
        for s in range(noise_slices):
            noise_map = noise_imgspace[:, :, coil, s]
            pad_h = target_H - H_noise
            pad_w = target_W - W_noise
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_map = np.pad(noise_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            padded_map = padded_map[:target_H, :target_W]
            coil_maps.append(padded_map)
        coil_maps = np.stack(coil_maps, axis=2)  # shape: (target_H, target_W, noise_slices)
        avg_map = np.mean(coil_maps, axis=2)
        final_maps = np.zeros((target_H, target_W, target_slices), dtype=coil_maps.dtype)
        if target_slices <= noise_slices:
            final_maps = coil_maps[:, :, :target_slices]
        else:
            final_maps[:, :, :noise_slices] = coil_maps
            for s in range(noise_slices, target_slices):
                final_maps[:, :, s] = avg_map
        expanded_noise[:, :, coil, :] = final_maps
    return expanded_noise

def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    # Model configured for per-coil processing with 2-channel input (MRI image and noise map) and 1-channel output.
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

###########################
# Inference Pipeline (Per-Coil Processing)
###########################

def run_inference_from_mat(model_pth, mri_mat_file, noise_mat_file, output_folder, device='cuda',
                           data_scale_factor=1):
    """
    Perform inference with a modified Restormer-like model used per-coil.
    For each MRI slice and for each coil:
      - Extract the MRI image for that coil and multiply by data_scale_factor.
      - Extract the corresponding noise map (expanded to MRI dimensions).
      - Form a 2-channel input: channel 1 is the MRI image, channel 2 is the noise map.
      - The model outputs a 1-channel denoised image for that coil.
    The outputs are assembled into 4D volumes of shape (H, W, mri_slices, num_coils)
    and saved as NIfTI files. Additionally, individual PNGs per coil are saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_out_dir = os.path.join(output_folder, "png_outputs")
    os.makedirs(png_out_dir, exist_ok=True)
    nii_dir = os.path.join(output_folder, "nii")
    os.makedirs(nii_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)
    
    # Load noise maps.
    noise_imgspace = load_pure_noise_kspace_and_transform_raw(noise_mat_file, key_kspace="k_gc")
    H_noise, W_noise, num_coils_noise, noise_slices = noise_imgspace.shape
    print(f"Loaded pure noise k-space with shape: {noise_imgspace.shape}")
    
    # Load MRI images.
    mri_imgspace = load_mri_kspace_and_transform_raw(mri_mat_file, key_kspace="k_gc")
    H, W, num_coils_mri, mri_slices, num_samples = mri_imgspace.shape
    print(f"Processing MRI data with shape: {mri_imgspace.shape}")
    
    # Expand noise maps to match MRI dimensions.
    expanded_noise_maps = replicate_noise_map(noise_imgspace, target_H=H, target_W=W, target_slices=mri_slices)
    # Rearrange noise maps for per-coil inference: shape (mri_slices, num_coils, H, W)
    noise_maps_all = np.transpose(expanded_noise_maps, (3, 2, 0, 1))
    
    # Save average noise maps (averaged over slices) for record.
    avg_noise = np.mean(expanded_noise_maps, axis=3)  # (H, W, num_coils)
    avg_noise = np.transpose(avg_noise, (1, 2, 0))       # (H, W, num_coils)
    noise_nii_path = os.path.join(output_folder, "noise_maps.nii")
    nib.save(nib.Nifti1Image(avg_noise, affine=np.eye(4)), noise_nii_path)
    print(f"Saved noise maps NIfTI: {noise_nii_path}")
    
    # Prepare lists to accumulate per-slice volumes (each slice has outputs for all coils).
    original_volumes = []   # List of arrays: (num_coils, H, W) for each slice.
    denoised_volumes = []   # List of arrays: (num_coils, H, W) for each slice.
    residual_volumes = []   # List of arrays: (num_coils, H, W) for each slice.
    
    sample_idx = 0  # Process one sample.
    
    for slice_idx in range(mri_slices):
        original_coil_list = []
        denoised_coil_list = []
        residual_coil_list = []
        for coil in range(num_coils_mri):
            # Get the MRI image for this coil.
            mri_img_coil = mri_imgspace[:, :, coil, slice_idx, sample_idx] * data_scale_factor
            # Get the corresponding noise map from expanded noise maps.
            noise_map_coil = noise_maps_all[slice_idx, coil, :, :]  # shape: (H, W)
            # Form the 2-channel input.
            input_two_channel = np.stack([mri_img_coil, noise_map_coil], axis=0)  # (2, H, W)
            inp_tensor = torch.from_numpy(input_two_channel).unsqueeze(0).to(device)  # (1,2,H,W)
            
            # Pad if necessary.
            _, _, h, w = inp_tensor.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                inp_tensor = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            
            with torch.no_grad():
                output = model(inp_tensor)
            if pad_h > 0 or pad_w > 0:
                output = output[..., :h, :w]
            output = torch.clamp(output, 0, 1)
            output = output[0, 0].detach().cpu().numpy()  # (H, W)
            
            # Compute residual.
            residual = mri_img_coil - output
            
            original_coil_list.append(mri_img_coil)
            denoised_coil_list.append(output)
            residual_coil_list.append(residual)
        
        # Stack results for this slice: (num_coils, H, W)
        original_volumes.append(np.stack(original_coil_list, axis=0))
        denoised_volumes.append(np.stack(denoised_coil_list, axis=0))
        residual_volumes.append(np.stack(residual_coil_list, axis=0))
    
    # Convert lists to arrays.
    # Current shapes: (mri_slices, num_coils, H, W)
    original_vol = np.stack(original_volumes, axis=0)
    denoised_vol = np.stack(denoised_volumes, axis=0)
    residual_vol = np.stack(residual_volumes, axis=0)
    
    # Rearrange to (H, W, mri_slices, num_coils)
    # Current shape is (mri_slices, num_coils, H, W), so we use (2,3,0,1)
    original_vol = np.transpose(original_vol, (2,3,0,1)) / data_scale_factor
    denoised_vol = np.transpose(denoised_vol, (2,3,0,1)) / data_scale_factor
    residual_vol = np.transpose(residual_vol, (2,3,0,1)) / data_scale_factor
    
    orig_nii_path = os.path.join(nii_dir, "original_4D.nii")
    denoised_nii_path = os.path.join(nii_dir, "denoised_4D.nii")
    residual_nii_path = os.path.join(nii_dir, "residual_4D.nii")
    nib.save(nib.Nifti1Image(original_vol, affine=np.eye(4)), orig_nii_path)
    nib.save(nib.Nifti1Image(denoised_vol, affine=np.eye(4)), denoised_nii_path)
    nib.save(nib.Nifti1Image(residual_vol, affine=np.eye(4)), residual_nii_path)
    print(f"Saved original volume as {orig_nii_path}")
    print(f"Saved denoised volume as {denoised_nii_path}")
    print(f"Saved residual volume as {residual_nii_path}")

###########################
# Argument Parsing & Main
###########################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference from MAT inputs with per-coil processing. 2-channel input per coil (MRI + noise) and 1-channel output per coil. Output is saved as a 4D NIfTI with shape (H, W, num_slices, num_coils)."
    )
    parser.add_argument('--model_pth', type=str, required=True,
                        help='Path to the trained .pth model file')
    parser.add_argument('--mri_mat', type=str, required=True,
                        help='Path to the MRI k-space .mat file')
    parser.add_argument('--noise_mat', type=str, required=True,
                        help='Path to the pure noise k-space .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device, e.g., cuda or cpu')
    parser.add_argument('--data_scale_factor', type=float, default=5e4,
                        help='Scaling factor applied to both MRI images and noise maps')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_from_mat(args.model_pth, args.mri_mat, args.noise_mat, args.output_folder,
                           device=args.device, data_scale_factor=args.data_scale_factor)

if __name__ == "__main__":
    main()
