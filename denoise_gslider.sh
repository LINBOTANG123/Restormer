#!/usr/bin/env bash
#
# run_denoise_gslider.sh
# Usage: ./run_denoise_gslider.sh

# Path to your Restormer checkpoint
MODEL_PTH="/home/lin/Research/denoise/Restormer/experiments_2_to_1/GaussianGrayDenoising_RestormerSigma_smooth/models/net_g_400000.pth"

# Input MRI and noise files
MRI_PTH="/home/lin/Research/denoise/data/gslider/gSlider_kspace_coil.mat"
NOISE_MAT="/home/lin/Research/denoise/data/gslider/pf1_noise_v4.mat"

# Output folder
OUTPUT_DIR="./denoised_gslider"

# Other parameters
MRI_FORMAT="gslider"
NOISE_FORMAT="gslider"
SAMPLES=3
SCALE=1e5

# Run the denoising
python inference_maps_2_to_1_b1000_scale.py \
  --model_pth "${MODEL_PTH}" \
  --mri_pth  "${MRI_PTH}" \
  --noise_mat "${NOISE_MAT}" \
  --output_folder "${OUTPUT_DIR}" \
  --mri_format "${MRI_FORMAT}" \
  --noise_format "${NOISE_FORMAT}" \
  --samples ${SAMPLES} \
  --data_scale_factor ${SCALE}
