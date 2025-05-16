#!/bin/bash
#SBATCH --job-name=fractals
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fractals_%j.out
#SBATCH --error=logs/fractals_%j.err

# load your shells’ config and activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate restomer

# ensure directories exist
mkdir -p output logs

# run the fractal‐generation script
# assume your Python code is saved as `generate_fractals.py` in the same folder
python fractal.py
