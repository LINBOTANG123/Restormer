#!/usr/bin/env python
# -------------------------------------------------------------
#  Histogram + metrics: |residual| vs. σ‑map  (brain mask only)
#  - Fills holes in mask
#  - Repeats noise slices (34 -> 170) 5×
#  - Auto-scales σ‑map to Ours (99th pct)
#  - Keeps PDF bins with density ≥ 1e-3
#  - Saves masked residual volumes
# -------------------------------------------------------------
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from utils.noise_loader import load_noise_data

# ---------- USER SETTINGS ----------
residual_paths = {
    'Ours'                 : '/home/lin/Research/denoise/results/gslider_new_human/ours_residual.nii',
    'Pretrained_Restormer' : '/home/lin/Research/denoise/results/gslider_new_human/restormer_(pretrained)_residual.nii',
    'MPPCA'                : '/home/lin/Research/denoise/results/gslider_new_human/mppca_residual.nii',
    'NORDIC'               : '/home/lin/Research/denoise/results/gslider_new_human/nordic_residual.nii.gz',
}
noise_mat_path   = '/home/lin/Research/denoise/data/new_human/pf1_noise_v4.mat'
noise_key        = 'kimage'
noise_format     = 'gslider'
mask_path      = '/home/lin/Research/denoise/results/gslider_new_human/new_3D_slicer_mask_by_hand.nii'

nbins            = 200
match_percentile = 99
pdf_floor        = 1e-13
masked_out_dir   = './masked_residuals'
# -------------------------------------------------------------

# 1) LOAD + FIX MASK (fill holes) ---------------------------------
mask_img = nib.load(mask_path)
mask_raw = mask_img.get_fdata() > 0
mask     = ndi.binary_fill_holes(mask_raw).astype(bool)

# 2) LOAD RESIDUALS & SAVE MASKED VOLUMES -------------------------
os.makedirs(masked_out_dir, exist_ok=True)
res_vals = {}

print('\n=== Residuals (brain mask applied) ===')
for lab, pth in residual_paths.items():
    img_obj = nib.load(pth)
    data    = img_obj.get_fdata().squeeze()   # remove any singleton dims
    if data.shape != mask.shape:
        raise ValueError(f"{lab} shape {data.shape} != mask {mask.shape}")

    vals = np.abs(data)[mask]
    vals = vals[np.isfinite(vals)]
    res_vals[lab] = vals
    print(f'{lab:22s}  mean={vals.mean():.3e}  σ={vals.std():.3e}')

    # Save masked residual volume
    masked_vol = np.zeros_like(data)
    masked_vol[mask] = np.abs(data)[mask]
    out_path = os.path.join(masked_out_dir, f'{lab.replace(" ", "_")}_masked.nii.gz')
    nib.save(nib.Nifti1Image(masked_vol.astype(np.float32), img_obj.affine), out_path)
    print(f'  -> saved masked NIfTI: {out_path}')

ours = res_vals['Ours']

# 3) LOAD NOISE & REPEAT SLICES -----------------------------------
noise_4d = load_noise_data(noise_mat_path, key=noise_key, data_format=noise_format)  # (H,W,C,34)
noise_rss = np.sqrt(np.sum(noise_4d**2, axis=2))  # (H,W,34)

if noise_rss.shape[-1] * 5 != mask.shape[-1]:
    raise ValueError("Expected noise slices *5 to match residual depth (34*5=170).")

noise_rss_170 = np.repeat(noise_rss, repeats=5, axis=2)

if noise_rss_170.shape != mask.shape:
    raise ValueError(f"Noise shape {noise_rss_170.shape} != mask {mask.shape}")

noise_raw = noise_rss_170[mask]
noise_raw = noise_raw[np.isfinite(noise_raw)]

# Auto-scale σ-map to match 'Ours'
gamma = np.percentile(ours, match_percentile) / (np.percentile(noise_raw, match_percentile) + 1e-12)
print("gamma: ", gamma)
noise_vals = noise_raw * gamma
res_vals['Pure Noise'] = noise_vals
print(f'\n[Pure Noise] mean={noise_vals.mean():.3e}  σ={noise_vals.std():.3e}  gamma={gamma:.3f}')

# 4) COMMON BINS ---------------------------------------------------
all_vals = np.concatenate(list(res_vals.values()))
bins   = np.linspace(0, all_vals.max(), nbins+1)
cent   = 0.5 * (bins[:-1] + bins[1:])

# 5) HISTOGRAMS + FLOOR -------------------------------------------
hist = {lab: np.histogram(v, bins=bins, density=True)[0] + 1e-12
        for lab, v in res_vals.items()}

keep = hist['Pure Noise'] >= pdf_floor
trim_centers = cent[keep]
for lab in hist:
    hist[lab] = hist[lab][keep]

# 6) METRICS -------------------------------------------------------
print('\n=== Metrics vs. σ‑map, bins with PDF ≥ 1e‑3 ===')
noise_hist = hist['Pure Noise']
for lab, v in res_vals.items():
    if lab == 'Pure Noise':
        continue
    h    = hist[lab]
    kl   = entropy(h, noise_hist)
    w1   = wasserstein_distance(v, noise_vals)
    ks_p = ks_2samp(v, noise_vals).pvalue
    print(f'{lab:22s}  KL={kl:.4f}  W₁={w1:.3e}  KS‑p={ks_p:.3e}')

# 7) PLOT ----------------------------------------------------------
plt.figure(figsize=(12, 8))
for lab in hist:
    if lab == 'Pure Noise':
        plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='--', color='red',   label=lab)
    elif lab == 'NORDIC':
        plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='-',  color='purple', label=lab)
    else:
        plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='-',  label=lab)

# plt.yscale('log')
plt.ylim(pdf_floor, None)
plt.xlabel('Intensity Value')
plt.ylabel('PDF')
plt.title('Residual Histogram')
plt.grid(True, linestyle=':', lw=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('residual_vs_noise_pdf_brainonly.png', dpi=300)
plt.show()
print('\nSaved → residual_vs_noise_pdf_brainonly.png')
