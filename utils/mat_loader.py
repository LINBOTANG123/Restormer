#!/usr/bin/env python

import numpy as np
import h5py
import scipy.io as sio


def process_kspace_to_img(kspace_2d):
    """
    Process a 2D k-space slice:
      - ifftshift, 2D IFFT, fftshift, and magnitude extraction.
    Returns a float32 magnitude image.
    """
    # shift zero-frequency component to center
    shifted = np.fft.ifftshift(kspace_2d)
    # 2D inverse FFT to image domain
    img_complex = np.fft.ifft2(shifted)
    # shift back
    img_complex = np.fft.fftshift(img_complex)
    # magnitude
    img_mag = np.abs(img_complex).astype(np.float32)
    return img_mag


def load_mri_data(
    file_path,
    key='image',
    data_format='b1000',
    num_samples_to_load=None,
    gslider_index=0
):
    """
    Load MRI k-space or related data and convert to magnitude images.

    Parameters
    ----------
    file_path : str
        Path to the .mat or HDF5 file.
    key : str
        Dataset key inside the file.
    data_format : {'Hwihun_phantom','b1000','C','gslider'}
        Format of stored data. For 'gslider', expects 6D array
        (dwi, gslider, slice, coil, y, x).
    num_samples_to_load : int or None
        If set, limit number of DWI channels.
    gslider_index : int
        Which gSlider encoding to select when data_format='gslider'.

    Returns
    -------
    mri_img : np.ndarray
        Array of shape (H, W, coils, slices, samples) with float32 magnitudes.
    """
    # 1) Load raw_data
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            raw_data = f[key][()]
    else:
        mat = sio.loadmat(file_path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}")
        raw_data = mat[key]

    # 2) Handle MATLAB complex void dtype
    if raw_data.dtype.kind == 'V':
        fields = raw_data.dtype.names or []
        if set(['real','imag']).issubset(fields):
            raw_data = raw_data['real'] + 1j * raw_data['imag']
        else:
            raw_data = raw_data[fields[0]]

    # 3) Reorder per format
    if data_format == 'Hwihun_phantom':
        data = np.transpose(raw_data, (3, 4, 0, 2, 1))
    elif data_format == 'b1000':
        data = np.transpose(raw_data, (3, 4, 2, 0, 1))
    elif data_format == 'C':
        data = raw_data
    elif data_format == 'gslider':
        if raw_data.ndim != 6:
            raise ValueError(f"Expected 6D gslider data, got {raw_data.shape}")
        n_dwi, n_gs, mri_slices, num_coils, H, W = raw_data.shape
        if not (0 <= gslider_index < n_gs):
            raise ValueError(f"gslider_index out of range [0,{n_gs}), got {gslider_index}")
        sel = raw_data[:, gslider_index, :, :, :, :]
        if num_samples_to_load is not None:
            sel = sel[:num_samples_to_load, ...]
        data = np.transpose(sel, (3, 4, 2, 1, 0))  # (H, W, coils, slices, dwi)  # (H, W, coils, slices, dwi)
    else:
        raise ValueError(f"Unrecognized data_format: {data_format}")

    if data.ndim != 5:
        raise ValueError(f"After reorder expected 5D, got {data.shape}")

    H, W, num_coils, mri_slices, num_samples = data.shape

    # 4) Process each 2D k-space with IFFT to magnitude
    mri_img = np.empty((H, W, num_coils, mri_slices, num_samples), dtype=np.float32)
    for c in range(num_coils):
        for s in range(mri_slices):
            for n in range(num_samples):
                mri_img[:, :, c, s, n] = process_kspace_to_img(data[:, :, c, s, n])

    return mri_img


if __name__ == "__main__":
    import argparse
    import os
    import nibabel as nib

    parser = argparse.ArgumentParser(
        description="Load MRI k-space/gSlider data, convert to magnitudes via IFFT, and save one DWI as 4D NIfTI."
    )
    parser.add_argument("file_path", help="Path to .mat/HDF5 file")
    parser.add_argument("--key", default="image", help="Dataset key in file")
    parser.add_argument(
        "--format", dest="data_format", default="b1000",
        choices=["Hwihun_phantom","b1000","C","gslider"],
        help="Data format; 'gslider' for 6D gSlider data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit number of DWI channels"
    )
    parser.add_argument(
        "--gslider-index", type=int, default=0,
        help="Which gSlider encoding to select"
    )
    args = parser.parse_args()

    print(f"Loading format={args.data_format!r}, key={args.key!r} from {args.file_path!r}")
    vol = load_mri_data(
        file_path=args.file_path,
        key=args.key,
        data_format=args.data_format,
        num_samples_to_load=args.num_samples,
        gslider_index=3,
    )

    print("Loaded volume shape:", vol.shape)
    print("min/max:", vol.min(), "/", vol.max())
    print("mean:", vol.mean())

    # Save first DWI channel as 4D NIfTI
    dwi_idx = 2
    H, W, C, S, N = vol.shape
    vol4d = vol[..., dwi_idx]            # (H,W,coils,slices)
    vol4d = np.transpose(vol4d, (0,1,3,2))  # (H,W,slices,coils)
    affine = np.eye(4)
    base = os.path.splitext(os.path.basename(args.file_path))[0]
    out_fname = f"{base}_dwi{dwi_idx}.nii"
    nib.save(nib.Nifti1Image(vol4d, affine), out_fname)
    print(f"Saved 4D NIfTI: {out_fname}")
