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
    if data_format == 'simulate':
        import nibabel as nib
        img = nib.load(file_path)
        arr = img.get_fdata().astype(np.float32)       # shape: (X, Y, Z, coils)
        # reorder to (X, Y, coils, Z)
        data = np.transpose(arr, (0, 1, 3, 2))
        # add singleton samples axis → (X, Y, coils, Z, 1)
        return data[..., np.newaxis]
    
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, 'r') as f:
            print("HDF5 keys:", list(f.keys()))
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
        # raw shape: (coils, samples, slices, H, W)
        if num_samples_to_load is not None:
            raw = raw_data[:, :num_samples_to_load, ...]
        # → (H, W, coils, slices, samples)
        data = np.transpose(raw, (3,4,0,2,1))

    elif data_format == 'b1000':
        # raw_data shape: (slices, samples, coils, H, W)
        if num_samples_to_load is not None:
            raw = raw_data[:, :num_samples_to_load, ...]
        else:
            raw = raw_data
        # permute to (H, W, coils, slices, samples)
        data = np.transpose(raw, (3,4,2,0,1))

        # **SPECIAL-CASE**: already in image space → just magnitude & cast
        # data is (H, W, coils, slices, samples)
        return np.abs(data).astype(np.float32)
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
    elif data_format == 'gslider_2':
        if raw_data.ndim != 5:
            raise ValueError(f"Expected 5D gslider data, got {raw_data.shape}")
        num_coil, n_dwi, num_slices, H, W = raw_data.shape
        if num_samples_to_load is not None:
            sel = raw_data[:, :num_samples_to_load, ...]
        data = np.transpose(sel, (3, 4, 0, 2, 1)) /29.0 # (H, W, coils, slices, dwi)  # (H, W, coils, slices, dwi)
        print("Final loaded gslider_2 shape: ", data.shape)
        return np.abs(data).astype(np.float32)
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
