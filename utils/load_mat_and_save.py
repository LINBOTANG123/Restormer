#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import h5py


def _to_numpy_complex(arr: np.ndarray) -> np.ndarray:
    """
    Handle MATLAB v7.3 complex storage where dtype is a compound with fields 'real'/'imag'.
    Otherwise return the array as-is.
    """
    if np.iscomplexobj(arr):
        return arr
    if hasattr(arr.dtype, "names") and arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
        return arr["real"] + 1j * arr["imag"]
    return arr


def load_mat_array(file_path: str, key: str) -> np.ndarray:
    """
    Load `key` from a .mat that could be classic MAT (<=v7) or HDF5 (v7.3).
    Returns a NumPy ndarray.
    """
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, "r") as f:
            print("HDF5 keys:", list(f.keys()))
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            data = f[key][()]
    else:
        mat = sio.loadmat(file_path, squeeze_me=False, struct_as_record=False)
        # Helpful for debugging:
        print("MAT keys:", [k for k in mat.keys() if not k.startswith("__")])
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}")
        data = mat[key]

    data = _to_numpy_complex(np.asarray(data))
    return data


def main():
    ap = argparse.ArgumentParser(description="Reorder MRI MAT: select first DWI and transpose to (coils, dwi, slices, H, W).")
    ap.add_argument("--in_mat", required=True, help="Path to input .mat")
    ap.add_argument("--key", required=True, help="Variable name inside the .mat to load/save")
    ap.add_argument("--out_mat", default=None, help="Path to output .mat (default: <in>_coils_first.mat)")
    args = ap.parse_args()

    in_path = args.in_mat
    out_path = args.out_mat or str(Path(in_path).with_name(Path(in_path).stem + "_coils_first.mat"))

    # 1) Load
    arr = load_mat_array(in_path, args.key)
    print("Loaded array shape:", arr.shape)

    # Expecting (dwi, slices, coils, H, W)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D array, got {arr.ndim}D with shape {arr.shape}")

    dwi, slices, coils, H, W = arr.shape
    print(f"Interpreted dims -> DWI:{dwi}, Slices:{slices}, Coils:{coils}, H:{H}, W:{W}")

    if dwi < 1:
        raise ValueError("DWI dimension is empty; cannot select first sample.")

    # 2) Select first DWI (keep the dimension as size-1)
    first_dwi = arr[4:5, ...]  # shape: (1, slices, coils, H, W)
    print("After selecting first DWI:", first_dwi.shape)

    # 3) Transpose to (coils, dwi, slices, H, W)
    out_arr = np.transpose(first_dwi, (2, 0, 1, 3, 4))  # (coils, dwi(=1), slices, H, W)
    print("After transpose (coils, dwi, slices, H, W):", out_arr.shape)

    # 4) Save back to .mat with the same key
    sio.savemat(out_path, {args.key: out_arr})
    print(f"Saved to {out_path} with key '{args.key}'.")


if __name__ == "__main__":
    main()
