"""
rsa/shared/cifti_io.py
======================
Minimal CIFTI I/O utilities for RSA scripts.

Adapted from cf_modeling/shared/cifti_io.py (Nicholas Hedger, MIT License),
keeping only what the RSA pipeline requires.
"""

import nibabel as nib
import numpy as np


def get_bm_axis(cifti_path: str):
    """Return the BrainModelAxis from a CIFTI file.

    Args:
        cifti_path: str — path to any CIFTI file (.dscalar.nii, .dtseries.nii, etc.)

    Returns:
        nibabel.cifti2.BrainModelAxis
    """
    img = nib.load(cifti_path)
    return img.header.get_axis(1)


def load_cifti_data(cifti_path: str) -> np.ndarray:
    """Load a CIFTI dtseries and return data as (n_vertices, T) float32.

    Args:
        cifti_path: str — path to .dtseries.nii

    Returns:
        (n_vertices, T) float32
    """
    img = nib.load(cifti_path)
    # dtseries on disk: (T, n_vertices) — transpose to (n_vertices, T)
    return img.get_fdata(dtype=np.float32).T


def save_cifti_map(data_1d: np.ndarray, template_path: str, output_path: str,
                   map_name: str = "rsa") -> None:
    """Save a 1-D cortical map as a CIFTI dscalar.nii.

    Args:
        data_1d: (n_grayordinates,) float32 — values for each vertex/grayordinate
        template_path: str — path to a reference CIFTI whose header/axes are reused
        output_path: str — destination path
        map_name: str — label for the scalar map (shown in wb_view)
    """
    bm_axis = get_bm_axis(template_path)
    scalar_axis = nib.cifti2.ScalarAxis([map_name])
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    arr = data_1d.astype(np.float32).reshape(1, -1)
    img = nib.Cifti2Image(arr, header=header)
    nib.save(img, output_path)


def get_cortex_vertex_indices(bm_axis):
    """Return left and right cortical vertex index arrays from a BrainModelAxis.

    Args:
        bm_axis: nibabel.cifti2.BrainModelAxis

    Returns:
        left_indices: (n_L,) int — vertex indices within the left cortex structure
        right_indices: (n_R,) int — vertex indices within the right cortex structure
    """
    left_indices = right_indices = None
    for name, _, model in bm_axis.iter_structures():
        if "CORTEX_LEFT" in name:
            left_indices = model.vertex
        elif "CORTEX_RIGHT" in name:
            right_indices = model.vertex
    if left_indices is None or right_indices is None:
        raise RuntimeError("Could not find left and/or right cortex in BrainModelAxis.")
    return left_indices, right_indices
