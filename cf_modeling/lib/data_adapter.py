"""
cf_modeling/lib/data_adapter.py
=================================
Data-space conversion between CIFTI grayordinate space and full-sphere
bilateral surface space.

Background
----------
HCP 59k_fs_LR CIFTI files store only non-medial-wall grayordinates:
  - 108441 grayordinates total (54216 L + 54225 R).
  - Full sphere has 59292 vertices per hemisphere (118584 bilateral), including
    the medial wall.

Vicsompy's Subsurface vertex indices (subsurface_verts_L/R) are in full-sphere
space, where verts_R already has the +59292 left-hemisphere offset applied.

Vicsompy's CiftiHandler.decompose_data() reconstructs full-sphere surface arrays
before data is passed to MssCf.make_dm() and test_xval().  That function:
  1. Extracts the data slice for each structure from the CIFTI grayordinates.
  2. Maps grayordinate values back to their sphere vertex positions
     (surf_data[vtx_indices] = data_slice), placing zeros at medial-wall positions.
  3. Stacks left (59292, T) and right (59292, T) → (118584, T).

This module provides:
  grayord_to_sphere_space  — convert (n_grayord, T) → (118584, T).
  sphere_to_grayord_space  — inverse: (118584, T) → (n_grayord, T).
  build_sphere_lut         — pre-compute the mapping for repeated use.

Mathematical note
-----------------
The medial-wall positions in sphere-space are filled with zeros.  When
MssCf.make_roi_data() indexes data[subsurface_verts, :], it accesses only
the ROI vertices, none of which are in the medial wall (vicsompy's ROI masks
are Boolean arrays over the full sphere, with False at medial-wall positions).
The zeros at medial-wall positions are therefore never used.
"""

import numpy as np

# HCP 59k_fs_LR standard: 59292 vertices per hemisphere in the full sphere.
N_VERTS_PER_HEM = 59292
N_VERTS_BILATERAL = 2 * N_VERTS_PER_HEM  # 118584


def build_sphere_lut(bm_axis, n_verts_per_hem: int = N_VERTS_PER_HEM) -> np.ndarray:
    """Pre-compute a grayordinate-row → sphere-vertex index lookup table.

    Returns
    -------
    lut : (n_grayord,) int32
        lut[grayord_row] = sphere_vertex_index (right-hem offset by n_verts_per_hem).
    """
    n_grayord = sum(
        len(model.vertex)
        for _, _, model in bm_axis.iter_structures()
        if "CORTEX" in _
    )
    lut = np.empty(n_grayord, dtype=np.int32)
    pos = 0
    for name, _, model in bm_axis.iter_structures():
        n = len(model.vertex)
        if "CORTEX_LEFT" in name:
            lut[pos : pos + n] = model.vertex                     # 0..59291
        elif "CORTEX_RIGHT" in name:
            lut[pos : pos + n] = model.vertex + n_verts_per_hem   # 59292..118583
        else:
            pos += n
            continue
        pos += n
    return lut


def grayord_to_sphere_space(
    grayord_data: np.ndarray,
    bm_axis,
    n_verts_per_hem: int = N_VERTS_PER_HEM,
) -> np.ndarray:
    """Map CIFTI grayordinate data to full-sphere bilateral surface space.

    Equivalent to vicsompy's CiftiHandler.decompose_data() surface component.
    Medial-wall positions are filled with 0.

    Parameters
    ----------
    grayord_data   : (n_grayord, T) float32 — CIFTI cortical grayordinate data.
                     n_grayord = 108441 for the HCP 59k_fs_LR cortex-only CIFTI.
    bm_axis        : nibabel BrainModelAxis from the CIFTI header.
    n_verts_per_hem: int — sphere vertices per hemisphere (59292 for 59k_fs_LR).

    Returns
    -------
    sphere_data : (2 * n_verts_per_hem, T) float32
        Left hemisphere at rows 0..n_verts_per_hem-1,
        right hemisphere at rows n_verts_per_hem..2*n_verts_per_hem-1.
        Zeros at medial-wall positions.
    """
    n_bilateral = 2 * n_verts_per_hem
    if grayord_data.ndim == 1:
        sphere_data = np.zeros(n_bilateral, dtype=grayord_data.dtype)
    else:
        sphere_data = np.zeros((n_bilateral, grayord_data.shape[1]), dtype=grayord_data.dtype)

    pos = 0
    for name, _, model in bm_axis.iter_structures():
        n = len(model.vertex)
        if "CORTEX_LEFT" in name:
            sphere_data[model.vertex] = grayord_data[pos : pos + n]
        elif "CORTEX_RIGHT" in name:
            sphere_data[n_verts_per_hem + model.vertex] = grayord_data[pos : pos + n]
        else:
            pos += n
            continue
        pos += n

    return sphere_data


def sphere_to_grayord_space(
    sphere_data: np.ndarray,
    bm_axis,
    n_verts_per_hem: int = N_VERTS_PER_HEM,
) -> np.ndarray:
    """Extract CIFTI grayordinate data from full-sphere bilateral surface data.

    Inverse of grayord_to_sphere_space.

    Parameters
    ----------
    sphere_data    : (2 * n_verts_per_hem, T) float32 or (2 * n_verts_per_hem,)
    bm_axis        : nibabel BrainModelAxis
    n_verts_per_hem: int

    Returns
    -------
    grayord_data : (n_grayord, T) float32 or (n_grayord,)
    """
    n_grayord = sum(
        len(model.vertex)
        for name, _, model in bm_axis.iter_structures()
        if "CORTEX" in name
    )

    if sphere_data.ndim == 1:
        grayord_data = np.zeros(n_grayord, dtype=sphere_data.dtype)
    else:
        grayord_data = np.zeros((n_grayord, sphere_data.shape[1]), dtype=sphere_data.dtype)

    pos = 0
    for name, _, model in bm_axis.iter_structures():
        n = len(model.vertex)
        if "CORTEX_LEFT" in name:
            grayord_data[pos : pos + n] = sphere_data[model.vertex]
        elif "CORTEX_RIGHT" in name:
            grayord_data[pos : pos + n] = sphere_data[n_verts_per_hem + model.vertex]
        else:
            pos += n
            continue
        pos += n

    return grayord_data
