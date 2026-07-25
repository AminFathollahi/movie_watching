"""
cifti_io.py
======================
Minimal CIFTI I/O utilities for scripts.

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


def load_named_map(cifti_path, map_name: str) -> np.ndarray:
    """Load one named scalar map from a multi-map CIFTI dscalar file.

    Args:
        cifti_path: str or Path — path to a .dscalar.nii with a ScalarAxis
        map_name: str — label of the map to extract

    Returns:
        (n_grayordinates,) float32

    Raises:
        KeyError: if map_name is not present in the file.
    """
    img = nib.load(str(cifti_path))
    names = list(img.header.get_axis(0).name)
    if map_name not in names:
        raise KeyError(f"Map '{map_name}' not found in {cifti_path}. Available: {names}")
    return img.get_fdata(dtype=np.float32)[names.index(map_name)]


def load_single_map(cifti_path) -> np.ndarray:
    """Load the (only) scalar map from a single-map CIFTI dscalar file.

    Args:
        cifti_path: str or Path — path to a .dscalar.nii with exactly one map

    Returns:
        (n_grayordinates,) float32
    """
    return nib.load(str(cifti_path)).get_fdata(dtype=np.float32)[0]


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


def save_cifti_multimap(data_2d: np.ndarray, map_names: list,
                        template_path: str, output_path: str) -> None:
    """Save multiple cortical maps as a multi-map CIFTI dscalar.nii.

    Args:
        data_2d: (n_maps, n_grayordinates) float32
        map_names: list[str] — one label per map (shown in wb_view)
        template_path: str — path to a reference CIFTI whose BrainModelAxis is reused
        output_path: str — destination path
    """
    bm_axis = get_bm_axis(template_path)
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    arr = data_2d.astype(np.float32)
    img = nib.Cifti2Image(arr, header=header)
    nib.save(img, output_path)


def get_combined_map_names(combined_path) -> list:
    """Return the scalar map names in an existing combined CIFTI dscalar.

    Returns an empty list if the file does not exist or cannot be read.
    Safe to call before the combined file has been created.
    """
    try:
        img = nib.load(str(combined_path))
        ax = img.header.get_axis(0)
        return [ax.name[i] for i in range(img.shape[0])]
    except Exception:
        return []


def _save_cifti_atomic(img, path: str) -> None:
    """Write a nibabel CIFTI image atomically (temp file → rename).

    Prevents partial/corrupt writes if the process is killed mid-write:
    the destination is only replaced once the full write succeeds.
    """
    import os
    import tempfile
    from pathlib import Path
    tmp = Path(path).with_suffix(".tmp.nii")
    try:
        nib.save(img, str(tmp))
        os.replace(str(tmp), path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def merge_into_combined(new_map: np.ndarray, map_name: str,
                         combined_path, template_cifti: str) -> None:
    """Add or overwrite one scalar map in a combined CIFTI dscalar file.

    If *combined_path* does not exist a new file is created.
    If it already contains *map_name* that map is replaced in-place.
    Otherwise the new map is appended.

    Corrupt existing files (readable header, unreadable data) are detected
    and silently replaced rather than crashing. All writes are atomic
    (temp file → rename) so a killed process never leaves a partial file.

    Parameters
    ----------
    new_map        : (n_grayords,) float32
    map_name       : label shown in wb_view
    combined_path  : destination .dscalar.nii (Path or str)
    template_cifti : any CIFTI whose BrainModelAxis is used when creating
                     a new combined file from scratch
    """
    import logging
    import os
    log = logging.getLogger(__name__)

    combined_path = str(combined_path)
    existing = get_combined_map_names(combined_path)

    # Try to load existing data; treat any read failure as a corrupt file.
    loaded_img = None
    loaded_data = None
    if existing:
        try:
            loaded_img  = nib.load(combined_path)
            loaded_data = loaded_img.get_fdata(dtype=np.float32)   # (n_maps, n_verts)
        except Exception as exc:
            log.warning(f"  Combined file unreadable ({exc.__class__.__name__}: {exc}) "
                        f"— discarding and recreating: {os.path.basename(combined_path)}")
            if loaded_img is not None:
                del loaded_img
            try:
                os.unlink(combined_path)
            except OSError:
                pass
            existing     = []
            loaded_img   = None
            loaded_data  = None

    if existing and loaded_data is not None:
        if map_name in existing:
            log.info(f"  Replacing map '{map_name}' in {os.path.basename(combined_path)}")
            loaded_data[existing.index(map_name)] = new_map.astype(np.float32)
            names = existing
        else:
            log.info(f"  Appending map '{map_name}' to {os.path.basename(combined_path)} "
                     f"(existing: {existing})")
            loaded_data = np.vstack([loaded_data,
                                     new_map.reshape(1, -1).astype(np.float32)])
            names = existing + [map_name]
        bm_axis     = loaded_img.header.get_axis(1)
        scalar_axis = nib.cifti2.ScalarAxis(names)
        header      = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
        new_img     = nib.Cifti2Image(loaded_data, header=header)
        del loaded_img, loaded_data                      # release file handle before overwriting
        _save_cifti_atomic(new_img, combined_path)
    else:
        log.info(f"  Creating combined file '{map_name}': {os.path.basename(combined_path)}")
        bm_axis     = get_bm_axis(template_cifti)
        scalar_axis = nib.cifti2.ScalarAxis([map_name])
        header      = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
        new_img     = nib.Cifti2Image(
            new_map.reshape(1, -1).astype(np.float32), header=header)
        _save_cifti_atomic(new_img, combined_path)

    log.info(f"  Combined saved: {os.path.basename(combined_path)}  "
             f"maps={get_combined_map_names(combined_path)}")


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
