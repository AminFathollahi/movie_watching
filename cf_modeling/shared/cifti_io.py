"""
shared/cifti_io.py
==================
CIFTI I/O utilities and 59k surface helpers.

CiftiHandler: CIFTI loading and surface decomposition (originally from vicsompy/surface.py).
Standalone helpers: CIFTI grayordinate conversion, multi-subject collection, CIFTI saving,
and wb_command resampling (originally from vicsompy_wrappers_59k.py).
"""

import logging
import os
import subprocess
from glob import glob

import nibabel as nb
import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# CiftiHandler — CIFTI loading and surface decomposition
# (originally from vicsompy/surface.py by Nicholas Hedger, MIT License)
# =============================================================================

class CiftiHandler(object):

    """Utility for loading, splitting and saving CIFTI data."""

    def __init__(self, dfile):
        self.dfile = dfile
        self.load_data()

    def load_data(self):
        self.img = nb.load(self.dfile)
        self.header = self.img.header
        self.brain_models = [self.header.get_axis(
            i) for i in range(self.img.ndim)][1]

    def get_data(self):
        self.load_data()
        self.data = self.img.get_fdata(dtype=np.float32)

    def surf_data_from_cifti(self, data, axis, surf_name):
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        for name, data_indices, model in axis.iter_structures():
            if name == surf_name:
                data = data.T[data_indices]
                vtx_indices = model.vertex
                surf_data = np.zeros(
                    (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data
        raise ValueError(f"No structure named {surf_name}")

    def volume_from_cifti(self, data, axis):
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        data = data.T[axis.volume_mask]
        volmask = axis.volume_mask
        vox_indices = tuple(axis.voxel[axis.volume_mask].T)
        vol_data = np.zeros(axis.volume_shape + data.shape[1:], dtype=data.dtype)
        vol_data[vox_indices] = data
        return nb.Nifti1Image(vol_data, axis.affine)

    def decompose_cifti(self, data):
        self.subcortex = self.volume_from_cifti(data, self.brain_models)
        self.surf_left = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        self.surf_right = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")

        if data.ndim == 1:
            self.surface = np.concatenate([self.surf_left, self.surf_right])
        else:
            self.surface = np.vstack([self.surf_left, self.surf_right])

    def decompose_data(self, data):
        subcortex = self.volume_from_cifti(data, self.brain_models)
        surf_left = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        surf_right = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")

        if data.ndim == 1:
            surface = np.concatenate([surf_left, surf_right])
        else:
            surface = np.vstack([surf_left, surf_right])

        return surface, subcortex

    def save_cii(self, data, filename):
        nb.Cifti2Image(data, header=self.img.header,
                       nifti_header=self.img.nifti_header).to_filename(filename)

    def save_subvol(self, data, filename):
        nb.save(data, filename)


# =============================================================================
# 59k CIFTI helpers (originally from vicsompy_wrappers_59k.py)
# =============================================================================

def _extract_grayords(surface_118k: np.ndarray, bm_axis) -> np.ndarray:
    """Map (118584, T) bilateral surface → (T, 108441) CIFTI grayordinates.

    Parameters
    ----------
    surface_118k : (118584, T) — full bilateral surface, L rows 0–59291,
                   R rows 59292–118583, medial wall = 0
    bm_axis      : nibabel.cifti2.BrainModelAxis

    Returns
    -------
    (T, 108441) float32
    """
    left_verts = right_verts = None
    for name, _, model in bm_axis.iter_structures():
        if "CORTEX_LEFT" in name:
            left_verts = model.vertex
        elif "CORTEX_RIGHT" in name:
            right_verts = model.vertex
    n_L, n_R = len(left_verts), len(right_verts)
    T = surface_118k.shape[1]
    # Pre-allocate final output; copy one hemisphere at a time to avoid
    # holding rows list + vstack + astype simultaneously (~2 GB saved vs vstack approach)
    out = np.empty((T, n_L + n_R), dtype=np.float32)
    tmp = surface_118k[left_verts, :]
    out[:, :n_L] = tmp.T
    del tmp
    tmp = surface_118k[59292 + right_verts, :]
    out[:, n_L:] = tmp.T
    del tmp
    return out


def cifti_59k_to_surface(arr_108k: np.ndarray, bm_axis) -> np.ndarray:
    """Map (108441,) CIFTI grayordinates → (118584,) surface array with medial wall = NaN.

    Parameters
    ----------
    arr_108k : (108441,) float32
    bm_axis  : nibabel.cifti2.BrainModelAxis

    Returns
    -------
    (118584,) float32 — bilateral surface; medial wall = NaN
    """
    surf = np.full(118584, np.nan, dtype=np.float32)
    row = 0
    for name, sl, model in bm_axis.iter_structures():
        n = sl.stop - sl.start
        if "CORTEX_LEFT" in name:
            surf[model.vertex] = arr_108k[row:row + n]
        elif "CORTEX_RIGHT" in name:
            surf[59292 + model.vertex] = arr_108k[row:row + n]
        row += n
    return surf


def collect_maps(subjects_dir: str, map_name: str, min_subjects: int = 1) -> np.ndarray:
    """Load map_name.npy from all completed subject directories.

    Parameters
    ----------
    subjects_dir  : str — path to directory containing per-subject subdirs
    map_name      : str — e.g. "R2_TA2_nc"
    min_subjects  : int — raise if fewer subjects found

    Returns
    -------
    (n_subjects, 108441) float32
    """
    sub_dirs = sorted(glob(os.path.join(subjects_dir, "*")))
    arrays, missing = [], []
    for sd in sub_dirs:
        path = os.path.join(sd, f"{map_name}.npy")
        if os.path.exists(path):
            arrays.append(np.load(path))
        else:
            missing.append(os.path.basename(sd))
    if missing:
        log.warning(f"  {map_name}: missing for {len(missing)} subjects "
                    f"(e.g. {missing[:3]})")
    if len(arrays) < min_subjects:
        raise RuntimeError(
            f"Only {len(arrays)} subjects have {map_name}.npy — "
            f"need at least {min_subjects}."
        )
    log.info(f"  {map_name}: loaded {len(arrays)} subjects")
    return np.stack(arrays, axis=0)


def get_59k_bm_axis(subjects_dir: str, cifti_raw_dir: str):
    """Return cortex-only BrainModelAxis (108441 grayordinates) from a reference CIFTI.

    Parameters
    ----------
    subjects_dir  : str — directory with per-subject output subdirs
    cifti_raw_dir : str — directory with raw CIFTI dtseries.nii files
    """
    for sd in sorted(glob(os.path.join(subjects_dir, "*"))):
        sub_id = os.path.basename(sd)
        path = os.path.join(cifti_raw_dir,
                            f"{sub_id}_tfMRI_MOVIE1_7T_AP_Atlas_1.6mm_hp2000_clean.dtseries.nii")
        if not os.path.exists(path):
            continue
        img = nb.load(path)
        bm = img.header.get_axis(1)
        left_sl = right_sl = None
        for name, sl, _ in bm.iter_structures():
            if "CORTEX_LEFT" in name:
                left_sl = sl
            elif "CORTEX_RIGHT" in name:
                right_sl = sl
        if left_sl is not None and right_sl is not None:
            log.info(f"  BrainModelAxis template: {os.path.basename(path)}")
            return bm[left_sl.start:right_sl.stop]
    raise RuntimeError(
        "No reference CIFTI found. Run 02_prep_subject_cifti.py --subject <id> first."
    )


def save_59k_cifti(arr: np.ndarray, name: str, bm_axis, out_dir: str) -> str:
    """Save a (108441,) array as a 59k CIFTI dscalar.nii.

    Parameters
    ----------
    arr     : (108441,) float32
    name    : str — scalar map name and output filename stem
    bm_axis : nibabel.cifti2.BrainModelAxis
    out_dir : str — output directory

    Returns
    -------
    str — path to saved file
    """
    arr_f32 = arr.astype(np.float32).reshape(1, -1)
    scalar_ax = nb.cifti2.ScalarAxis([name])
    header = nb.cifti2.Cifti2Header.from_axes((scalar_ax, bm_axis))
    img = nb.Cifti2Image(arr_f32, header=header)
    path = os.path.join(out_dir, f"{name}.59k.dscalar.nii")
    nb.save(img, path)
    log.info(f"  Saved: {path}")
    return path


def resample_to_32k(cifti_59k_path: str, out_name: str, out_dir: str,
                    template_32k: str,
                    sphere_59k_l: str, sphere_59k_r: str,
                    sphere_32k_l: str, sphere_32k_r: str) -> str | None:
    """Resample a 59k dscalar.nii to 32k via wb_command -cifti-resample.

    Returns
    -------
    str or None — path to 32k file, or None if resampling failed
    """
    out_path = os.path.join(out_dir, f"{out_name}.32k.dscalar.nii")
    for f in [template_32k, sphere_59k_l, sphere_59k_r, sphere_32k_l, sphere_32k_r]:
        if not os.path.exists(f):
            log.warning(f"  Skipping 32k resample — missing: {f}")
            return None
    cmd = [
        "wb_command", "-cifti-resample",
        cifti_59k_path, "COLUMN",
        template_32k, "COLUMN",
        "BARYCENTRIC", "BARYCENTRIC",
        out_path,
        "-left-spheres",  sphere_59k_l, sphere_32k_l,
        "-right-spheres", sphere_59k_r, sphere_32k_r,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  wb_command failed: {result.stderr}")
        return None
    log.info(f"  Resampled → {os.path.basename(out_path)}")
    return out_path
