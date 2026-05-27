"""
cf_modeling/lib/config_builder.py
===================================
Build a temporary YAML config compatible with vicsompy's MssCf class.

MssCf internalizes several sub-dicts from the YAML in its constructor and
methods.  This module generates a minimal, complete YAML from CLI-level
parameters so that MssCf can be used without modifying or shipping the
vicsompy repo's own config.yml.

Required sections (matched to MssCf.__init__ + method calls)
-------------------------------------------------------------
  dm             — modalities list, with_mean/with_std flags
  patches        — per-modality: n laplacians, roilabs, vars2splice (empty)
  source_regions — paths and naming wildcards for masks/surfaces/lookups
  output         — npy/cifti output naming parameters
  modeling       — himalaya solver parameters
  splicing       — minimal stub (we skip splicing in CfModel)

Usage
-----
    from cf_modeling.lib.config_builder import build_temp_yaml
    yaml_path = build_temp_yaml(
        roi_a="3b", roi_b="V1",
        n_lboe_a=200, n_lboe_b=200,
        masks_dir="/path/to/roi_root/masks",
        surfaces_dir="/path/to/roi_root/surfaces",
        out_dir="/path/to/roi_root/subjects/group_average",
        pcx_sub="hcp_999999_draw_NH",
        surftype="fiducial",
        backend_engine="torch_cuda",
    )
    # yaml_path is a NamedTemporaryFile path — keep it alive for the duration of the run.
    # Delete it (or let the temp dir handle it) when modeling is complete.
"""

import os
import tempfile
from pathlib import Path

import yaml


def build_temp_yaml(
    roi_a: str,
    roi_b: str,
    n_lboe_a: int,
    n_lboe_b: int,
    masks_dir: str,
    surfaces_dir: str,
    out_dir: str,
    pcx_sub: str = "hcp_999999_draw_NH",
    surftype: str = "fiducial",
    backend_engine: str = "torch_cuda",
    solver: str = "random_search",
    n_iter: int = 20,
    n_targets_batch: int = 20000,
    n_alphas_batch: int = 10,
    n_targets_batch_refit: int = 20000,
    alpha_min: int = 1,
    alpha_max: int = 20,
    alpha_vals: int = 20,
    with_mean: bool = True,
    with_std: bool = True,
    tmp_dir: str | None = None,
) -> str:
    """Write a temporary YAML config for vicsompy's MssCf and return its path.

    Parameters
    ----------
    roi_a, roi_b       : str — Glasser ROI short names (e.g. "3b", "V1").
                         These become the modality names in the YAML and are
                         used as keys in the patches sub-dict.
    n_lboe_a, n_lboe_b: int — number of Laplace-Beltrami eigenfunctions per ROI.
                         Must match the actual subsurfaces built by 01_extract_geometry.py.
    masks_dir          : str — absolute path to directory holding CSV mask files
                         named {roi}_{L/R}_mask.csv.  Used as source_region_dir/maskdir.
    surfaces_dir       : str — absolute path to directory holding pre-built Subsurface
                         pickle files named {roi}_subsurface.pickle.
    out_dir            : str — directory where MssCf will write npy outputs
                         (subject.out_csv in the mock adapter).
    pcx_sub            : str — pycortex subject name in the filestore.
    surftype           : str — pycortex surface type ("fiducial", "sphere", …).
    backend_engine     : str — himalaya backend ("torch_cuda", "torch", "numpy", "cupy").
    solver             : str — himalaya MultipleKernelRidgeCV solver.
    n_iter             : int — solver iterations.
    n_targets_batch    : int — number of targets per forward-pass batch (GPU tuning).
    n_alphas_batch     : int — number of alphas per batch.
    n_targets_batch_refit: int — targets per batch in the refit step.
    alpha_min, alpha_max, alpha_vals: alpha grid = np.logspace(alpha_min, alpha_max, alpha_vals).
    with_mean, with_std: StandardScaler flags (match vicsompy config defaults: both True).
    tmp_dir            : str | None — directory for the temp file; None → system default.

    Returns
    -------
    str — absolute path to the written YAML file.
         This file must remain on disk for the lifetime of the MssCf instance.
         Use a caller-managed temp directory or delete explicitly after modeling.
    """
    # Source-region layout expected by MssCf.make_subsurface_from_mask():
    #   source_region_dir/maskdir/{modality}_{hem}_mask.csv
    # Layout expected by make_subsurfaces(force_new=False):
    #   source_region_dir/surfdir/{modality}_subsurface.pickle
    masks_dir     = str(Path(masks_dir).resolve())
    surfaces_dir  = str(Path(surfaces_dir).resolve())

    # MssCf uses source_region_dir as the common root and maskdir/surfdir as relative sub-dirs.
    # Since our masks and surfaces may be in different paths, we set source_region_dir to the
    # parent of masks_dir and use relative paths — or just use absolute paths as sub-dirs.
    # The safest approach: set source_region_dir to "" and use absolute paths in the wildcards.
    # But MssCf joins them with os.path.join, so set source_region_dir to the masks parent:
    masks_parent     = str(Path(masks_dir).parent)
    masks_reldir     = Path(masks_dir).name
    surfaces_parent  = str(Path(surfaces_dir).parent)
    surfaces_reldir  = Path(surfaces_dir).name

    # If both are under the same parent, use that as source_region_dir.
    # Otherwise, use masks_parent (surfaces_dir is injected directly in CfModel).
    source_region_dir = masks_parent

    config = {
        "dm": {
            "modalities": [roi_a, roi_b],
            "with_mean": bool(with_mean),
            "with_std": bool(with_std),
        },

        "patches": {
            roi_a: {
                "laplacians": int(n_lboe_a),
                "vars2splice": [],          # empty — we skip splicing entirely
                "roilabs": [roi_a],
            },
            roi_b: {
                "laplacians": int(n_lboe_b),
                "vars2splice": [],
                "roilabs": [roi_b],
            },
        },

        "source_regions": {
            "source_region_dir": source_region_dir,
            "pcx_sub": pcx_sub,
            "surftype": surftype,
            # Mask wildcard: {modality}_{hem}_mask.csv (built by 00_make_roi_masks.py)
            "maskwcard":  "{modality}_{hem}_mask.csv",
            # Surface wildcard: {modality}_subsurface.pickle (built by 01_extract_geometry.py)
            "surfwcard":  "{modality}_subsurface.pickle",
            # Lookup wildcard: not used (no splicing), but must be a valid key
            "lookupwcard": "{modality}_lookup.csv",
            "maskdir":  masks_reldir,
            "lookupdir": "lookups",         # stub — never read without splicing
            "surfdir":  surfaces_reldir,
            "subsurface_wildcard": "{modality}_subsurface.pickle",
        },

        "output": {
            "varsinframe":          ["train_scores", "test_scores", "best_alphas"],
            "out_cifti_wildcard":   "{param}.nii",
            "out_csv_wildcard":     "params.csv",
            "save_yhat":            False,
            "alphaname":            "best_alphas",
            "train_scorename":      "train_scores",
            "test_scorename":       "test_scores",
            "spliced_paramname":    "spliced_params",
            "npy":                  True,
            "out_npy_wildcard":     "{param}.npy",
        },

        "modeling": {
            "backend_engine":           backend_engine,
            "solver":                   solver,
            "n_iter":                   int(n_iter),
            "n_targets_batch":          int(n_targets_batch),
            "n_alphas_batch":           int(n_alphas_batch),
            "n_targets_batch_refit":    int(n_targets_batch_refit),
            "alpha_min":                int(alpha_min),
            "alpha_max":                int(alpha_max),
            "alpha_vals":               int(alpha_vals),
        },

        # Minimal splicing stub — CfModel never calls splice_lookups(),
        # but MssCf may read these keys in internalize_config if called.
        "splicing": {
            "dot_product":        [False, False],
            "pos_only":           [True,  True],
            "regress_out_mean":   False,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    if tmp_dir is not None:
        os.makedirs(tmp_dir, exist_ok=True)

    # Write to a named temp file (not auto-deleted so it survives the with-block).
    fd, yaml_path = tempfile.mkstemp(
        suffix=f"_cf_config_{roi_a}_{roi_b}.yml",
        dir=tmp_dir,
        prefix="cf_",
    )
    with os.fdopen(fd, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    return yaml_path
