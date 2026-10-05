#!/usr/bin/env python3
"""encoding/diff_maps.py
=========================
Encoding-currency analogue of rsa/dummy_diff_maps.py.

For each native-AV model and each variant (split x feature scaling), compares
the intact joint embedding against its dummy conditions (clsav_from_a,
clsav_from_v) on two signals produced by variance_partition.py:
  - joint_r  : map r_j of encoding_pearson_r_{split}_{scaling}_unimodal_own_models, the
               held-out Pearson r of the joint-only model.
  - unique_j : map unique_j of encoding_r2_{split}_{scaling}_unimodal_own_partition, the
               held-out R2 gained by adding J to the additive A+V model.

The global temporal scramble (avscramble) is not compared: its embeddings mix
training and held-out clips, and variance_partition.py rejects them.

Per-condition output (both dummy conditions):
  {model}_{cond}/{config}/encoding_diff_{split}_{scaling}_unimodal_own_dummy.dscalar.nii
    maps: joint_r_dummy, unique_j_dummy, diff_joint_r (intact - dummy)
Per-model output (max over the dummy conditions):
  {model}/{config}/encoding_diff_{split}_{scaling}_unimodal_own_modality_presence.dscalar.nii
    map: modality_presence_diff = unique_j_intact - max(unique_j_dummy)

Run with:
    conda run -n movie python encoding/diff_maps.py
"""
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import load_named_map, merge_into_combined  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/encoding/group_average")
CONFIG = "delay5s_bin5s_skip5s"
SPLITS = ("loro",)
SCALINGS = ("demean",)

# Same native-AV model roster as rsa/run_diff_study.sh's BASE_MODELS.
BASE_MODELS = [
    "pe-av-small-16-frame",
    "nemotron_layer9_mp", "nemotron_layer18_mp", "nemotron_layer27_mp", "nemotron_layer36_mp",
    "omni3b_layer9_mp", "omni3b_layer18_mp", "omni3b_layer27_mp",
    "topoomni_layer9_mp", "topoomni_layer18_mp", "topoomni_layer27_mp",
    "topoomni_layer9_sheet_mp", "topoomni_layer18_sheet_mp", "topoomni_layer27_sheet_mp",
]
DUMMY_CONDITIONS = ["clsav_from_a", "clsav_from_v"]


def _load(path: Path, map_name: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return load_named_map(path, map_name)
    except KeyError:
        return None


def _config_dir(model: str, cond: str | None = None) -> Path:
    return GROUP_DIR / (model if cond is None else f"{model}_{cond}") / CONFIG


def _signals(config_dir: Path, tag: str):
    joint_r = _load(config_dir / f"encoding_pearson_r_{tag}_models.dscalar.nii", "r_j")
    unique_j = _load(config_dir / f"encoding_r2_{tag}_partition.dscalar.nii", "unique_j")
    return joint_r, unique_j


def consolidate(split: str, scaling: str) -> None:
    tag = f"{split}_{scaling}_unimodal_own"
    for model in BASE_MODELS:
        intact_dir = _config_dir(model)
        intact_r, intact_unique = _signals(intact_dir, tag)
        if intact_r is None or intact_unique is None:
            log.info(f"[{model} {tag}] SKIP — missing intact maps")
            continue
        unique_by_condition = {}
        for cond in DUMMY_CONDITIONS:
            cond_dir = _config_dir(model, cond)
            dummy_r, dummy_unique = _signals(cond_dir, tag)
            if dummy_r is None or dummy_unique is None:
                log.info(f"[{model}/{cond} {tag}] SKIP — missing dummy maps")
                continue
            out = cond_dir / f"encoding_diff_{tag}_dummy.dscalar.nii"
            template = str(cond_dir / f"encoding_pearson_r_{tag}_models.dscalar.nii")
            merge_into_combined(dummy_r, "joint_r_dummy", out, template)
            merge_into_combined(dummy_unique, "unique_j_dummy", out, template)
            merge_into_combined(intact_r - dummy_r, "diff_joint_r", out, template)
            unique_by_condition[cond] = dummy_unique
            log.info(f"[{model}/{cond} {tag}] mean diff_joint_r={np.mean(intact_r - dummy_r):.4f}")
        if unique_by_condition:
            difference = intact_unique - np.maximum.reduce(list(unique_by_condition.values()))
            out = intact_dir / f"encoding_diff_{tag}_modality_presence.dscalar.nii"
            merge_into_combined(
                difference, "modality_presence_diff", out,
                str(intact_dir / f"encoding_r2_{tag}_partition.dscalar.nii"),
            )
            log.info(f"[{model} {tag}] mean modality_presence_diff={difference.mean():.4f}")


def main():
    for split, scaling in product(SPLITS, SCALINGS):
        consolidate(split, scaling)


if __name__ == "__main__":
    main()
