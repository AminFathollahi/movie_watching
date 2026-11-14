#!/usr/bin/env python3
"""encoding/diff_maps.py
=========================
Encoding-currency analogue of deprecated/rsa/dummy_diff_maps.py.

For each native-AV model and each variant (split x feature scaling), compares
the intact joint embedding against its dummy conditions (clsav_from_a,
clsav_from_v) on two signals produced by variance_partition.py:
  - joint_r  : map r_j of encoding_pearson_r_{split}_{scaling}_{own tag}_models, the
               held-out Pearson r of the joint-only model.
  - unique_j : r2_avj - r2_av of encoding_r2_{split}_{scaling}_{own tag}_models, the
               held-out R2 gained by adding J to the additive A+V model.

The global temporal scramble (avscramble) is not compared: its embeddings mix
training and held-out clips, and variance_partition.py rejects them.

Per-condition output (both dummy conditions):
  {model}_{cond}/{config}/encoding_diff_{split}_{scaling}_{own tag}_dummy.dscalar.nii
    maps: joint_r_dummy, unique_j_dummy, diff_joint_r (intact - dummy)
Per-model output (max over the dummy conditions):
  {model}/{config}/encoding_diff_{split}_{scaling}_{own tag}_modality_presence.dscalar.nii
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
from shared.cifti_io import load_named_map, merge_into_combined  # noqa: E402
from encoding.shared.layout import fit_stem, group_dir  # noqa: E402
from encoding.shared.tags import own_tag  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = group_dir("cortex")
BIN_SEC = 5
SPLITS = ("loro",)
SCALINGS = ("demean",)

# Same native-AV model roster as deprecated/rsa/run_diff_study.sh's BASE_MODELS.
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


def _stem(model: str, cond: str | None, metric: str, split: str, scaling: str) -> str:
    return str(fit_stem(GROUP_DIR, model if cond is None else f"{model}_{cond}", BIN_SEC, metric, scaling, own_tag(model), split))


def _signals(model: str, cond: str | None, split: str, scaling: str):
    joint_r = _load(Path(_stem(model, cond, "pearson_r", split, scaling) + "_models.dscalar.nii"), "r_j")
    r2_path = Path(_stem(model, cond, "r2", split, scaling) + "_models.dscalar.nii")
    avj, av = _load(r2_path, "r2_avj"), _load(r2_path, "r2_av")
    return joint_r, None if avj is None or av is None else avj - av


def consolidate(split: str, scaling: str) -> None:
    for model in BASE_MODELS:
        tag = f"{split}_{scaling}_{own_tag(model)}"
        intact_r, intact_unique = _signals(model, None, split, scaling)
        if intact_r is None or intact_unique is None:
            log.info(f"[{model} {tag}] SKIP — missing intact maps")
            continue
        unique_by_condition = {}
        for cond in DUMMY_CONDITIONS:
            dummy_r, dummy_unique = _signals(model, cond, split, scaling)
            if dummy_r is None or dummy_unique is None:
                log.info(f"[{model}/{cond} {tag}] SKIP — missing dummy maps")
                continue
            out = _stem(model, cond, "diff", split, scaling) + "_dummy.dscalar.nii"
            template = _stem(model, cond, "pearson_r", split, scaling) + "_models.dscalar.nii"
            merge_into_combined(dummy_r, "joint_r_dummy", out, template)
            merge_into_combined(dummy_unique, "unique_j_dummy", out, template)
            merge_into_combined(intact_r - dummy_r, "diff_joint_r", out, template)
            unique_by_condition[cond] = dummy_unique
            log.info(f"[{model}/{cond} {tag}] mean diff_joint_r={np.mean(intact_r - dummy_r):.4f}")
        if unique_by_condition:
            difference = intact_unique - np.maximum.reduce(list(unique_by_condition.values()))
            out = _stem(model, None, "diff", split, scaling) + "_modality_presence.dscalar.nii"
            merge_into_combined(
                difference, "modality_presence_diff", out,
                _stem(model, None, "r2", split, scaling) + "_models.dscalar.nii",
            )
            log.info(f"[{model} {tag}] mean modality_presence_diff={difference.mean():.4f}")


def main():
    for split, scaling in product(SPLITS, SCALINGS):
        consolidate(split, scaling)


if __name__ == "__main__":
    main()
