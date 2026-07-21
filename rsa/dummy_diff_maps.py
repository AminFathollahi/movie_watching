#!/usr/bin/env python3
"""rsa/dummy_diff_maps.py
==========================
Dummy-modality analogue of rsa/scramble_diff_maps.py: consolidates plain RSA
(dummy-modality) and its intact-minus-dummy diff into one CIFTI per
clsav_from_{a,v} model config, testing modality-presence selectivity (does a
model's AV signal depend on both real modalities being present, vs. one real
modality + a fixed content-free placeholder for the other) -- the
whole-embedding companion to the Ward's-linkage localizer's --design dummy
test (rsa/topoomni_av_separability_localizer.py), which asks the same
question at the level of a functionally-defined cluster of units instead of
the whole embedding.

Inputs (all pre-computed):
  - plain RSA, dummy:   {GROUP_DIR}/{model}_clsav_from_{a,v}_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - plain RSA, intact:  {GROUP_DIR}/{model}_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - partial RSA, dummy (integration contrast rerun on dummy-modality
    embeddings) -- OPTIONAL, included only if already computed:
        {GROUP_DIR}/{model}_clsav_from_{a,v}_av_INTEGRATION/k100_delay5s_bin5s_skip5s_spearman/integration_partial_r_searchlight.dscalar.nii

Output: {GROUP_DIR}/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii
  maps: plain_rsa_dummy, diff_intact_minus_dummy_plain_rsa
        (+ partial_rsa_dummy if the optional INTEGRATION rerun exists)

Run with:
    conda run -n movie python rsa/dummy_diff_maps.py
"""
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import merge_into_combined  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average")
CONFIG = "k100_delay5s_bin5s_skip5s_spearman"

# Same 4 model families requested for the scramble diff-study, at every layer
# already covered there (rsa/scramble_diff_maps.py's MODELS list).
MODELS = [
    "pe-av-small-16-frame",
    "omni3b_layer9_mp", "omni3b_layer18_mp", "omni3b_layer27_mp",
    "omni3b_layer9_lt", "omni3b_layer18_lt", "omni3b_layer27_lt",
    "topoomni_layer9_mp", "topoomni_layer18_mp", "topoomni_layer27_mp",
    "topoomni_layer9_lt", "topoomni_layer18_lt", "topoomni_layer27_lt",
    "nemotron_layer9_mp", "nemotron_layer18_mp", "nemotron_layer27_mp", "nemotron_layer36_mp",
    "nemotron_layer9_lt", "nemotron_layer18_lt", "nemotron_layer27_lt",
    "nemotron_layer36_lt",
]
DUMMY_CONDITIONS = ["clsav_from_a", "clsav_from_v"]


def load_map(cifti_path: Path, map_name: str) -> np.ndarray:
    img = nib.load(str(cifti_path))
    names = list(img.header.get_axis(0).name)
    idx = names.index(map_name)
    return img.get_fdata(dtype=np.float32)[idx]


def main():
    done, skipped = [], []
    for model in MODELS:
        intact_path = GROUP_DIR / f"{model}_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        if not intact_path.exists():
            log.info(f"[{model}] SKIP — missing intact RSA: {intact_path}")
            skipped.append(model)
            continue
        plain_intact = load_map(intact_path, "searchlight_spearman_rho")

        for cond in DUMMY_CONDITIONS:
            dummy_dir = f"{model}_{cond}_av"
            dummy_path = GROUP_DIR / dummy_dir / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
            if not dummy_path.exists():
                log.info(f"[{model}/{cond}] SKIP — missing dummy RSA: {dummy_path}")
                skipped.append(f"{model}/{cond}")
                continue

            plain_dummy = load_map(dummy_path, "searchlight_spearman_rho")
            diff = plain_intact - plain_dummy

            out_path = GROUP_DIR / dummy_dir / "dummy_consolidated_maps.dscalar.nii"
            merge_into_combined(plain_dummy, "plain_rsa_dummy", out_path, str(dummy_path))
            merge_into_combined(diff, "diff_intact_minus_dummy_plain_rsa", out_path, str(dummy_path))

            partial_dummy_path = (GROUP_DIR / f"{dummy_dir}_INTEGRATION" / CONFIG
                                   / "integration_partial_r_searchlight.dscalar.nii")
            if partial_dummy_path.exists():
                partial_dummy = nib.load(str(partial_dummy_path)).get_fdata(dtype=np.float32)[0]
                merge_into_combined(partial_dummy, "partial_rsa_dummy", out_path, str(dummy_path))

            log.info(f"[{model}/{cond}] DONE — mean plain_dummy={plain_dummy.mean():.4f}  "
                     f"mean diff={diff.mean():.4f}  max diff={diff.max():.4f}")
            done.append(f"{model}/{cond}")

    log.info(f"\nConsolidated {len(done)}/{len(MODELS) * len(DUMMY_CONDITIONS)} model/condition pairs. "
             f"Skipped (not yet ready): {skipped}")


if __name__ == "__main__":
    main()
