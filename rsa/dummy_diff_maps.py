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
  - partial RSA, intact (the same integration contrast on the intact model):
        {GROUP_DIR}/{model}_av_INTEGRATION/k100_delay5s_bin5s_skip5s_spearman/integration_partial_r_searchlight.dscalar.nii

Per-condition output: {GROUP_DIR}/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii
  maps: plain_rsa_dummy, diff_intact_minus_dummy_plain_rsa
        (+ partial_rsa_dummy if the optional INTEGRATION rerun exists)

Per-MODEL output (one map, combining both dummy conditions via max() since
there are two of them vs. scramble's one):
  {GROUP_DIR}/{model}_av/dummy_partial_diff_maps.dscalar.nii
  map: modality_presence_diff
     = partial_rsa_intact - max(partial_rsa_dummy_from_a, partial_rsa_dummy_from_v)
  Only written once partial_rsa_intact and AT LEAST ONE dummy condition's
  partial rerun exist (max() over whichever are available).

  Deliberately NOT named "binding" like scramble_diff_maps.py's equivalent
  map: scramble tests whether fusion needs correct TEMPORAL pairing (a
  binding/synchrony question), dummy tests whether fusion needs BOTH real
  modalities present at all (a modality-presence question) -- same
  intact-minus-integration-contrast arithmetic, different manipulation, so
  it gets its own name instead of overloading "binding" for something that
  isn't about binding.

Expectation (AV-integration hypothesis): in true integration regions, this
map (and its scramble counterpart, "binding", in scramble_diff_maps.py)
should be positive and large -- a real modality + placeholder weakens
partial alignment relative to both real modalities present; non-integration regions
should stay near zero.

Run with:
    conda run -n movie python rsa/dummy_diff_maps.py
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import load_named_map, load_single_map, merge_into_combined  # noqa: E402
from rsa.shared.naming import DEFAULT_MODEL_NORM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average")
CONFIG = f"k100_delay5s_bin5s_skip5s_spearman_{os.environ.get('MODEL_NORM', DEFAULT_MODEL_NORM)}"

# Same model families and layers as rsa/scramble_diff_maps.py's MODELS list.
MODELS = [
    "pe-av-small-16-frame",
    "omni3b_layer9_mp", "omni3b_layer18_mp", "omni3b_layer27_mp",
    "omni3b_layer9_lt", "omni3b_layer18_lt", "omni3b_layer27_lt",
    "topoomni_layer9_mp", "topoomni_layer18_mp", "topoomni_layer27_mp",
    "topoomni_layer9_lt", "topoomni_layer18_lt", "topoomni_layer27_lt",
    "topoomni_layer9_sheet_mp", "topoomni_layer18_sheet_mp", "topoomni_layer27_sheet_mp",
    "topoomni_layer9_sheet_lt", "topoomni_layer18_sheet_lt", "topoomni_layer27_sheet_lt",
    "nemotron_layer9_mp", "nemotron_layer18_mp", "nemotron_layer27_mp", "nemotron_layer36_mp",
    "nemotron_layer9_lt", "nemotron_layer18_lt", "nemotron_layer27_lt",
    "nemotron_layer36_lt",
]
DUMMY_CONDITIONS = ["clsav_from_a", "clsav_from_v"]


def main():
    done, partial_model_done, skipped = [], [], []
    for model in MODELS:
        intact_path = GROUP_DIR / f"{model}_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        if not intact_path.exists():
            log.info(f"[{model}] SKIP — missing intact RSA: {intact_path}")
            skipped.append(model)
            continue
        plain_intact = load_named_map(intact_path, "searchlight_spearman_rho")

        partial_dummy_by_cond = {}
        for cond in DUMMY_CONDITIONS:
            dummy_dir = f"{model}_{cond}_av"
            dummy_path = GROUP_DIR / dummy_dir / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
            if not dummy_path.exists():
                log.info(f"[{model}/{cond}] SKIP — missing dummy RSA: {dummy_path}")
                skipped.append(f"{model}/{cond}")
                continue

            plain_dummy = load_named_map(dummy_path, "searchlight_spearman_rho")
            diff = plain_intact - plain_dummy

            out_path = GROUP_DIR / dummy_dir / "dummy_consolidated_maps.dscalar.nii"
            merge_into_combined(plain_dummy, "plain_rsa_dummy", out_path, str(dummy_path))
            merge_into_combined(diff, "diff_intact_minus_dummy_plain_rsa", out_path, str(dummy_path))

            partial_dummy_path = (GROUP_DIR / f"{dummy_dir}_INTEGRATION" / CONFIG
                                   / "integration_partial_r_searchlight.dscalar.nii")
            if partial_dummy_path.exists():
                partial_dummy = load_single_map(partial_dummy_path)
                merge_into_combined(partial_dummy, "partial_rsa_dummy", out_path, str(dummy_path))
                partial_dummy_by_cond[cond] = partial_dummy

            log.info(f"[{model}/{cond}] DONE — mean plain_dummy={plain_dummy.mean():.4f}  "
                     f"mean diff={diff.mean():.4f}  max diff={diff.max():.4f}")
            done.append(f"{model}/{cond}")

        # ── Per-model combined partial diff: partial_intact - max(both dummy
        # conditions' partial rsa). Needs the intact integration map plus at
        # least one dummy condition's partial rerun (max() over whichever
        # are available covers the single-condition case trivially). ──────
        partial_intact_path = GROUP_DIR / f"{model}_av_INTEGRATION" / CONFIG / "integration_partial_r_searchlight.dscalar.nii"
        if not partial_intact_path.exists():
            log.info(f"[{model}] partial diff SKIP — missing intact integration map: {partial_intact_path}")
            continue
        if not partial_dummy_by_cond:
            log.info(f"[{model}] partial diff SKIP — no dummy condition has a partial rerun yet")
            continue

        partial_intact = load_single_map(partial_intact_path)
        max_dummy_partial = np.maximum.reduce(list(partial_dummy_by_cond.values()))
        modality_presence_diff = partial_intact - max_dummy_partial

        partial_out_path = GROUP_DIR / f"{model}_av" / "dummy_partial_diff_maps.dscalar.nii"
        merge_into_combined(modality_presence_diff, "modality_presence_diff",
                             partial_out_path, str(intact_path))
        log.info(f"[{model}] DONE (modality_presence_diff, max over {list(partial_dummy_by_cond)}) — "
                 f"mean={modality_presence_diff.mean():.4f}  max={modality_presence_diff.max():.4f}")
        partial_model_done.append(model)

    log.info(f"\nConsolidated {len(done)}/{len(MODELS) * len(DUMMY_CONDITIONS)} model/condition pairs (plain diff). "
             f"{len(partial_model_done)}/{len(MODELS)} models also got the combined partial diff. "
             f"Skipped (not yet ready): {skipped}")


if __name__ == "__main__":
    main()
