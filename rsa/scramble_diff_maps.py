#!/usr/bin/env python3
"""Consolidate plain RSA (scrambled), partial RSA (scrambled), and BOTH
intact-minus-scrambled diffs (plain and partial) into one CIFTI per
avscramble model config.

Inputs (all pre-computed):
  - plain RSA, scrambled:  {GROUP_DIR}/{model}_avscramble_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - plain RSA, intact:     {GROUP_DIR}/{model}_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - partial RSA, scrambled (best-additive integration contrast, re-run on scrambled embeddings):
        {GROUP_DIR}/{model}_avscramble_av_INTEGRATION/k100_delay5s_bin5s_skip5s_spearman/integration_partial_r_searchlight.dscalar.nii
  - partial RSA, intact (the same integration contrast on the intact model):
        {GROUP_DIR}/{model}_av_INTEGRATION/k100_delay5s_bin5s_skip5s_spearman/integration_partial_r_searchlight.dscalar.nii

Output: {GROUP_DIR}/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii
  maps: plain_rsa_scrambled, partial_rsa_scrambled,
        diff_intact_minus_scrambled_plain_rsa, binding

  "binding" = integration_intact - integration_scrambled, i.e. EXACTLY what
  rsa/temporal_scramble_binding.py computes and calls "binding" across a
  batch of models -- same name here deliberately, so this per-model bundle
  and that script's cross-model convergence output are never read as two
  different quantities. Use temporal_scramble_binding.py when you want the
  cross-model convergence view; use this file's "binding" map when you want
  one model's value alongside its plain-RSA counterpart in a single CIFTI.

Expectation (AV-integration hypothesis): in true integration regions, BOTH
diff maps should be positive and large (scrambled pairing weakens both plain
and partial alignment); non-integration regions should stay near zero.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import load_named_map, load_single_map, merge_into_combined  # noqa: E402
from rsa.shared.naming import DEFAULT_MODEL_NORM  # noqa: E402
from paths import OUTPUTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = OUTPUTS / "rsa/raw/group_average"
CONFIG = f"k100_delay5s_bin5s_skip5s_spearman_{os.environ.get('MODEL_NORM', DEFAULT_MODEL_NORM)}"

MODELS = [
    "pe-av-small-16-frame",
    "cav-mae-sync",
    "omni3b_layer9_mp", "omni3b_layer18_mp", "omni3b_layer27_mp",
    "omni3b_layer9_lt", "omni3b_layer18_lt", "omni3b_layer27_lt",
    "topoomni_layer9_mp", "topoomni_layer18_mp", "topoomni_layer27_mp",
    "topoomni_layer9_lt", "topoomni_layer18_lt", "topoomni_layer27_lt",
    "topoomni_layer9_sheet_mp", "topoomni_layer18_sheet_mp", "topoomni_layer27_sheet_mp",
    "topoomni_layer9_sheet_lt", "topoomni_layer18_sheet_lt", "topoomni_layer27_sheet_lt",
    "nemotron_layer9_mp", "nemotron_layer18_mp", "nemotron_layer27_mp", "nemotron_layer36_mp",
]


def main():
    done, partial_done, skipped = [], [], []
    for model in MODELS:
        scrambled_plain_path = GROUP_DIR / f"{model}_avscramble_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        intact_plain_path = GROUP_DIR / f"{model}_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        partial_scrambled_path = GROUP_DIR / f"{model}_avscramble_av_INTEGRATION" / CONFIG / "integration_partial_r_searchlight.dscalar.nii"
        partial_intact_path = GROUP_DIR / f"{model}_av_INTEGRATION" / CONFIG / "integration_partial_r_searchlight.dscalar.nii"

        missing = [p for p in (scrambled_plain_path, intact_plain_path, partial_scrambled_path) if not p.exists()]
        if missing:
            log.info(f"[{model}] SKIP — missing: {[str(p) for p in missing]}")
            skipped.append(model)
            continue

        plain_scrambled = load_named_map(scrambled_plain_path, "searchlight_spearman_rho")
        plain_intact = load_named_map(intact_plain_path, "searchlight_spearman_rho")
        partial_scrambled = load_single_map(partial_scrambled_path)

        diff_plain = plain_intact - plain_scrambled

        out_path = GROUP_DIR / f"{model}_avscramble_av" / "scramble_consolidated_maps.dscalar.nii"
        merge_into_combined(plain_scrambled, "plain_rsa_scrambled", out_path, str(scrambled_plain_path))
        merge_into_combined(partial_scrambled, "partial_rsa_scrambled", out_path, str(scrambled_plain_path))
        merge_into_combined(diff_plain, "diff_intact_minus_scrambled_plain_rsa", out_path, str(scrambled_plain_path))

        log.info(f"[{model}] DONE (plain) — mean plain_scrambled={plain_scrambled.mean():.4f}  "
                 f"mean diff_plain={diff_plain.mean():.4f}  max diff_plain={diff_plain.max():.4f}")
        done.append(model)

        if not partial_intact_path.exists():
            log.info(f"[{model}] partial diff SKIP — missing intact integration map: {partial_intact_path}")
            continue

        partial_intact = load_single_map(partial_intact_path)
        binding = partial_intact - partial_scrambled
        merge_into_combined(binding, "binding", out_path, str(scrambled_plain_path))
        log.info(f"[{model}] DONE (binding) — mean binding={binding.mean():.4f}  "
                 f"max binding={binding.max():.4f}")
        partial_done.append(model)

    log.info(f"\nConsolidated {len(done)}/{len(MODELS)} models (plain diff). "
             f"{len(partial_done)}/{len(MODELS)} also got the partial diff. "
             f"Skipped entirely (not yet ready): {skipped}")


if __name__ == "__main__":
    main()
