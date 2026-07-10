#!/usr/bin/env python3
"""Consolidate plain RSA (scrambled), partial RSA (scrambled), and their
intact-minus-scrambled diff into one 3-map CIFTI per avscramble model config.

Inputs (all pre-computed):
  - plain RSA, scrambled:  {GROUP_DIR}/{model}_avscramble_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - plain RSA, intact:     {GROUP_DIR}/{model}_av/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii
        map 'searchlight_spearman_rho'
  - partial RSA, scrambled (Move 1 integration contrast, re-run on scrambled embeddings):
        {GROUP_DIR}/{model}_avscramble_av_INTEGRATION/k100_delay5s_bin5s_skip5s_spearman/integration_partial_r_searchlight.dscalar.nii

Output: {GROUP_DIR}/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii
  maps: plain_rsa_scrambled, partial_rsa_scrambled, diff_intact_minus_scrambled_plain_rsa
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

MODELS = [
    "pe-av-small-16-frame",
    "cav-mae-sync",
    "omni3b_layer9", "omni3b_layer18", "omni3b_layer27",
    "omni3b_layer9_lasttoken", "omni3b_layer18_lasttoken", "omni3b_layer27_lasttoken",
    "topoomni_layer9", "topoomni_layer18", "topoomni_layer27",
    "topoomni_layer9_lasttoken", "topoomni_layer18_lasttoken", "topoomni_layer27_lasttoken",
    "nemotron_layer9", "nemotron_layer18", "nemotron_layer27", "nemotron_layer36",
]


def load_map(cifti_path: Path, map_name: str) -> np.ndarray:
    img = nib.load(str(cifti_path))
    names = list(img.header.get_axis(0).name)
    idx = names.index(map_name)
    return img.get_fdata(dtype=np.float32)[idx]


def main():
    done, skipped = [], []
    for model in MODELS:
        scrambled_plain_path = GROUP_DIR / f"{model}_avscramble_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        intact_plain_path = GROUP_DIR / f"{model}_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        partial_scrambled_path = GROUP_DIR / f"{model}_avscramble_av_INTEGRATION" / CONFIG / "integration_partial_r_searchlight.dscalar.nii"

        missing = [p for p in (scrambled_plain_path, intact_plain_path, partial_scrambled_path) if not p.exists()]
        if missing:
            log.info(f"[{model}] SKIP — missing: {[str(p) for p in missing]}")
            skipped.append(model)
            continue

        plain_scrambled = load_map(scrambled_plain_path, "searchlight_spearman_rho")
        plain_intact = load_map(intact_plain_path, "searchlight_spearman_rho")
        partial_scrambled_img = nib.load(str(partial_scrambled_path))
        partial_scrambled = partial_scrambled_img.get_fdata(dtype=np.float32)[0]

        diff = plain_intact - plain_scrambled

        out_path = GROUP_DIR / f"{model}_avscramble_av" / "scramble_consolidated_maps.dscalar.nii"
        merge_into_combined(plain_scrambled, "plain_rsa_scrambled", out_path, str(scrambled_plain_path))
        merge_into_combined(partial_scrambled, "partial_rsa_scrambled", out_path, str(scrambled_plain_path))
        merge_into_combined(diff, "diff_intact_minus_scrambled_plain_rsa", out_path, str(scrambled_plain_path))

        log.info(f"[{model}] DONE — mean plain_scrambled={plain_scrambled.mean():.4f}  "
                 f"mean diff={diff.mean():.4f}  max diff={diff.max():.4f}")
        done.append(model)

    log.info(f"\nConsolidated {len(done)}/{len(MODELS)} models. Skipped (not yet ready): {skipped}")


if __name__ == "__main__":
    main()
