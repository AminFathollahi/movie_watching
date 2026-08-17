#!/usr/bin/env python3
"""subcortical/diff_maps.py
============================
Subcortical analogue of rsa/scramble_diff_maps.py + rsa/dummy_diff_maps.py
(and encoding/diff_maps.py). Same naming, same meaning, same math -- only
the brain structure being searchlit differs.

For each native-AV model, consolidates the intact-vs-scramble and
intact-vs-dummy diffs on both signals subcortical_partial_rsa.py /
subcortical_rsa.py produce:
  - plain     : searchlight_rho (subcortical_rsa.py) -- raw representational
                alignment on subcortical voxels.
  - partial   : partial_rsa_{label} (subcortical_partial_rsa.py, Move 1) --
                the same best-additive integration contrast as cortex, just
                run on subcortical structures.

  - "binding"               = partial_intact - partial_scrambled. Tests
                              whether the unique-fusion signal needs correct
                              TEMPORAL pairing.
  - "modality_presence_diff" = partial_intact - max(partial_dummy_from_a,
                              partial_dummy_from_v). Tests whether the
                              unique-fusion signal needs BOTH real
                              modalities present (never called "binding" --
                              not a pairing question).

Inputs (all pre-computed; see subcortical/run_diff_study.sh):
  {OUTPUT_DIR}/group_average/{model}_av/{config}/rsa_subcortical_raw_{config}_maps.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}_av_INTEGRATION/{config}/integration_partial_r_searchlight.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}_avscramble_av/... (same two files)
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}_av/... (same two files)

Per-condition output:
  {OUTPUT_DIR}/group_average/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii
    maps: plain_rsa_scrambled, partial_rsa_scrambled, diff_intact_minus_scrambled_plain_rsa, binding
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii
    maps: plain_rsa_dummy, diff_intact_minus_dummy_plain_rsa, partial_rsa_dummy (if computed)

Per-MODEL output (dummy only, combining both conditions via max()):
  {OUTPUT_DIR}/group_average/{model}_av/dummy_partial_diff_maps.dscalar.nii
    map: modality_presence_diff

Run with:
    conda run -n movie python subcortical/diff_maps.py
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

GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/subcortical/group_average")
CONFIG = "k100_delay5s_bin5s_skip5s_spearman"

# Same native-AV roster as rsa/run_diff_study.sh's BASE_MODELS.
BASE_MODELS = [
    "pe-av-small-16-frame",
    "nemotron_layer9_mp", "nemotron_layer9_lt",
    "nemotron_layer18_mp", "nemotron_layer18_lt",
    "nemotron_layer27_mp", "nemotron_layer27_lt",
    "nemotron_layer36_mp", "nemotron_layer36_lt",
    "omni3b_layer9_mp", "omni3b_layer9_lt",
    "omni3b_layer18_mp", "omni3b_layer18_lt",
    "omni3b_layer27_mp", "omni3b_layer27_lt",
    "topoomni_layer9_mp", "topoomni_layer9_lt",
    "topoomni_layer18_mp", "topoomni_layer18_lt",
    "topoomni_layer27_mp", "topoomni_layer27_lt",
    "topoomni_layer9_sheet_mp", "topoomni_layer9_sheet_lt",
    "topoomni_layer18_sheet_mp", "topoomni_layer18_sheet_lt",
    "topoomni_layer27_sheet_mp", "topoomni_layer27_sheet_lt",
]
DUMMY_CONDITIONS = ["clsav_from_a", "clsav_from_v"]

PLAIN_FNAME = f"rsa_subcortical_raw_{CONFIG}_maps.dscalar.nii"
PLAIN_MAP = "searchlight_rho"
INTEGRATION_FNAME = "integration_partial_r_searchlight.dscalar.nii"


def _load(path: Path, map_name: str) -> np.ndarray | None:
    if not path.exists():
        return None
    img = nib.load(str(path))
    names = list(img.header.get_axis(0).name)
    if map_name not in names:
        return None
    idx = names.index(map_name)
    return img.get_fdata(dtype=np.float32)[idx]


def _plain_path(model_mod_dir: str) -> Path:
    return GROUP_DIR / model_mod_dir / CONFIG / PLAIN_FNAME


def _integration_path(model_mod_dir: str) -> Path:
    return GROUP_DIR / f"{model_mod_dir}_INTEGRATION" / CONFIG / INTEGRATION_FNAME


def main():
    scramble_done, dummy_done, modpres_done, skipped = [], [], [], []

    for model in BASE_MODELS:
        intact_dir = f"{model}_av"
        plain_intact = _load(_plain_path(intact_dir), PLAIN_MAP)
        partial_intact = _load(_integration_path(intact_dir), f"partial_rsa_{intact_dir}_INTEGRATION")
        if plain_intact is None:
            log.info(f"[{model}] SKIP — missing intact plain rsa")
            skipped.append(model)
            continue

        # ── Scramble ──────────────────────────────────────────────────────
        scr_dir = f"{model}_avscramble_av"
        plain_scr = _load(_plain_path(scr_dir), PLAIN_MAP)
        if plain_scr is not None:
            diff_plain = plain_intact - plain_scr
            out_path = GROUP_DIR / scr_dir / "scramble_consolidated_maps.dscalar.nii"
            template = str(_plain_path(scr_dir))
            merge_into_combined(plain_scr, "plain_rsa_scrambled", out_path, template)
            merge_into_combined(diff_plain, "diff_intact_minus_scrambled_plain_rsa", out_path, template)

            partial_scr = _load(_integration_path(scr_dir), f"partial_rsa_{scr_dir}_INTEGRATION")
            if partial_intact is not None and partial_scr is not None:
                binding = partial_intact - partial_scr
                merge_into_combined(partial_scr, "partial_rsa_scrambled", out_path, template)
                merge_into_combined(binding, "binding", out_path, template)
                log.info(f"[{model}] scramble DONE — mean diff_plain={diff_plain.mean():.4f}  "
                         f"mean binding={binding.mean():.4f}")
            else:
                log.info(f"[{model}] scramble plain diff done; binding SKIP (missing partial rsa)")
            scramble_done.append(model)
        else:
            log.info(f"[{model}] scramble SKIP — missing plain rsa")

        # ── Dummy (both conditions; also feeds the per-model max() combo) ──
        partial_dummy_by_cond = {}
        for cond in DUMMY_CONDITIONS:
            cond_dir = f"{model}_{cond}_av"
            plain_dummy = _load(_plain_path(cond_dir), PLAIN_MAP)
            if plain_dummy is None:
                log.info(f"[{model}/{cond}] SKIP — missing plain rsa")
                continue
            diff_plain = plain_intact - plain_dummy
            out_path = GROUP_DIR / cond_dir / "dummy_consolidated_maps.dscalar.nii"
            template = str(_plain_path(cond_dir))
            merge_into_combined(plain_dummy, "plain_rsa_dummy", out_path, template)
            merge_into_combined(diff_plain, "diff_intact_minus_dummy_plain_rsa", out_path, template)

            partial_dummy = _load(_integration_path(cond_dir), f"partial_rsa_{cond_dir}_INTEGRATION")
            if partial_dummy is not None:
                merge_into_combined(partial_dummy, "partial_rsa_dummy", out_path, template)
                partial_dummy_by_cond[cond] = partial_dummy
            log.info(f"[{model}/{cond}] dummy DONE — mean diff_plain={diff_plain.mean():.4f}")
            dummy_done.append(f"{model}/{cond}")

        if partial_intact is None:
            log.info(f"[{model}] modality_presence_diff SKIP — missing intact partial rsa")
            continue
        if not partial_dummy_by_cond:
            log.info(f"[{model}] modality_presence_diff SKIP — no dummy partial rsa available")
            continue
        max_dummy = np.maximum.reduce(list(partial_dummy_by_cond.values()))
        modality_presence_diff = partial_intact - max_dummy
        mp_out_path = GROUP_DIR / intact_dir / "dummy_partial_diff_maps.dscalar.nii"
        merge_into_combined(modality_presence_diff, "modality_presence_diff", mp_out_path,
                             str(_plain_path(intact_dir)))
        log.info(f"[{model}] modality_presence_diff DONE (max over {list(partial_dummy_by_cond)}) — "
                 f"mean={modality_presence_diff.mean():.4f}")
        modpres_done.append(model)

    log.info(f"\nScramble consolidated: {len(scramble_done)}/{len(BASE_MODELS)} models. "
             f"Dummy consolidated: {len(dummy_done)}/{len(BASE_MODELS) * len(DUMMY_CONDITIONS)} model/conditions. "
             f"modality_presence_diff: {len(modpres_done)}/{len(BASE_MODELS)} models. "
             f"Fully skipped (no intact): {skipped}")


if __name__ == "__main__":
    main()
