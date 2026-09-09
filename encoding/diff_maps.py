#!/usr/bin/env python3
"""encoding/diff_maps.py
=========================
Encoding-currency analogue of rsa/scramble_diff_maps.py + rsa/dummy_diff_maps.py.

For each native-AV model, consolidates the intact-vs-scramble and
intact-vs-dummy diffs on BOTH signals encoding produces:
  - plain     : encoding_r2_audiovisual (encoding.py) -- raw predictive alignment.
  - incremental: incremental_av_delta_r2 (variance_partition.py), the direct
                 held-out gain from adding J to the additive A+V baseline.

Contrasts are named for the tested comparison:
  - "pairing_specific_delta" = incremental_intact - incremental_scrambled. Tests
                              whether the unique-fusion signal needs correct
                              TEMPORAL pairing.
  - "modality_presence_diff" = incremental_intact - max(incremental_dummy_from_a,
                              incremental_dummy_from_v). Tests whether the
                              incremental joint signal needs BOTH real
                              modalities present (not a binding/pairing
                              question, so never called "binding").

Inputs (all pre-computed; see encoding/run_diff_study.sh):
  {OUTPUT_DIR}/group_average/{model}/{config}/encoding_r2_audiovisual.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}/{config}/incremental_av_delta_r2.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}_avscramble/{config}/... (same two files)
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}/{config}/... (same two files)

Per-condition output:
  {OUTPUT_DIR}/group_average/{model}_avscramble/{config}/scramble_consolidated_maps.dscalar.nii
    maps: plain_r2_scrambled, incremental_r2_scrambled, diff_plain,
          pairing_specific_delta
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}/{config}/dummy_consolidated_maps.dscalar.nii
    maps: plain_r2_dummy, incremental_r2_dummy, diff_plain

Per-MODEL output (dummy only, combining both conditions via max()):
  {OUTPUT_DIR}/group_average/{model}/{config}/dummy_partial_diff_maps.dscalar.nii
    map: modality_presence_diff

Run with:
    conda run -n movie python encoding/diff_maps.py
"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import load_named_map, merge_into_combined  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/encoding/group_average")
CONFIG = "delay5s_norm_bin5s_skip5s"

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


def _model_dir(model: str, cond: str | None = None) -> Path:
    return GROUP_DIR / (model if cond is None else f"{model}_{cond}")


def main():
    scramble_done, dummy_done, modpres_done, skipped = [], [], [], []

    for model in BASE_MODELS:
        intact_dir = _model_dir(model)
        plain_intact = _load(intact_dir / CONFIG / "encoding_r2_audiovisual.dscalar.nii", "encoding_r2_audiovisual")
        incremental_intact = _load(
            intact_dir / CONFIG / "incremental_av_delta_r2.dscalar.nii",
            "incremental_av_delta_r2",
        )
        if plain_intact is None:
            log.info(f"[{model}] SKIP — missing intact encoding_r2_audiovisual")
            skipped.append(model)
            continue

        # ── Scramble ──────────────────────────────────────────────────────
        scr_dir = _model_dir(model, "avscramble")
        plain_scr = _load(scr_dir / CONFIG / "encoding_r2_audiovisual.dscalar.nii", "encoding_r2_audiovisual")
        if plain_scr is not None:
            diff_plain = plain_intact - plain_scr
            out_path = scr_dir / CONFIG / "scramble_consolidated_maps.dscalar.nii"
            template = str(scr_dir / CONFIG / "encoding_r2_audiovisual.dscalar.nii")
            merge_into_combined(plain_scr, "plain_r2_scrambled", out_path, template)
            merge_into_combined(diff_plain, "diff_plain", out_path, template)

            incremental_scr = _load(
                scr_dir / CONFIG / "incremental_av_delta_r2.dscalar.nii",
                "incremental_av_delta_r2",
            )
            if incremental_intact is not None and incremental_scr is not None:
                pairing_delta = incremental_intact - incremental_scr
                merge_into_combined(
                    incremental_scr, "incremental_r2_scrambled", out_path, template
                )
                merge_into_combined(
                    pairing_delta, "pairing_specific_delta", out_path, template
                )
                log.info(f"[{model}] scramble DONE — mean diff_plain={diff_plain.mean():.4f}  "
                         f"mean pairing_specific_delta={pairing_delta.mean():.4f}")
            else:
                log.info(f"[{model}] scramble plain diff done; incremental comparison missing")
            scramble_done.append(model)
        else:
            log.info(f"[{model}] scramble SKIP — missing encoding_r2_audiovisual")

        # ── Dummy (both conditions; also feeds the per-model max() combo) ──
        incremental_dummy_by_cond = {}
        for cond in DUMMY_CONDITIONS:
            cond_dir = _model_dir(model, cond)
            plain_dummy = _load(cond_dir / CONFIG / "encoding_r2_audiovisual.dscalar.nii", "encoding_r2_audiovisual")
            if plain_dummy is None:
                log.info(f"[{model}/{cond}] SKIP — missing encoding_r2_audiovisual")
                continue
            diff_plain = plain_intact - plain_dummy
            out_path = cond_dir / CONFIG / "dummy_consolidated_maps.dscalar.nii"
            template = str(cond_dir / CONFIG / "encoding_r2_audiovisual.dscalar.nii")
            merge_into_combined(plain_dummy, "plain_r2_dummy", out_path, template)
            merge_into_combined(diff_plain, "diff_plain", out_path, template)

            incremental_dummy = _load(
                cond_dir / CONFIG / "incremental_av_delta_r2.dscalar.nii",
                "incremental_av_delta_r2",
            )
            if incremental_dummy is not None:
                merge_into_combined(
                    incremental_dummy, "incremental_r2_dummy", out_path, template
                )
                incremental_dummy_by_cond[cond] = incremental_dummy
            log.info(f"[{model}/{cond}] dummy DONE — mean diff_plain={diff_plain.mean():.4f}")
            dummy_done.append(f"{model}/{cond}")

        if incremental_intact is None:
            log.info(f"[{model}] modality_presence_diff SKIP — missing intact incremental map")
            continue
        if not incremental_dummy_by_cond:
            log.info(f"[{model}] modality_presence_diff SKIP — no dummy incremental maps")
            continue
        max_dummy = np.maximum.reduce(list(incremental_dummy_by_cond.values()))
        modality_presence_diff = incremental_intact - max_dummy
        mp_out_path = intact_dir / CONFIG / "dummy_partial_diff_maps.dscalar.nii"
        merge_into_combined(modality_presence_diff, "modality_presence_diff", mp_out_path,
                             str(intact_dir / CONFIG / "incremental_av_delta_r2.dscalar.nii"))
        log.info(f"[{model}] modality_presence_diff DONE (max over {list(incremental_dummy_by_cond)}) — "
                 f"mean={modality_presence_diff.mean():.4f}")
        modpres_done.append(model)

    log.info(f"\nScramble consolidated: {len(scramble_done)}/{len(BASE_MODELS)} models. "
             f"Dummy consolidated: {len(dummy_done)}/{len(BASE_MODELS) * len(DUMMY_CONDITIONS)} model/conditions. "
             f"modality_presence_diff: {len(modpres_done)}/{len(BASE_MODELS)} models. "
             f"Fully skipped (no intact): {skipped}")


if __name__ == "__main__":
    main()
