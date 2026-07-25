#!/usr/bin/env python3
"""encoding/diff_maps.py
=========================
Encoding-currency analogue of rsa/scramble_diff_maps.py + rsa/dummy_diff_maps.py.

For each native-AV model, consolidates the intact-vs-scramble and
intact-vs-dummy diffs on BOTH signals encoding produces:
  - plain     : encoding_r2_av (encoding.py) -- raw predictive alignment.
  - AVresid   : encoding_r2_unique_av (encoding/variance_partition.py, Move 6)
                -- the encoding-currency counterpart of RSA's "integration".

Naming mirrors the RSA side exactly (same words, same meaning, so a reader
never has to ask whether "binding" means something different here):
  - "binding"               = AVresid_intact - AVresid_scrambled. Tests
                              whether the unique-fusion signal needs correct
                              TEMPORAL pairing.
  - "modality_presence_diff" = AVresid_intact - max(AVresid_dummy_from_a,
                              AVresid_dummy_from_v). Tests whether the
                              unique-fusion signal needs BOTH real
                              modalities present (not a binding/pairing
                              question, so never called "binding").

Inputs (all pre-computed; see encoding/run_diff_study.sh):
  {OUTPUT_DIR}/group_average/{model}/{config}/encoding_r2_av.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}/{config}/encoding_r2_unique_av.dscalar.nii
  {OUTPUT_DIR}/group_average/{model}_avscramble/{config}/... (same two files)
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}/{config}/... (same two files)

Per-condition output:
  {OUTPUT_DIR}/group_average/{model}_avscramble/{config}/scramble_consolidated_maps.dscalar.nii
    maps: plain_r2_scrambled, avresid_r2_scrambled, diff_plain, binding
  {OUTPUT_DIR}/group_average/{model}_clsav_from_{a,v}/{config}/dummy_consolidated_maps.dscalar.nii
    maps: plain_r2_dummy, avresid_r2_dummy, diff_plain

Per-MODEL output (dummy only, combining both conditions via max()):
  {OUTPUT_DIR}/group_average/{model}/{config}/dummy_partial_diff_maps.dscalar.nii
    map: modality_presence_diff

Run with:
    conda run -n movie python encoding/diff_maps.py
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
    img = nib.load(str(path))
    names = list(img.header.get_axis(0).name)
    if map_name not in names:
        return None
    idx = names.index(map_name)
    return img.get_fdata(dtype=np.float32)[idx]


def _model_dir(model: str, cond: str | None = None) -> Path:
    return GROUP_DIR / (model if cond is None else f"{model}_{cond}")


def main():
    scramble_done, dummy_done, modpres_done, skipped = [], [], [], []

    for model in BASE_MODELS:
        intact_dir = _model_dir(model)
        plain_intact = _load(intact_dir / CONFIG / "encoding_r2_av.dscalar.nii", "encoding_r2_av")
        avresid_intact = _load(intact_dir / CONFIG / "encoding_r2_unique_av.dscalar.nii", "encoding_r2_unique_av")
        if plain_intact is None:
            log.info(f"[{model}] SKIP — missing intact encoding_r2_av")
            skipped.append(model)
            continue

        # ── Scramble ──────────────────────────────────────────────────────
        scr_dir = _model_dir(model, "avscramble")
        plain_scr = _load(scr_dir / CONFIG / "encoding_r2_av.dscalar.nii", "encoding_r2_av")
        if plain_scr is not None:
            diff_plain = plain_intact - plain_scr
            out_path = scr_dir / CONFIG / "scramble_consolidated_maps.dscalar.nii"
            template = str(scr_dir / CONFIG / "encoding_r2_av.dscalar.nii")
            merge_into_combined(plain_scr, "plain_r2_scrambled", out_path, template)
            merge_into_combined(diff_plain, "diff_plain", out_path, template)

            avresid_scr = _load(scr_dir / CONFIG / "encoding_r2_unique_av.dscalar.nii", "encoding_r2_unique_av")
            if avresid_intact is not None and avresid_scr is not None:
                binding = avresid_intact - avresid_scr
                merge_into_combined(avresid_scr, "avresid_r2_scrambled", out_path, template)
                merge_into_combined(binding, "binding", out_path, template)
                log.info(f"[{model}] scramble DONE — mean diff_plain={diff_plain.mean():.4f}  "
                         f"mean binding={binding.mean():.4f}")
            else:
                log.info(f"[{model}] scramble plain diff done; binding SKIP (missing AVresid)")
            scramble_done.append(model)
        else:
            log.info(f"[{model}] scramble SKIP — missing encoding_r2_av")

        # ── Dummy (both conditions; also feeds the per-model max() combo) ──
        avresid_dummy_by_cond = {}
        for cond in DUMMY_CONDITIONS:
            cond_dir = _model_dir(model, cond)
            plain_dummy = _load(cond_dir / CONFIG / "encoding_r2_av.dscalar.nii", "encoding_r2_av")
            if plain_dummy is None:
                log.info(f"[{model}/{cond}] SKIP — missing encoding_r2_av")
                continue
            diff_plain = plain_intact - plain_dummy
            out_path = cond_dir / CONFIG / "dummy_consolidated_maps.dscalar.nii"
            template = str(cond_dir / CONFIG / "encoding_r2_av.dscalar.nii")
            merge_into_combined(plain_dummy, "plain_r2_dummy", out_path, template)
            merge_into_combined(diff_plain, "diff_plain", out_path, template)

            avresid_dummy = _load(cond_dir / CONFIG / "encoding_r2_unique_av.dscalar.nii", "encoding_r2_unique_av")
            if avresid_dummy is not None:
                merge_into_combined(avresid_dummy, "avresid_r2_dummy", out_path, template)
                avresid_dummy_by_cond[cond] = avresid_dummy
            log.info(f"[{model}/{cond}] dummy DONE — mean diff_plain={diff_plain.mean():.4f}")
            dummy_done.append(f"{model}/{cond}")

        if avresid_intact is None:
            log.info(f"[{model}] modality_presence_diff SKIP — missing intact AVresid")
            continue
        if not avresid_dummy_by_cond:
            log.info(f"[{model}] modality_presence_diff SKIP — no dummy AVresid available")
            continue
        max_dummy = np.maximum.reduce(list(avresid_dummy_by_cond.values()))
        modality_presence_diff = avresid_intact - max_dummy
        mp_out_path = intact_dir / CONFIG / "dummy_partial_diff_maps.dscalar.nii"
        merge_into_combined(modality_presence_diff, "modality_presence_diff", mp_out_path,
                             str(intact_dir / CONFIG / "encoding_r2_unique_av.dscalar.nii"))
        log.info(f"[{model}] modality_presence_diff DONE (max over {list(avresid_dummy_by_cond)}) — "
                 f"mean={modality_presence_diff.mean():.4f}")
        modpres_done.append(model)

    log.info(f"\nScramble consolidated: {len(scramble_done)}/{len(BASE_MODELS)} models. "
             f"Dummy consolidated: {len(dummy_done)}/{len(BASE_MODELS) * len(DUMMY_CONDITIONS)} model/conditions. "
             f"modality_presence_diff: {len(modpres_done)}/{len(BASE_MODELS)} models. "
             f"Fully skipped (no intact): {skipped}")


if __name__ == "__main__":
    main()
