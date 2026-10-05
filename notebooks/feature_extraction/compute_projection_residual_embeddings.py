"""
notebooks/feature_extraction/compute_projection_residual_embeddings.py
===========================================================================
Generates "_projection_resid_own" pseudo-model embeddings: the per-timepoint
orthogonal-projection residual of a joint AV embedding after removing its
component in span{a_t, v_t} (see rsa.shared.residuals.projection_residual --
distinct from "_linear_resid_unimodal", which fits one shared linear map
ACROSS all samples; this is a purely local, per-row geometric decomposition).

Two families of runs:

1. Own-unimodal variant (dimension-compatible: target and nuisance share the
   SAME embedding space, so a genuine 2D projection is well-defined):
     - PE-AV and legacy bare probes: AV vs own A/V in the same space.
     - CAV-MAE: av=concat(cls-a, cls-v), with own A/V placed in their
       corresponding zero-padded blocks before projection.
     - omni-family: av={base}_mp/_lt vs (a={base}_mp_a, v={base}_mp_v)
                    [thinker-space nuisance -- "_lt" targets are regressed
                    against the base model's "_mp" real unimodal streams,
                    mirroring model_registry._lasttoken_integration_run]
   Saved as {target_model}_av_projection_resid_own.

2. Encoder-penultimate variant, omni-family only: intended nuisance is
   {family}_encoder_penultimate (a/v), extracted by
   notebooks/feature_extraction/{omni3b,topo_omni,nemotron}_extract_encoder_penultimate.py.
   IMPORTANT CAVEAT: encoder-penultimate embeddings live in the tower's OWN
   pre-projection hidden space (d=1280, before the audio_tower.proj /
   visual.merger step that maps into the thinker's embedding space), while
   av_mp/av_lt live in the THINKER's hidden space (d=2048 for omni3b/
   topoomni, d=4096 for nemotron per its text_config). These are NOT the
   same vector space, so a genuine geometric projection (which requires
   target and nuisance to share one space) is mathematically undefined here.
   This pairing therefore uses the cross-validated ridge residual
   (rsa.shared.residuals.linear_residual) and is tagged
   "_linear_resid_encoder" rather than "_projection_resid_own".

Usage
-----
conda run --no-capture-output -n movie python \
    notebooks/feature_extraction/compute_projection_residual_embeddings.py \
    --embeddings-dir ../outputs/model_embeddings \
    --timing-csv ../data/movie_timing.csv \
    --run-trs ../data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy \
    --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import preprocess_fmri, process_model_embeddings, align_and_assert_bins
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    LEGACY_BARE_AV_MODELS,
    RESIDUALIZED_AV_MODELS,
    TR_DEFAULT,
    emb_path,
)
from rsa.shared.residuals import projection_residual, linear_residual

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAME_SPACE_OWN_MODELS = (
    "pe-av-small-16-frame",
    "cav-mae-sync",
    *LEGACY_BARE_AV_MODELS,
)
OMNI_FAMILY_BASES = list(dict.fromkeys(
    model.rsplit("_", 1)[0]
    for model in RESIDUALIZED_AV_MODELS
    if model.startswith(("omni3b_layer", "topoomni_layer", "nemotron_layer"))
    and model.endswith(("_mp", "_lt"))
))
READOUTS = ["mp", "lt"]


def _family(base: str) -> str:
    if base.startswith("omni3b"):
        return "omni3b"
    if base.startswith("topoomni"):
        return "topoomni"
    if base.startswith("nemotron"):
        return "nemotron"
    raise ValueError(f"Unknown family for base={base}")


def _place_own_streams_in_av_space(
    model: str,
    av: np.ndarray,
    audio: np.ndarray,
    video: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place CAV-MAE's own streams in its explicit concatenated AV space."""
    if audio.shape == av.shape and video.shape == av.shape:
        return audio, video
    if model == "cav-mae-sync" and av.shape[1] == audio.shape[1] + video.shape[1]:
        audio_block = np.concatenate(
            [audio, np.zeros((audio.shape[0], video.shape[1]))], axis=1
        )
        video_block = np.concatenate(
            [np.zeros((video.shape[0], audio.shape[1])), video], axis=1
        )
        return audio_block, video_block
    raise ValueError(
        f"No common projection space for {model}: "
        f"av={av.shape} a={audio.shape} v={video.shape}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate _projection_resid_own (and, where dims mismatch, _linear_resid_encoder) pseudo-models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--run-trs", required=True, dest="run_trs")
    p.add_argument("--bin-sec", type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr", type=float, default=TR_DEFAULT)
    p.add_argument("--force", action="store_true",
                   help="Replace residual embeddings that already exist.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of RESIDUALIZED_AV_MODELS to process.")
    p.add_argument("--skip-encoder", action="store_true",
                   help="Skip the separate encoder-space linear residual variants.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    timing_df = pd.read_csv(args.timing_csv)
    run_trs = np.load(args.run_trs)
    n_trs_total = int(run_trs.sum())
    fmri_binned = preprocess_fmri(
        np.zeros((1, n_trs_total), dtype=np.float32), timing_df, run_trs,
        args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec,
    )
    log.info(f"Reference n_bins={fmri_binned.shape[0]}")

    def load_emb(model, modality):
        path = emb_path(args.embeddings_dir, model, modality, args.bin_sec, args.skip_sec)
        if not path.exists():
            raise FileNotFoundError(str(path))
        emb = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
        )
        _, emb = align_and_assert_bins(fmri_binned, emb)
        return emb.astype(np.float64)

    def save(out_model, R):
        out_path = emb_path(args.embeddings_dir, out_model, "av", args.bin_sec, args.skip_sec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, R.astype(np.float32))
        return out_path

    done, skipped = [], []
    selected = set(args.models or RESIDUALIZED_AV_MODELS)
    unknown = selected.difference(RESIDUALIZED_AV_MODELS)
    if unknown:
        raise ValueError(f"Unsupported residualized AV model(s): {sorted(unknown)}")

    # ── 1. Joint models with dimension-compatible own A/V embeddings ────────
    for model in SAME_SPACE_OWN_MODELS:
        if model not in selected:
            continue
        out_model = f"{model}_av_projection_resid_own"
        out_path = emb_path(args.embeddings_dir, out_model, "av", args.bin_sec, args.skip_sec)
        if out_path.is_file() and not args.force:
            log.info(f"[{model}] SKIP projection_resid_own -- output exists: {out_path}")
            continue
        try:
            av = load_emb(model, "av")
            a = load_emb(model, "a")
            v = load_emb(model, "v")
            a, v = _place_own_streams_in_av_space(model, av, a, v)
            R = projection_residual(av, a, v)
            rel_norm = float(np.linalg.norm(R) / np.linalg.norm(av))
            if rel_norm < 1e-10:
                R = np.zeros_like(R)
                log.info(f"[{model}] projection residual is degenerate; saving exact zeros")
            p = save(out_model, R)
            log.info(f"[{out_model}] ||R||/||av||={rel_norm:.4f} -> {p}")
            done.append(out_model)
        except FileNotFoundError as e:
            log.warning(f"[{model}] SKIP projection_resid_own -- {e}")
            skipped.append(model)

    # ── 2. omni-family: own-mp-unimodal projection residual (mp and lt targets) ──
    for base in OMNI_FAMILY_BASES:
        selected_readouts = [
            readout for readout in READOUTS
            if f"{base}_{readout}" in selected
        ]
        if not selected_readouts:
            continue
        try:
            a_mp = load_emb(f"{base}_mp", "a")
            v_mp = load_emb(f"{base}_mp", "v")
        except FileNotFoundError as e:
            log.warning(f"[{base}] SKIP (own-unimodal nuisance missing) -- {e}")
            skipped.append(f"{base}_*_projection_resid_own")
            continue

        for readout in selected_readouts:
            target_model = f"{base}_{readout}"
            out_model = f"{target_model}_av_projection_resid_own"
            out_path = emb_path(args.embeddings_dir, out_model, "av", args.bin_sec, args.skip_sec)
            if out_path.is_file() and not args.force:
                log.info(f"[{target_model}] SKIP projection_resid_own -- output exists: {out_path}")
                continue
            try:
                target = load_emb(target_model, "av")
            except FileNotFoundError as e:
                log.warning(f"[{target_model}] SKIP projection_resid_own -- {e}")
                skipped.append(f"{target_model}_projection_resid_own")
                continue
            if target.shape[1] != a_mp.shape[1]:
                log.warning(f"[{target_model}] SKIP projection_resid_own -- dim mismatch "
                            f"target={target.shape[1]} nuisance={a_mp.shape[1]}")
                skipped.append(f"{target_model}_projection_resid_own (dim mismatch)")
                continue
            R = projection_residual(target, a_mp, v_mp)
            p = save(out_model, R)
            rel_norm = float(np.linalg.norm(R) / np.linalg.norm(target))
            log.info(f"[{out_model}] ||R||/||av||={rel_norm:.4f} -> {p}")
            done.append(out_model)

    # ── 3. omni-family: encoder-penultimate variant. Dimension mismatch vs.
    # thinker-space av_mp/av_lt (see module docstring) -- uses linear_residual
    # (ridge) instead of a true geometric projection; tagged _linear_resid_encoder.
    if args.skip_encoder:
        log.info("Skipping encoder-space linear residual variants")
        log.info(f"Done. {len(done)} pseudo-models saved, {len(skipped)} skipped.")
        return

    for family in ("omni3b", "topoomni", "nemotron"):
        enc_model = f"{family}_encoder_penultimate"
        try:
            enc_a = load_emb(enc_model, "a")
            enc_v = load_emb(enc_model, "v")
        except FileNotFoundError as e:
            log.warning(f"[{enc_model}] SKIP encoder-penultimate variant -- {e}")
            skipped.append(f"{family}_*_linear_resid_encoder")
            continue

        for base in OMNI_FAMILY_BASES:
            if _family(base) != family:
                continue
            for readout in READOUTS:
                target_model = f"{base}_{readout}"
                if target_model not in selected:
                    continue
                out_model = f"{target_model}_av_linear_resid_encoder"
                out_path = emb_path(args.embeddings_dir, out_model, "av", args.bin_sec, args.skip_sec)
                if out_path.is_file() and not args.force:
                    log.info(f"[{target_model}] SKIP linear_resid_encoder -- output exists: {out_path}")
                    continue
                try:
                    target = load_emb(target_model, "av")
                except FileNotFoundError as e:
                    log.warning(f"[{target_model}] SKIP linear_resid_encoder -- {e}")
                    skipped.append(f"{target_model}_linear_resid_encoder")
                    continue
                R, ms, alpha = linear_residual(target, [enc_a, enc_v])
                p = save(out_model, R)
                log.info(f"[{out_model}] MS={ms:.4f} alpha={alpha:.2e} -> {p}")
                done.append(out_model)

    log.info("=" * 60)
    log.info(f"Done. {len(done)} pseudo-models saved, {len(skipped)} skipped.")
    if skipped:
        log.info("Skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
