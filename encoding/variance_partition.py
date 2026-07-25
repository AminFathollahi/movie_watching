"""
encoding/variance_partition.py
================================
Move 6 — Encoding fold-in: banded-ridge unique-AV-variance map.

Asks the SAME integration question as rsa/partial_rsa.py's Move-1 contrast,
but in PREDICTION currency instead of representational-geometry currency
(MIRAGE framing: native multimodal features beat post-hoc unimodal
aggregation). Fits a group ridge (himalaya GroupRidgeCV, i.e. "BandedRidgeCV")
with one feature band per --nuisance-modalities entry, plus:

  AV-resid-band : the component of --model's AV joint embedding that is
                  orthogonal to the nuisance band(s) — computed by
                  rsa.multimodal_decomposition.compute_interaction_residual_cv(),
                  the encoding-currency counterpart of Move 1's best-additive
                  integration contrast.

--nuisance-modalities defaults to "a,v" (2 bands: A, V — the intact/scramble
case, where --model's OWN a/v are the right nuisance, since --nuisance-model
also defaults to --model). Pass a single modality (e.g. "a") together with
--nuisance-model pointing at the INTACT base model for a dummy-modality
--model: the placeholder modality has ~zero variance and can't be a nuisance
band, so only the one real modality is used (mirrors
rsa/shared/model_registry.py's _dummy_integration_run() reasoning).

r2_score_split() (himalaya) partitions the joint model's test-set R^2 into a
contribution per band; the AV-resid band's split R^2 is the UNIQUE variance
explained by fusion beyond any linear reweighting of the nuisance band(s) —
the encoding analogue of the RSA integration map ("encoding_r2_unique_av").

Design choices vs. cf_modeling/02_fit_cf_model.py's himalaya pattern
----------------------------------------------------------------------
cf_modeling's bands are LBOE geometric bases from two cortical ROIs (a
different construct — a connective-field model, not a movie-embedding
encoding model) and its Delayer step models an FIR temporal lag basis
because CF regressors are the OTHER ROI's raw signal. Here the hemodynamic
delay is already handled the same way as encoding/encoding.py (shifting the
fMRI extraction window by --delay-sec), so no Delayer transform is added —
adding one would double-shift and break consistency with the rest of the
encoding pipeline. What IS copied is the general pattern: multiple named
feature bands -> himalaya group/banded ridge with per-band regularization,
LORO-CV, band_sizes tracked explicitly, r2_score_split for the variance
partition.

Usage
-----
python encoding/variance_partition.py \\
    --preprocessed-dir /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw \\
    --fmri-suffix raw \\
    --timing-csv /home/amin/Research/Representation/Movie/data/movie_timing.csv \\
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \\
    --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \\
    --output-dir /home/amin/Research/Representation/Movie/outputs/encoding \\
    --subject group_average --model pe-av-small-16-frame \\
    --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 --normalize \\
    --alpha-min -2 --alpha-max 9 --n-alphas 23 --n-iter 20 \\
    --test-video-ids video5,video9,video14,video18
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Remove the script's own directory from sys.path so "encoding/encoding.py"
# doesn't shadow the top-level encoding/ package when resolving
# "encoding.shared..." (same fix as encoding/encoding.py).
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(ROOT))

from encoding.shared.encoding_utils import (
    build_fmri_arrays, split_embedding_array, make_loro_splitter, save_cifti,
)
from rsa.multimodal_decomposition import compute_interaction_residual_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Move 6: banded-ridge unique-AV-variance encoding map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessed-dir", required=True, dest="preprocessed_dir")
    p.add_argument("--fmri-suffix",      default="raw", dest="fmri_suffix")
    p.add_argument("--timing-csv",       required=True)
    p.add_argument("--embeddings-dir",   required=True)
    p.add_argument("--template-cifti",   required=True)
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--subject",          default="group_average")
    p.add_argument("--model",            required=True,
                   help="Native-AV model name (must have an _av embedding under this name).")
    p.add_argument("--nuisance-model",      default=None, dest="nuisance_model",
                   help="Model whose unimodal embedding(s) form the nuisance band(s). "
                        "Defaults to --model (the intact/scramble case, where the "
                        "target's OWN _a/_v are the right nuisance). Pass the INTACT "
                        "base model here for a dummy-modality --model (e.g. "
                        "--model omni3b_layer9_mp_clsav_from_a --nuisance-model "
                        "omni3b_layer9_mp --nuisance-modalities a) -- the placeholder "
                        "modality has ~zero variance and can't be a nuisance band.")
    p.add_argument("--nuisance-modalities", default="a,v", dest="nuisance_modalities",
                   help="Comma-separated modalities to load from --nuisance-model as "
                        "nuisance bands, e.g. 'a,v' (default, intact/scramble) or "
                        "'a' / 'v' (dummy-modality: only the real modality is usable).")
    p.add_argument("--bin-sec",   type=float, required=True)
    p.add_argument("--skip-sec",  type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr",        type=float, required=True)
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score of training embeddings (matches encoding.py).")
    p.add_argument("--alpha-min", type=float, required=True)
    p.add_argument("--alpha-max", type=float, required=True)
    p.add_argument("--n-alphas",  type=int,   required=True)
    p.add_argument("--n-iter",    type=int,   default=20,
                   help="Random-search iterations for the group-ridge scaling deltas.")
    p.add_argument("--backend",   default="torch_cuda")
    p.add_argument("--test-video-ids", required=True)
    return p.parse_args()


def _config_label(args) -> str:
    return f"delay{args.delay_sec:.0f}s_{'norm' if args.normalize else 'demean'}_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"


# =============================================================================
# Main
# =============================================================================

def run_analysis(args):
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    timing_df = pd.read_csv(args.timing_csv)
    test_ids  = [v.strip() for v in args.test_video_ids.split(",")]
    alphas    = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    config    = _config_label(args)

    out_root = Path(args.output_dir) / args.subject / args.model / config
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "encoding_r2_unique_av.dscalar.nii"

    log.info("=" * 70)
    log.info(f"Move 6 — Banded-ridge unique-AV-variance: {args.model}")
    log.info(f"  bin_sec={args.bin_sec}  skip_sec={args.skip_sec}  "
             f"delay_sec={args.delay_sec}  tr={args.tr}")
    log.info(f"  Output: {out_path}")
    log.info("=" * 70)

    # ── fMRI: same train/test split as encoding.py ───────────────────────────
    cifti_path = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    run_trs_path = (Path(args.preprocessed_dir) /
                     f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    Y_train, Y_test, run_onsets = build_fmri_arrays(
        str(cifti_path), str(run_trs_path), timing_df, test_ids,
        args.bin_sec, args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    log.info(f"  Y_train={Y_train.shape}  Y_test={Y_test.shape}  run_onsets={run_onsets}")

    # ── Load the target AV embedding and the nuisance modality embedding(s).
    # Nuisance defaults to --model's own a/v (intact + scramble: the target's
    # OWN unimodal streams are the right nuisance). Dummy-modality --model
    # values pass --nuisance-model/--nuisance-modalities to point at the
    # INTACT base model's real modality instead -- the placeholder modality
    # has ~zero variance and can't be a nuisance band (see parse_args()). ───
    nuisance_model = args.nuisance_model or args.model
    nuisance_mods = [m.strip() for m in args.nuisance_modalities.split(",")]
    band_names = [m.upper() for m in nuisance_mods] + ["AVresid"]

    emb_dir = (Path(args.embeddings_dir) / args.model /
               f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s")
    av_emb = np.load(emb_dir / f"{args.model}_av.npy")

    nuisance_dir = (Path(args.embeddings_dir) / nuisance_model /
                    f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s")
    nuisance_embs = [np.load(nuisance_dir / f"{nuisance_model}_{m}.npy") for m in nuisance_mods]
    log.info(f"  Embeddings: av={av_emb.shape}  nuisance({nuisance_model}, {nuisance_mods})="
             f"{[e.shape for e in nuisance_embs]}")

    # NOTE: uses the cross-validated-ridge variant, NOT
    # rsa.multimodal_decomposition.compute_interaction_residual()'s near-zero-eps
    # closed form — that form degenerates (R -> ~0) when n_samples (626 bins) <<
    # n_nuisance_features (2048 concatenated A+V dims), which is exactly the
    # regime here. See compute_interaction_residual_cv()'s docstring.
    R, ms_score, best_alpha = compute_interaction_residual_cv(av_emb, nuisance_embs)
    log.info(f"  AV-joint residual (CV ridge, alpha={best_alpha:.2e}): shape={R.shape}  "
             f"multimodality_score(||R||^2/||J||^2)={ms_score:.4f}")

    # ── Split each band into train/test using the SAME segment logic as
    #    encoding.py's build_embedding_arrays (per-run demean/normalize on
    #    training bins only, applied consistently to test bins). ─────────────
    bands_raw = dict(zip([m.upper() for m in nuisance_mods], nuisance_embs))
    bands_raw["AVresid"] = R
    X_train_bands, X_test_bands, band_sizes = [], [], []
    for name in band_names:
        X_tr, X_te = split_embedding_array(
            bands_raw[name], timing_df, test_ids, args.bin_sec,
            hrf=False, normalize=args.normalize, skip_sec=args.skip_sec,
        )
        log.info(f"  [{name}] X_train={X_tr.shape}  X_te={X_te.shape}")
        X_train_bands.append(X_tr.astype(np.float32))
        X_test_bands.append(X_te.astype(np.float32))
        band_sizes.append(X_tr.shape[1])
    np.save(out_root / "band_sizes.npy", np.array(band_sizes, dtype=np.int64))

    # ── Fit banded (group) ridge: per-band regularization via random search
    #    over group scalings, LORO-CV (same run_onsets as the fMRI split). ───
    import torch as _torch
    from himalaya.ridge import GroupRidgeCV
    from himalaya.backend import set_backend
    from himalaya.scoring import r2_score, r2_score_split

    backend = args.backend
    if backend == "torch_cuda" and not _torch.cuda.is_available():
        log.warning("CUDA not available — falling back to torch backend")
        backend = "torch"
    set_backend(backend, on_error="warn")

    cv_splitter = make_loro_splitter(len(Y_train), run_onsets)
    model = GroupRidgeCV(
        groups="input",
        solver="random_search",
        solver_params={
            "alphas": alphas,
            "n_iter": args.n_iter,
            "n_targets_batch": 2000,
            "n_targets_batch_refit": 2000,
            "progress_bar": False,
        },
        cv=cv_splitter,
        fit_intercept=False,
        Y_in_cpu=True,
    )
    log.info(f"  Fitting GroupRidgeCV (banded ridge): bands={band_names} "
             f"sizes={band_sizes}  n_iter={args.n_iter}  backend={backend}")
    model.fit(X_train_bands, Y_train)

    log.info("  Predicting on test set (split per band) ...")
    Y_pred_split = model.predict(X_test_bands, split=True)  # (3, n_test, n_vertices)
    if hasattr(Y_pred_split, "cpu"):
        Y_pred_split = Y_pred_split.cpu().numpy()
    Y_pred_split = np.asarray(Y_pred_split, dtype=np.float32)

    # r2_score_split requires y_true zero-mean over samples.
    Y_test_c = (Y_test - Y_test.mean(axis=0, keepdims=True)).astype(np.float32)

    r2_split = r2_score_split(Y_test_c, Y_pred_split)  # (3, n_vertices)
    if hasattr(r2_split, "cpu"):
        r2_split = r2_split.cpu().numpy()
    r2_split = np.asarray(r2_split, dtype=np.float32)

    r2_full = r2_score(Y_test_c, Y_pred_split.sum(axis=0))
    if hasattr(r2_full, "cpu"):
        r2_full = r2_full.cpu().numpy()
    r2_full = np.asarray(r2_full, dtype=np.float32)

    r2_unique_av = r2_split[band_names.index("AVresid")]
    r2_nuisance = {name: r2_split[band_names.index(name)] for name in band_names if name != "AVresid"}

    nuisance_log = "\n".join(
        f"  r2_{name}: mean={vals.mean():.4f} max={vals.max():.4f}" for name, vals in r2_nuisance.items()
    )
    log.info(
        f"  r2_full: mean={r2_full.mean():.4f} max={r2_full.max():.4f}\n"
        f"{nuisance_log}\n"
        f"  r2_unique_av (AVresid band, PRIMARY): mean={r2_unique_av.mean():.4f} "
        f"max={r2_unique_av.max():.4f} frac_pos={(r2_unique_av > 0).mean():.3f}"
    )

    # ── Save CIFTI outputs ─────────────────────────────────────────────────
    save_cifti(r2_unique_av, args.template_cifti, str(out_path),
               map_name="encoding_r2_unique_av")
    save_cifti(r2_full, args.template_cifti,
               str(out_root / "encoding_r2_full.dscalar.nii"), map_name="encoding_r2_full")
    for name, vals in r2_nuisance.items():
        save_cifti(vals, args.template_cifti,
                   str(out_root / f"encoding_r2_{name}.dscalar.nii"), map_name=f"encoding_r2_{name}")
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
