"""
notebooks/feature_extraction/compute_linear_residual_embeddings.py
=====================================================================
Generates the "_linear_resid_unimodal" pseudo-model embeddings: for every model in
rsa.shared.model_registry.RESIDUALIZED_AV_MODELS, the ridge-regression residual of
its AV joint embedding after regressing out AudioMAE(audio) +
VideoMAEv2-Large(video) -- two independent unimodal specialists (same
nuisance pair as rsa/partial_rsa.py's generalized partial_corr_* runs, but
computed directly in embedding space here so the result is a normal
{model}_{modality}.npy file that both RSA and the encoding pipeline can
consume unchanged).

Uses rsa.shared.residuals.linear_residual() (thin wrapper around
cka.multimodal_decomposition.compute_interaction_residual_cv, the
cross-validated-ridge residual already relied on by
encoding/variance_partition.py's Move-6 AVresid band).

Output convention (matches every other pseudo-model, e.g. "_avscramble"):
    {embeddings_dir}/{model}_av_linear_resid_unimodal/bin{B}s_skip{S}s/{model}_av_linear_resid_unimodal_av.npy

The "_unimodal" tag names the nuisance (external unimodal specialists,
AudioMAE + VideoMAEv2-Large) as opposed to "_own" (--variant-suffix _own),
which regresses out the target model's own a/v streams instead -- see
--variant-suffix below.

Once saved, run e.g.:
    python rsa/searchlight.py --model {model}_av_linear_resid_unimodal --modality av ...

Usage
-----
conda run --no-capture-output -n movie python \
    notebooks/feature_extraction/compute_linear_residual_embeddings.py \
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
    TR_DEFAULT,
    RESIDUALIZED_AV_MODELS,
    check_embeddings_exist,
    emb_path,
)
from rsa.shared.residuals import linear_residual

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUISANCE_DEFAULT = [("audiomae", "a"), ("videomaev2-large", "v")]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate _linear_resid_unimodal pseudo-model embeddings for supported AV models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--run-trs", required=True, dest="run_trs")
    p.add_argument("--bin-sec", type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                    help="Defaults to --bin-sec.")
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr", type=float, default=TR_DEFAULT)
    p.add_argument("--models", nargs="+", default=None,
                    help="Subset of RESIDUALIZED_AV_MODELS to process (default: all).")
    p.add_argument("--nuisance", nargs="+", default=None,
                    help="model:modality pairs (default: audiomae:a videomaev2-large:v).")
    p.add_argument("--variant-suffix", default="", dest="variant_suffix",
                    help="Suffix appended after '_av_linear_resid' in the output "
                         "pseudo-model name, naming the nuisance choice. Empty "
                         "(default) falls back to '_unimodal' -- the external-"
                         "specialist nuisance. Pass e.g. '_own' with a "
                         "model's own a/v streams as --nuisance to produce a "
                         "distinctly-named variant instead of overwriting "
                         "the default one.")
    p.add_argument("--force", action="store_true",
                    help="Replace residual embeddings that already exist.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    models = args.models or RESIDUALIZED_AV_MODELS
    nuisance_spec = (
        [tuple(s.split(":", 1)) for s in args.nuisance] if args.nuisance else NUISANCE_DEFAULT
    )

    timing_df = pd.read_csv(args.timing_csv)
    run_trs = np.load(args.run_trs)

    # A single-vertex zero array is enough to derive the correct n_bins from
    # preprocess_fmri's run-boundary binning logic -- avoids loading the real
    # (large) group-average CIFTI just to learn a bin count.
    n_trs_total = int(run_trs.sum())
    fmri_binned = preprocess_fmri(
        np.zeros((1, n_trs_total), dtype=np.float32), timing_df, run_trs,
        args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec,
    )
    log.info(f"Reference n_bins={fmri_binned.shape[0]} (derived from run_trs; no real fMRI loaded)")

    def load_emb(model, modality):
        path = check_embeddings_exist(args.embeddings_dir, model, modality, args.bin_sec, args.skip_sec)
        emb = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
        )
        _, emb = align_and_assert_bins(fmri_binned, emb)
        return emb.astype(np.float64)

    nuisance_embs = [load_emb(m, mod) for m, mod in nuisance_spec]
    log.info(f"Nuisance: {nuisance_spec}  shapes={[e.shape for e in nuisance_embs]}")

    variant_suffix = args.variant_suffix or "_unimodal"

    results = {}
    for model in models:
        out_model = f"{model}_av_linear_resid{variant_suffix}"
        out_path = emb_path(args.embeddings_dir, out_model, "av", args.bin_sec, args.skip_sec)
        if out_path.is_file() and not args.force:
            log.info(f"[{model}] SKIP -- output exists: {out_path}")
            continue
        try:
            J = load_emb(model, "av")
        except FileNotFoundError as e:
            log.warning(f"[{model}] SKIP -- {e}")
            continue

        R, ms, alpha = linear_residual(J, nuisance_embs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, R.astype(np.float32))
        results[model] = (ms, alpha)
        log.info(f"[{model}] MS={ms:.4f} alpha={alpha:.2e} -> {out_path}")

    log.info("=" * 60)
    log.info("Summary (multimodality score = ||R||^2 / ||J||^2, higher = more unique variance):")
    for model, (ms, alpha) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        log.info(f"  {model:45s} MS={ms:.4f}  alpha={alpha:.2e}")
    log.info(f"Done. {len(results)}/{len(models)} models processed.")


if __name__ == "__main__":
    main()
