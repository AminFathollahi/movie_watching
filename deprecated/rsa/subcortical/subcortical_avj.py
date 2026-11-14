"""Euclidean-Spearman RSA of the subcortical group average with the audio (A), video (V) and joint (J) embeddings of
each model.

    python subcortical/subcortical_avj.py rsa --models M [M ...] --group-average-dir DIR --output-dir DIR ...

Writes the rho, semi-partial and partial maps of shared/rdm_searchlight.py (--distance euclidean --comparator
spearman); A and V are the model's own (tag unimodal_own). Searchlights are the within-structure neighborhoods of subcortical_rsa.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "subcortical"))
sys.path.insert(0, str(ROOT / "deprecated"))

from shared.cifti_io import get_bm_axis, load_cifti_data, save_cifti_multimap  # noqa: E402
from rsa.shared.rdm_searchlight import (  # noqa: E402
    BRAIN_RDMS, DEFINITIONS, OWN_TAG, model_rdm, partial_correlations, searchlight, semipartial, unit_ranks,
)
from shared.naming import MODEL_NORMS, searchlight_config  # noqa: E402
from shared.binning import assert_segment_timing, preprocess_fmri, process_model_embeddings  # noqa: E402
from subcortical_io import neighbor_columns, struct_slices  # noqa: E402

log = logging.getLogger(__name__)
PREFIX = "subcortical"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["rsa"])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--group-average-dir", required=True)
    p.add_argument("--fmri-suffix", default="raw")
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--neighbor-cache-dir", required=True)
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--model-norm", choices=MODEL_NORMS, default="center")
    return p.parse_args(argv)


def load(args):
    """Binned, per-run z-scored group-average responses (windows, voxels), the A, V, J embeddings of each model and
    the neighbor columns."""
    ga = Path(args.group_average_dir)
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(ga / f"group_average_{args.fmri_suffix}_run_trs.npy")
    assert_segment_timing(timing, args.bin_sec, args.tr, args.delay_sec, args.skip_sec, run_trs)
    fmri = preprocess_fmri(load_cifti_data(str(ga / f"group_average_{args.fmri_suffix}_subcortical.dtseries.nii")),
                           timing, run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    folder = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"
    embs = {}
    for model in args.models:
        sources = (model, model, model)
        for c, kind, source in zip("avj", ("a", "v", "av"), sources):
            emb = process_model_embeddings(str(Path(args.embeddings_dir) / source / folder / f"{source}_{kind}.npy"), timing,
                                           bin_sec=args.bin_sec, tr=args.tr, run_trs=run_trs, delay_sec=args.delay_sec,
                                           hrf=False, skip_sec=args.skip_sec, model_norm=args.model_norm)
            assert len(emb) == len(fmri), (model, kind, emb.shape, fmri.shape)
            embs[(model, c)] = np.nan_to_num(emb.astype(np.float64))
    cols = neighbor_columns(struct_slices(get_bm_axis(args.template_cifti)), args.k, args.neighbor_cache_dir,
                            args.workbench)
    assert len(cols) == fmri.shape[1], (cols.shape, fmri.shape)
    log.info(f"fMRI {fmri.shape}, {len(args.models)} models")
    return fmri, embs, cols


def rsa(args, fmri, embs, cols):
    n = len(fmri)
    tril = np.tril_indices(n, k=-1)
    rdms = {key: model_rdm(x, tril, "euclidean") for key, x in embs.items()}
    targets = []
    for model in args.models:
        raw = [rdms[(model, c)] for c in "avj"]
        targets += raw
    values = searchlight(fmri, targets, cols, np.arange(len(cols)), n)
    cfg = searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec, "euclidean_spearman", args.model_norm)
    for i, model in enumerate(args.models):
        directory = Path(args.output_dir) / "group_average" / model
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"rsa_{PREFIX}_{args.fmri_suffix}_{cfg}_{OWN_TAG}"
        rho = values[3 * i:3 * i + 3]
        ranks = np.stack([unit_ranks(rdms[(model, c)]) for c in "avj"])
        R = ranks @ ranks.T
        for names, data, kind in ((["rho_a", "rho_v", "rho_j"], rho, "models"),
                                  (["sp_a", "sp_v", "sp_j"], semipartial(rho, R), "semipartial"),
                                  (["partial_a", "partial_v", "partial_j"], partial_correlations(rho, R), "partial")):
            save_cifti_multimap(data.astype(np.float32), names, args.template_cifti,
                                str(directory / f"{stem}_{kind}.dscalar.nii"))
        (directory / f"{stem}_provenance.json").write_text(json.dumps({
            "audio": f"{model}_a", "video": f"{model}_v", "joint": f"{model}_av", "tag": OWN_TAG, "k": args.k,
            "bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "delay_sec": args.delay_sec,
            "model_norm": args.model_norm, "n_windows": n, "brain_rdm": BRAIN_RDMS["euclidean"],
            "model_rdm": "Euclidean distance", "searchlight": "k nearest voxels within the structure",
            **{k: DEFINITIONS[k] for k in ("rho", "semipartial", "partial")},
            "component_rank_correlation": {"order": "a, v, j", "matrix": R.tolist()},
        }, indent=2) + "\n")
        log.info(f"Saved {directory / stem}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_args()
    rsa(arguments, *load(arguments))
