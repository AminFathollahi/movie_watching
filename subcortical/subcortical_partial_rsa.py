"""
subcortical/subcortical_partial_rsa.py
=========================================
Subcortical analogue of rsa/partial_rsa.py -- the Move-1 "integration"
best-additive contrast (target AV joint embedding, controlling for its own
audio-only/video-only unimodal RDMs via banded-ridge nuisance projection),
run on subcortical structures instead of cortical surface vertices.

Did NOT exist before this pass: subcortical/ had zero partial-RSA
infrastructure, so scramble/dummy diff analyses there had nothing to
consolidate against (see subcortical/diff_maps.py).

Everything geometry-agnostic (RDM construction, banded-ridge projection,
the residualized searchlight kernel) is reused VERBATIM from rsa/partial_rsa.py
-- those functions operate on RDM vectors and neighbor-index arrays, with no
cortical-specific assumption. Everything subcortical-specific (fMRI source,
per-structure neighbor caches) is reused verbatim from subcortical_rsa.py /
subcortical_io.py. This script is the thin glue between the two, exactly
like subcortical_rsa.py is the thin glue between rsa/searchlight.py and
subcortical_io.py.

Run keys come from the SAME rsa/shared/model_registry.py::PARTIAL_RSA_RUNS
registry used by the cortical script -- target/nuisance are just
(model, modality) embedding lookups, agnostic to which brain structure the
searchlight runs on, so no separate subcortical registry is needed.

Usage
-----
python subcortical/subcortical_partial_rsa.py \\
    --run integration_pe-av-small-16-frame_clsav_from_a \\
    --raw-dir /home/amin/Research/Representation/Movie/data/individual-59k \\
    --subjects-list /home/amin/Research/Representation/Movie/data/subjects.txt \\
    --subject group_average \\
    --timing-csv /home/amin/Research/Representation/Movie/data/movie_timing.csv \\
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \\
    --template-cifti /home/amin/Research/Representation/Movie/outputs/subcortical/subcortical_template.dscalar.nii \\
    --output-dir /home/amin/Research/Representation/Movie/outputs/subcortical \\
    --group-average-dir /home/amin/Research/Representation/Movie/outputs/subcortical/group_average_cache \\
    --neighbor-cache-dir /home/amin/Research/Representation/Movie/outputs/subcortical/_neighbor_cache \\
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman
"""

import argparse
import gc
import logging
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsa"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa.shared.rsa_utils import (  # noqa: E402
    preprocess_fmri, process_model_embeddings, align_and_assert_bins,
    assert_segment_timing,
)
from rsa.shared.model_registry import (  # noqa: E402
    check_embeddings_exist, validate_run,
)
from rsa.partial_rsa import (  # noqa: E402
    _rdm_lower_tri, fit_banded_projection, residualize, run_partial_searchlight,
)
from cifti_io import save_cifti_multimap  # noqa: E402
from subcortical_io import (  # noqa: E402
    SUBCORTICAL_STRUCTURES, STRUCTURE_NEIGHBOR_MODE,
    struct_slices, build_neighbors, get_cerebellum_neighbors,
    preprocess_subject_subcortical, compute_group_average_subcortical,
)
from preprocess_individual import load_subjects  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Key into rsa.shared.model_registry.PARTIAL_RSA_RUNS")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--subjects-list", default=None)
    p.add_argument("--subject", default="group_average")
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--group-average-dir", default=None)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--n-alphas", type=int, default=30, dest="n_alphas")
    p.add_argument("--cv-folds", type=int, default=5, dest="cv_folds")
    p.add_argument("--neighbor-cache-dir", default=None)
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    p.add_argument("--gpu-batch-size", type=int, default=512)
    prep = p.add_argument_group("streaming preprocessing")
    prep.add_argument("--sg-filter", action="store_true", dest="sg_filter")
    prep.add_argument("--psc", action="store_true")
    prep.add_argument("--gsr", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _fmri_tag(args) -> str:
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


def _get_fmri(args, fmri_tag: str):
    prep_args = types.SimpleNamespace(sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr)

    if args.subject == "group_average":
        ga_dir = Path(args.group_average_dir or (Path(args.output_dir) / "group_average_cache"))
        ga_cifti = ga_dir / f"group_average_{fmri_tag}_subcortical.dtseries.nii"
        ga_trs = ga_dir / f"group_average_{fmri_tag}_run_trs.npy"
        if ga_cifti.exists() and ga_trs.exists():
            log.info(f"Group-average subcortical timeseries cached: {ga_cifti}")
            img = nib.load(str(ga_cifti))
            data = img.get_fdata(dtype=np.float32).T
            run_trs = np.load(str(ga_trs))
            return data, run_trs

        if not args.subjects_list:
            raise ValueError("--subjects-list required to build the group average.")
        subjects = load_subjects(args.subjects_list)
        log.info(f"Building group-average subcortical timeseries from {len(subjects)} subjects ...")
        data, _bm_axis, run_trs = compute_group_average_subcortical(
            subjects, Path(args.raw_dir), args.tr, prep_args,
            SUBCORTICAL_STRUCTURES, out_path=ga_cifti,
        )
        return data, run_trs

    log.info(f"Streaming preprocessing subject {args.subject} ...")
    data, _bm_axis, run_trs = preprocess_subject_subcortical(
        args.subject, Path(args.raw_dir), args.tr, prep_args, SUBCORTICAL_STRUCTURES)
    return data, run_trs


def main():
    args = parse_args()
    cfg = validate_run(args.run)
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    fmri_tag = _fmri_tag(args)

    log.info("=" * 70)
    log.info(f"Subcortical partial RSA — {args.run}: {cfg.description}")
    log.info(f"  Target   : {cfg.target}")
    log.info(f"  Nuisance : {cfg.nuisance}")
    log.info("=" * 70)

    bin_int, skip_int, delay_int = int(args.bin_sec), int(args.skip_sec), int(args.delay_sec)
    if cfg.kind != "integration":
        raise ValueError(f"subcortical_partial_rsa.py only supports kind='integration' runs, got {cfg.kind!r}")
    target_model, target_mod = cfg.target
    out_dir = (Path(args.output_dir) / args.subject /
               f"{target_model}_{target_mod}_INTEGRATION" /
               f"k{args.k}_delay{delay_int}s_bin{bin_int}s_skip{skip_int}s_{args.method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "integration_partial_r_searchlight.dscalar.nii"
    if out_path.exists():
        log.info("Output already exists — skipping.")
        return

    timing_df = pd.read_csv(args.timing_csv)
    neighbor_cache_dir = Path(args.neighbor_cache_dir or
                               (ROOT / "outputs" / "subcortical" / "_neighbor_cache"))

    data, run_trs = _get_fmri(args, fmri_tag)
    log.info(f"fMRI continuous: {data.shape}  run_trs={run_trs.tolist()}")

    assert_segment_timing(timing_df, args.bin_sec, args.tr, args.delay_sec, args.skip_sec, run_trs)
    fmri_binned = preprocess_fmri(data, timing_df, run_trs, args.bin_sec, args.tr,
                                   args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    del data
    gc.collect()
    log.info(f"fMRI binned: {fmri_binned.shape}")

    # ── Load and bin target + nuisance embeddings ────────────────────────────
    def load_emb(model, modality):
        path = check_embeddings_exist(args.embeddings_dir, model, modality,
                                       args.bin_sec, args.skip_sec)
        log.info(f"  Loading: {path.name}")
        emb = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
        )
        return emb

    target_emb = load_emb(target_model, target_mod)
    nuis_embs = [load_emb(m, mod) for m, mod in cfg.nuisance]

    fmri_binned, target_emb = align_and_assert_bins(fmri_binned, target_emb)
    n_bins = fmri_binned.shape[0]
    n_total = fmri_binned.shape[1]
    nuis_embs = [e[:n_bins] for e in nuis_embs]

    # ── RDMs + banded-ridge nuisance projection (verbatim from rsa/partial_rsa.py) ──
    log.info("  Computing model RDMs ...")
    y_target = _rdm_lower_tri(target_emb)
    nuis_vecs = [_rdm_lower_tri(e) for e in nuis_embs]
    X_nuis = np.column_stack(nuis_vecs).astype(np.float32)
    log.info(f"  RDM shapes: y_target={y_target.shape}  X_nuis={X_nuis.shape}")
    del target_emb, nuis_embs
    gc.collect()

    log.info("  Fitting banded ridge (per-band alpha via RidgeCV) ...")
    C, best_alphas = fit_banded_projection(X_nuis, y_target, n_alphas=args.n_alphas, cv_folds=args.cv_folds)
    log.info(f"  Per-band alphas: {[f'{a:.2e}' for a in best_alphas]}")
    e_target = residualize(y_target, X_nuis, C)
    log.info(f"  e_target: mean={e_target.mean():.4f}  std={e_target.std():.4f}")

    # ── Per-structure partial searchlight (mirrors subcortical_rsa.py's loop) ──
    slices = struct_slices_from_template(args.template_cifti)
    corr_full = np.zeros(n_total, dtype=np.float32)
    for s in SUBCORTICAL_STRUCTURES:
        info = slices[s]
        start, stop = info["start"], info["stop"]
        n_struct = stop - start
        fmri_struct = fmri_binned[:, start:stop]

        if s.startswith("CEREBELLUM_"):
            hem = "LEFT" if s.endswith("LEFT") else "RIGHT"
            neighbors, mode_used = get_cerebellum_neighbors(
                hem, info["world_xyz"], info["voxel_ijk"], args.k,
                neighbor_cache_dir, args.workbench)
        else:
            mode = STRUCTURE_NEIGHBOR_MODE[s]
            cache_path = neighbor_cache_dir / f"{s}_neighbors_k{args.k}_{mode}.npy"
            neighbors = build_neighbors(info["voxel_ijk"], info["world_xyz"], args.k, mode, cache_path)
            mode_used = mode

        idx = np.arange(n_struct, dtype=np.int32)
        corr_struct = run_partial_searchlight(
            fmri_struct, X_nuis, C, e_target, neighbors,
            surface_indices=idx, vertex_to_col=idx,
            method=args.method, batch_size=args.gpu_batch_size,
        )
        corr_full[start:stop] = corr_struct
        log.info(f"  {s}: n_vox={n_struct}  mode={mode_used}  "
                 f"mean_r={corr_struct.mean():.4f}  max_r={corr_struct.max():.4f}")

    del fmri_binned
    gc.collect()

    map_name = f"partial_rsa_{cfg.label}"
    save_cifti_multimap(corr_full.reshape(1, -1), [map_name], args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}  mean_r={corr_full.mean():.4f}  max_r={corr_full.max():.4f}  "
             f"frac>0={float(np.mean(corr_full > 0)):.2f}")


def struct_slices_from_template(template_cifti: str):
    """subcortical_rsa.py's struct_slices() takes a BrainModelAxis, not a path
    -- reuse it by loading the template's own axis (works for both group-average
    and per-subject templates, since the subcortical grid is subject-invariant).
    """
    from subcortical_io import struct_slices as _struct_slices
    img = nib.load(template_cifti)
    bm_axis = img.header.get_axis(1)
    return _struct_slices(bm_axis)


if __name__ == "__main__":
    main()
