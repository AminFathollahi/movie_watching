"""Searchlight centered kernel alignment (CKA) between cortical responses and model embeddings.

    python cka/cka_searchlight.py partitions --raw-dir DIR --subjects-list FILE --partitions-dir DIR ...
    python cka/cka_searchlight.py run --models M [M ...] --partitions-dir DIR --output-dir DIR ...

`partitions` averages the preprocessed responses of disjoint subject groups. `run` writes the
non-cross-validated, cross-validated and whitened cross-validated CKA maps, their log-likelihoods
and the log-likelihood differences between models (see cka/README.md).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, load_cifti_data, save_cifti_map
from cka.shared import kernels
from rsa.shared.naming import DEFAULT_MODEL_NORM, MODEL_NORMS, searchlight_config
from rsa.shared.rsa_utils import preprocess_fmri, process_model_embeddings

log = logging.getLogger(__name__)
REPEATED_CLIPS = "video5,video9,video14,video18"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preprocessed-dir", required=True)
    common.add_argument("--fmri-suffix", default="raw")
    common.add_argument("--subject", default="group_average")
    common.add_argument("--timing-csv", required=True)
    common.add_argument("--template-cifti", required=True)
    common.add_argument("--partitions-dir", required=True)
    common.add_argument("--n-partitions", type=int, default=25)
    common.add_argument("--bin-sec", type=float, default=5.0)
    common.add_argument("--skip-sec", type=float, default=5.0)
    common.add_argument("--delay-sec", type=float, default=5.0)
    common.add_argument("--tr", type=float, default=1.0)

    part = sub.add_parser("partitions", parents=[common], help="Average disjoint subject groups into partition responses.")
    part.add_argument("--raw-dir", required=True)
    part.add_argument("--subjects-list", required=True)
    part.add_argument("--limit", type=int, default=0, help="Use only the first N listed subjects.")
    part.add_argument("--checkpoint-every", type=int, default=20)

    run = sub.add_parser("run", parents=[common], help="Compute the CKA maps for each model.")
    run.add_argument("--models", nargs="+", required=True, help="Embedding models; `{model}_av.npy` is used.")
    run.add_argument("--embeddings-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--left-surface", required=True)
    run.add_argument("--right-surface", required=True)
    run.add_argument("--workbench", required=True)
    run.add_argument("--geodesic-cache-dir", required=True)
    run.add_argument("--k", type=int, default=100)
    run.add_argument("--feature-scaling", choices=MODEL_NORMS, default=DEFAULT_MODEL_NORM)
    run.add_argument("--exclude-video-ids", default=REPEATED_CLIPS, help="Clips dropped for the `norepeats` diagnostic.")
    run.add_argument("--max-vertices", type=int, default=0, help="Evaluate only this many evenly spaced grayordinates.")
    return parser


def partition_path(args) -> Path:
    stem = f"partitions_M{args.n_partitions}_delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    return Path(args.partitions_dir) / f"{stem}.npy"


def map_stem(args, method, suffix=""):
    config = searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec, method, args.feature_scaling)
    return f"cka_59k_{args.fmri_suffix}_{config}{suffix}"


def load_group_average(args):
    prefix = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}"
    run_trs = np.load(f"{prefix}_run_trs.npy")
    fmri = load_cifti_data(f"{prefix}_cortex_59k.dtseries.nii")
    timing = pd.read_csv(args.timing_csv)
    binned = preprocess_fmri(fmri, timing, run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    return binned, timing, run_trs


def load_embeddings(args, timing, run_trs):
    folder = f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    return {
        model: np.nan_to_num(process_model_embeddings(
            str(Path(args.embeddings_dir) / model / folder / f"{model}_av.npy"), timing, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec,
            model_norm=args.feature_scaling).astype(np.float64))
        for model in args.models
    }


def partitions(args):
    from preprocess_individual import RUN_IDS, get_run_path, load_subjects, preprocess_subject

    n_groups = args.n_partitions
    final = partition_path(args)
    final.parent.mkdir(parents=True, exist_ok=True)
    checkpoint, checkpoint_json = final.with_name(final.stem + "_ckpt.npy"), final.with_name(final.stem + "_ckpt.json")
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy").astype(np.int32)
    _, runs = kernels.window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)
    run_windows = np.bincount(runs, minlength=len(run_trs))
    n_windows = int(run_windows.sum())
    reference = get_bm_axis(args.template_cifti)
    ref_name, ref_vertex, n_vertices = np.array(reference.name), np.array(reference.vertex), len(reference.name)

    listed = load_subjects(args.subjects_list)
    subjects = listed[:args.limit] if args.limit else listed
    raw = Path(args.raw_dir)
    skipped, available = {}, []
    for s in subjects:
        missing = [r for r in RUN_IDS if not get_run_path(raw, s, r).exists()]
        if missing:
            skipped[s] = f"missing runs {missing}"
        else:
            available.append(s)
    group = {s: i % n_groups for i, s in enumerate(available)}
    if checkpoint.exists() and checkpoint_json.exists():
        state = json.loads(checkpoint_json.read_text())
        assert state["group"] == group and state["K"] == n_windows, "checkpoint does not match this configuration"
        sums = np.load(checkpoint)
        done, counts, times = state["done"], np.array(state["counts"]), state["times"]
        skipped.update(state["skipped"])
        log.info("resumed from checkpoint: %d subjects done", len(done))
    else:
        sums = np.zeros((n_groups, n_windows, n_vertices), np.float32)
        done, counts, times = [], np.zeros(n_groups, int), {}
    flags = types.SimpleNamespace(sg_filter=False, psc=False, gsr=False)

    def load_binned(s):
        t = time.time()
        data, axis, subject_trs = preprocess_subject(s, raw, args.tr, flags)
        if not np.array_equal(subject_trs, run_trs):
            raise ValueError(f"run_trs {subject_trs.tolist()} != {run_trs.tolist()}")
        if not (np.array_equal(np.array(axis.name), ref_name) and np.array_equal(np.array(axis.vertex), ref_vertex)):
            raise ValueError("cortex axis differs from template")
        binned = preprocess_fmri(data, timing, subject_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=False)
        return binned, time.time() - t

    def save_checkpoint():
        np.save(str(checkpoint) + ".tmp.npy", sums)
        os.replace(str(checkpoint) + ".tmp.npy", checkpoint)
        checkpoint_json.write_text(json.dumps({"group": group, "K": n_windows, "done": done, "counts": counts.tolist(),
                                               "times": times, "skipped": skipped}))

    pending = [s for s in available if s not in done and s not in skipped]
    log.info("available %d, pending %d, skipped %s, windows %d, per run %s", len(available), len(pending), skipped, n_windows, run_windows.tolist())
    executor = ThreadPoolExecutor(1)
    queue_ = deque((s, executor.submit(load_binned, s)) for s in pending[:2])
    next_index, n_new, t0 = min(2, len(pending)), 0, time.time()
    while queue_:
        s, future = queue_.popleft()
        if next_index < len(pending):
            queue_.append((pending[next_index], executor.submit(load_binned, pending[next_index])))
            next_index += 1
        try:
            binned, seconds = future.result()
            assert binned.shape == (n_windows, n_vertices), binned.shape
        except Exception as error:
            skipped[s] = repr(error)
            log.warning("skipped %s: %r", s, error)
            continue
        g = group[s]
        sums[g] += binned
        counts[g] += 1
        done.append(s)
        times[s] = round(seconds, 1)
        n_new += 1
        log.info("[%d/%d] %s group %d load+bin %.1fs; elapsed %.0fs", len(done), len(available), s, g, seconds, time.time() - t0)
        if n_new % args.checkpoint_every == 0:
            save_checkpoint()
    executor.shutdown()
    assert (counts > 0).all(), counts
    save_checkpoint()
    n_subjects = int(counts.sum())
    log.info("loaded %d subjects; group sizes %s", n_subjects, counts.tolist())

    chunk = 4096
    for c0 in range(0, n_vertices, chunk):
        c1 = min(n_vertices, c0 + chunk)
        full = sums[:, :, c0:c1].sum(0, dtype=np.float64) / n_subjects
        sd = np.empty_like(full)
        offset = 0
        for n in run_windows:
            sd[offset:offset + n] = full[offset:offset + n].std(0)
            offset += n
        sd[sd < 1e-10] = 1.0
        for g in range(n_groups):
            sums[g, :, c0:c1] = (sums[g, :, c0:c1] / counts[g] / sd).astype(np.float32)

    group_average, _, _ = load_group_average(args)
    corr = np.empty(n_vertices)
    for c0 in range(0, n_vertices, chunk):
        c1 = min(n_vertices, c0 + chunk)
        mean = sums[:, :, c0:c1].mean(0, dtype=np.float64)
        ref = group_average[:, c0:c1].astype(np.float64)
        mean -= mean.mean(0)
        ref -= ref.mean(0)
        corr[c0:c1] = (mean * ref).sum(0) / np.sqrt((mean ** 2).sum(0) * (ref ** 2).sum(0) + 1e-30)
    log.info("per-vertex correlation of the partition mean with the group average: median %.6f, min %.6f, 1st percentile %.6f",
             np.median(corr), corr.min(), np.percentile(corr, 1))
    np.save(str(final) + ".tmp.npy", sums)
    os.replace(str(final) + ".tmp.npy", final)
    final.with_suffix(".json").write_text(json.dumps({
        "subject_to_group": {s: group[s] for s in done}, "K": n_windows, "run_trs": run_trs.tolist(), "run_windows": run_windows.tolist(),
        "n_partitions": n_groups, "n_subjects_listed": len(subjects), "n_subjects_loaded": n_subjects, "group_sizes": counts.tolist(),
        "skipped": skipped, "load_seconds": times, "validation_median": float(np.median(corr)), "validation_min": float(corr.min()),
    }, indent=1))
    checkpoint.unlink(missing_ok=True)
    checkpoint_json.unlink(missing_ok=True)
    log.info("saved %s %s", final, sums.shape)


def save_map(array, args, directory, work, method, name, suffix=""):
    stem = map_stem(args, method, suffix)
    np.save(work / f"{stem}.npy", array)
    save_cifti_map(array, args.template_cifti, str(directory / f"{stem}_maps.dscalar.nii"), name)


def run(args):
    device = kernels.default_device()
    out = Path(args.output_dir) / args.fmri_suffix / args.subject
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    binned, timing, run_trs = load_group_average(args)
    embeddings = load_embeddings(args, timing, run_trs)
    videos, runs = kernels.window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)
    n_windows = len(videos)
    assert not np.isnan(binned).any()
    assert n_windows == binned.shape[0] == next(iter(embeddings.values())).shape[0], (n_windows, binned.shape)
    ncols = kernels.neighbour_columns(args.template_cifti, args.left_surface, args.right_surface, args.workbench,
                                      args.geodesic_cache_dir, args.subject, args.k)
    n_vertices = ncols.shape[0]
    verts = np.linspace(0, n_vertices - 1, args.max_vertices).astype(int) if args.max_vertices else np.arange(n_vertices)
    np.save(work / "vertices.npy", verts)
    directories = {model: out / f"{model}_av" for model in args.models}
    for model, directory in directories.items():
        (directory / "diagnostics").mkdir(parents=True, exist_ok=True)
        (work / f"{model}_av").mkdir(exist_ok=True)

    keep = ~np.isin(videos, args.exclude_video_ids.split(","))
    for suffix, rows in (("", slice(None)), ("_norepeats", keep)):
        grams = {model: kernels.model_gram(x[rows]) for model, x in embeddings.items()}
        maps = kernels.noncv_pass(binned[rows], grams, ncols, verts, device)
        for model, array in maps.items():
            save_map(array, args, directories[model] / "diagnostics", work / f"{model}_av", "noncv", "cka", suffix)
        log.info("non-cross-validated pass%s: %d windows", suffix, binned[rows].shape[0])

    t0 = time.time()
    ut = kernels.to_vmk(partition_path(args))
    assert ut.shape[0] == n_vertices and ut.shape[2] == n_windows, (ut.shape, n_vertices, n_windows)
    log.info("partitions loaded %s in %.0fs", ut.shape, time.time() - t0)
    whiten = kernels.noise_model(ut, np.bincount(runs, minlength=len(run_trs)), work, partition_path(args).stem, device)
    results = kernels.cv_pass(ut, ncols, embeddings, whiten, verts, device)
    for model in args.models:
        for measure in ("cv", "cv-ar"):
            r = results[(model, measure)]
            save_map(r, args, directories[model], work / f"{model}_av", measure, f"cka-{measure}")
            save_map(kernels.loglik(r, n_windows), args, directories[model], work / f"{model}_av", f"{measure}-loglik", f"cka-{measure}-loglik")
    for a, b in itertools.combinations(args.models, 2):
        for measure in ("cv", "cv-ar"):
            diff = kernels.loglik_diff(results[(a, measure)], results[(b, measure)], n_windows)
            save_map(diff, args, out, work, f"{measure}-loglik-diff_{a}_av_minus_{b}_av", f"cka-{measure}-loglik-diff")
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    {"partitions": partitions, "run": run}[arguments.command](arguments)
