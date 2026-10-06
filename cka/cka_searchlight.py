"""Searchlight centered kernel alignment (CKA) between cortical responses and model embeddings.

    python cka/cka_searchlight.py partitions --raw-dir DIR --subjects-list FILE --partitions-dir DIR ...
    python cka/cka_searchlight.py run --models M [M ...] --partitions-dir DIR --output-dir DIR ...
    python cka/cka_searchlight.py noncv --models M [M ...] --partitions-dir DIR --output-dir DIR ...
    python cka/cka_searchlight.py subjects --models M [M ...] --raw-dir DIR --subjects-list FILE --output-dir DIR ...
    python cka/cka_searchlight.py aggregate --models M [M ...] --subjects-list FILE --output-dir DIR ...
    python cka/cka_searchlight.py commonality --models M [M ...] --output-dir DIR ...

`partitions` averages the preprocessed responses of disjoint subject groups. `run` writes the
non-cross-validated, cross-validated and whitened cross-validated CKA maps of the joint, audio and
video embeddings, and the semi-partial and commonality maps; `noncv` only the non-cross-validated
ones. `subjects` writes the cross-validated maps of each subject against the other subjects and
`aggregate` the subject semi-partial and commonality maps, and the mean, standard error and
random-effects maps of all three (see cka/README.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, load_cifti_data, save_cifti_map, save_cifti_multimap
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

    individual = argparse.ArgumentParser(add_help=False)
    individual.add_argument("--subjects-list", required=True)
    individual.add_argument("--limit", type=int, default=0, help="Use only the first N listed subjects.")

    maps = argparse.ArgumentParser(add_help=False)
    maps.add_argument("--models", nargs="+", required=True, help="Joint embedding models; `{model}_av.npy` is used.")
    maps.add_argument("--audio-model", help="Model supplying `_a.npy` for every model (default: the model itself).")
    maps.add_argument("--video-model", help="Model supplying `_v.npy` for every model (default: the model itself).")
    maps.add_argument("--tag", default="unimodal_own", help="Inserted after the scaling in the audio, video and joint output names.")
    maps.add_argument("--embeddings-dir", required=True)
    maps.add_argument("--output-dir", required=True)
    maps.add_argument("--left-surface", required=True)
    maps.add_argument("--right-surface", required=True)
    maps.add_argument("--workbench", required=True)
    maps.add_argument("--geodesic-cache-dir", required=True)
    maps.add_argument("--k", type=int, default=100)
    maps.add_argument("--feature-scaling", choices=MODEL_NORMS, default=DEFAULT_MODEL_NORM)
    maps.add_argument("--max-vertices", type=int, default=0, help="Evaluate only this many evenly spaced grayordinates.")

    part = sub.add_parser("partitions", parents=[common, individual], help="Average disjoint subject groups into partition responses.")
    part.add_argument("--raw-dir", required=True)
    part.add_argument("--checkpoint-every", type=int, default=20)

    run = sub.add_parser("run", parents=[common, maps], help="CKA maps of the joint, audio and video embeddings of each model.")
    run.add_argument("--exclude-video-ids", default=REPEATED_CLIPS, help="Clips dropped for the `norepeats` diagnostic.")
    sub.add_parser("noncv", parents=[run], add_help=False, help="Non-cross-validated maps of `run` only.")
    subjects = sub.add_parser("subjects", parents=[common, individual, maps], help="Cross-validated CKA maps of each subject.")
    subjects.add_argument("--raw-dir", required=True)
    sub.add_parser("aggregate", parents=[common, individual, maps], help="Mean and standard error of the subject maps.")
    sub.add_parser("commonality", parents=[common, maps], help="Commonality maps of the stored group-average CKA maps.")
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


def load_embedding(args, model, kind, timing, run_trs):
    folder = f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    return np.nan_to_num(process_model_embeddings(
        str(Path(args.embeddings_dir) / model / folder / f"{model}_{kind}.npy"), timing, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec,
        model_norm=args.feature_scaling).astype(np.float64))


def load_embeddings(args, timing, run_trs):
    return {model: load_embedding(args, model, "av", timing, run_trs) for model in args.models}


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


def save_map(array, args, directory, method, name, suffix=""):
    stem = map_stem(args, method, suffix)
    directory.mkdir(parents=True, exist_ok=True)
    save_cifti_map(array, args.template_cifti, str(directory / f"{stem}_maps.dscalar.nii"), name)


MEASURES = {
    "cv": "CKA between the window-centered cross-validated Gram matrix of all {m} groups and the component Gram matrix "
          "(whitened unbiased RDM cosine with independent noise, Diedrichsen et al. 2021, eq. 21)",
    "cv-ar": "CKA between the cross-validated Gram matrix of all {m} groups and the component Gram matrix, both whitened "
             "with the inverse square root of the window noise covariance (whitened unbiased RDM cosine with the window "
             "noise covariance, Diedrichsen et al. 2021)",
}
NONCV = "CKA between the window-centered Gram matrix of the group-average responses and the component Gram matrix (noise-biased)"
COMPONENT_MAPS = ["cka_a", "cka_v", "cka_j"]


def component_models(args, model):
    return args.audio_model or model, args.video_model or model


def write_provenance(path, entries):
    path.write_text(json.dumps(entries, indent=2) + "\n")


def save_models(args, cka, model, directory, measure, definition, **entries):
    stem = f"{map_stem(args, measure)}_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    save_cifti_multimap(cka, COMPONENT_MAPS, args.template_cifti, str(directory / f"{stem}_models.dscalar.nii"))
    audio, video = component_models(args, model)
    write_provenance(directory / f"{stem}_provenance.json", {
        "audio_model": audio, "video_model": video, "joint_model": f"{model}_av", "tag": args.tag,
        "k": args.k, "bin_sec": args.bin_sec, "delay_sec": args.delay_sec, "feature_scaling": args.feature_scaling,
        "measure": measure, "cka": definition, **entries,
    })
    return stem


SEMIPARTIAL = ("sp_x: correlation of the brain matrix with the residual of the x matrix after least-squares regression on the "
               "other two component matrices; nothing is fitted to the brain")
SEMIPARTIAL_MAPS = ["sp_a", "sp_v", "sp_j"]


def save_semipartial(args, cka, cosines, directory, stem):
    values = kernels.semipartial(cka, cosines).astype(np.float32)
    save_cifti_multimap(values, SEMIPARTIAL_MAPS, args.template_cifti, str(directory / f"{stem}_semipartial.dscalar.nii"))
    return values


COMMONALITY = ("r2_S = c_S' R_SS^-1 c_S, the squared multiple correlation of the brain matrix with the component matrices in "
               "subset S (c: CKA values, R: cosines among the component matrices); unique_x, shared_*_only and shared_avj by "
               "inclusion-exclusion over the seven r2_S (commonality analysis, Seibold & McPhee 1979); shared_av = r2_a + r2_v - r2_av. "
               "r2_S is quadratic in the CKA values, so noise in them adds a positive bias that is larger for single subjects "
               "than for the group average")
COMMONALITY_MAPS = [f"r2_{k}" for k in ("a", "v", "j", "av", "aj", "vj", "avj")] + [
    "unique_a", "unique_v", "unique_j", "shared_av_only", "shared_aj_only", "shared_vj_only", "shared_avj", "shared_av"]


def save_commonality(args, cka, cosines, directory, stem):
    parts = kernels.commonality(cka, cosines)
    values = np.stack([parts[name] for name in COMMONALITY_MAPS]).astype(np.float32)
    save_cifti_multimap(values, COMMONALITY_MAPS, args.template_cifti, str(directory / f"{stem}_commonality.dscalar.nii"))
    return values


def component_cosines(components, joint, whiten):
    return dict(zip(MEASURES, kernels.component_grams((*components, joint), whiten)[1]))


def semipartial_provenance(cosines):
    norms = kernels.semipartial_coefficients(cosines)[1]
    return dict(
        semipartial=SEMIPARTIAL, component_cosines={"order": "a, v, j", "matrix": cosines.tolist()},
        residual_norms=dict(zip(("sp_a", "sp_v", "sp_j"), norms.tolist())),
        zeroed_semipartial_maps=[f"sp_{c}" for c, n in zip("avj", norms) if n < kernels.RESIDUAL_FLOOR],
    )


def setup(args):
    root = Path(args.output_dir) / args.fmri_suffix
    out, work = root / args.subject, root / "intermediate"
    work.mkdir(parents=True, exist_ok=True)
    binned, timing, run_trs = load_group_average(args)
    embeddings = load_embeddings(args, timing, run_trs)
    videos, runs = kernels.window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)
    assert not np.isnan(binned).any()
    assert len(videos) == binned.shape[0] == next(iter(embeddings.values())).shape[0], (len(videos), binned.shape)
    ncols = kernels.neighbour_columns(args.template_cifti, args.left_surface, args.right_surface, args.workbench,
                                      args.geodesic_cache_dir, args.subject, args.k)
    n_vertices = ncols.shape[0]
    verts = np.linspace(0, n_vertices - 1, args.max_vertices).astype(int) if args.max_vertices else np.arange(n_vertices)
    np.save(work / "vertices.npy", verts)
    components = {
        model: tuple(load_embedding(args, name, kind, timing, run_trs) for name, kind in zip(component_models(args, model), "av"))
        for model in args.models
    }
    return types.SimpleNamespace(root=root, out=out, work=work, binned=binned, timing=timing, run_trs=run_trs, embeddings=embeddings,
                                 components=components, videos=videos, runs=runs, ncols=ncols, verts=verts,
                                 device=kernels.default_device())


def whitener_path(args, work):
    return work / f"{partition_path(args).stem}_whitener.npy"


def all_embeddings(args, s):
    return {(model, c): x for model in args.models for c, x in zip("avj", (*s.components[model], s.embeddings[model]))}


def cross_validated(args, s):
    t0 = time.time()
    ut = kernels.to_vmk(partition_path(args))
    assert ut.shape[0] == s.ncols.shape[0] and ut.shape[2] == len(s.videos), (ut.shape, s.ncols.shape, len(s.videos))
    log.info("partitions loaded %s in %.0fs", ut.shape, time.time() - t0)
    whiten = kernels.noise_model(ut, np.bincount(s.runs, minlength=len(s.run_trs)), s.work, partition_path(args).stem, s.device)
    return kernels.cv_pass(ut, s.ncols, all_embeddings(args, s), whiten, s.verts, s.device), whiten


def noncv(args, s=None):
    s = s or setup(args)
    keep = ~np.isin(s.videos, args.exclude_video_ids.split(","))
    grams = {model: kernels.model_gram(x[keep]) for model, x in s.embeddings.items()}
    for model, array in kernels.noncv_pass(s.binned[keep], grams, s.ncols, s.verts, s.device).items():
        save_map(array, args, s.out / model / "diagnostics", "noncv", "cka", "_norepeats")

    grams = {key: kernels.model_gram(x) for key, x in all_embeddings(args, s).items()}
    results = kernels.noncv_pass(s.binned, grams, s.ncols, s.verts, s.device)
    for model in args.models:
        columns = np.stack([grams[(model, c)] for c in "avj"]).astype(np.float64)
        cosines = columns @ columns.T
        cka = np.stack([results[(model, c)] for c in "avj"])
        stem = save_models(args, cka, model, s.out / model, "noncv", NONCV, n_windows=len(s.videos), commonality=COMMONALITY,
                           **semipartial_provenance(cosines))
        save_semipartial(args, cka, cosines, s.out / model, stem)
        save_commonality(args, cka, cosines, s.out / model, stem)
    log.info("non-cross-validated pass: %d windows, %d without repeats", len(s.videos), keep.sum())
    return s


def run(args):
    s = noncv(args)
    n_windows = len(s.videos)
    t0 = time.time()
    results, whiten = cross_validated(args, s)
    for model in args.models:
        cosines = component_cosines(s.components[model], s.embeddings[model], whiten)
        for measure, definition in MEASURES.items():
            cka = np.stack([results[((model, c), measure)] for c in "avj"])
            stem = save_models(args, cka, model, s.out / model, measure, definition.format(m=args.n_partitions),
                               n_groups=args.n_partitions, n_windows=n_windows, commonality=COMMONALITY, **semipartial_provenance(cosines[measure]))
            save_semipartial(args, cka, cosines[measure], s.out / model, stem)
            save_commonality(args, cka, cosines[measure], s.out / model, stem)
    log.info("done in %.0fs", time.time() - t0)


def commonality(args):
    for model in args.models:
        directory = Path(args.output_dir) / args.fmri_suffix / args.subject / model
        for measure in MEASURES:
            stem = f"{map_stem(args, measure)}_{args.tag}"
            provenance = json.loads((directory / f"{stem}_provenance.json").read_text())
            cka = nib.load(directory / f"{stem}_models.dscalar.nii")
            assert list(cka.header.get_axis(0).name) == COMPONENT_MAPS
            save_commonality(args, np.array(cka.get_fdata(dtype=np.float32)), np.array(provenance["component_cosines"]["matrix"]), directory, stem)
            write_provenance(directory / f"{stem}_provenance.json", {**provenance, "commonality": COMMONALITY})
            log.info("%s %s commonality", model, measure)


SUBJECT_CKA = ("CKA between the cross-validated Gram matrix of one subject, the symmetrized products of the subject's responses "
               "with the summed responses of the other {n} subjects, and the component Gram matrix")


def subjects(args):
    from preprocess_individual import load_subjects, preprocess_subject

    s = setup(args)
    whiten = np.load(whitener_path(args, s.work))
    listed = load_subjects(args.subjects_list)
    fmri = load_cifti_data(str(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"))
    mean = preprocess_fmri(fmri, s.timing, s.run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=False)
    sd = np.concatenate([np.broadcast_to(rows.std(0), rows.shape) for rows in np.split(mean, np.cumsum(np.bincount(s.runs))[:-1])])
    sd = np.where(sd < 1e-10, 1.0, sd)
    total = len(listed) * mean / sd
    columns = [np.concatenate(parts, 1) for parts in zip(*(
        kernels.component_grams((*s.components[model], s.embeddings[model]), whiten)[0] for model in args.models))]
    root = s.root / "subjects"
    stems = {measure: f"{map_stem(args, measure)}_{args.tag}" for measure in MEASURES}
    flags = types.SimpleNamespace(sg_filter=False, psc=False, gsr=False)

    def done(subject):
        return all((root / subject / model / f"{stem}_models.dscalar.nii").exists() for model in args.models for stem in stems.values())

    def load_binned(subject):
        data, _, subject_trs = preprocess_subject(subject, Path(args.raw_dir), args.tr, flags)
        if not np.array_equal(subject_trs, s.run_trs):
            raise ValueError(f"run_trs {subject_trs.tolist()} != {s.run_trs.tolist()}")
        return preprocess_fmri(data, s.timing, subject_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=False) / sd

    pending = [subject for subject in (listed[:args.limit] if args.limit else listed) if not done(subject)]
    log.info("%d subjects listed, %d pending", len(listed), len(pending))
    executor = ThreadPoolExecutor(1)
    queue_ = deque((subject, executor.submit(load_binned, subject)) for subject in pending[:2])
    next_index, failed, t0 = min(2, len(pending)), {}, time.time()
    while queue_:
        subject, future = queue_.popleft()
        if next_index < len(pending):
            queue_.append((pending[next_index], executor.submit(load_binned, pending[next_index])))
            next_index += 1
        try:
            y = future.result()
        except Exception as error:
            failed[subject] = repr(error)
            log.warning("skipped %s: %r", subject, error)
            continue
        cosines, norms = kernels.subject_pass(y.T, (total - y).T, s.ncols, columns, whiten, s.verts, s.device)
        for m, model in enumerate(args.models):
            work = s.work / "subjects" / subject / model
            work.mkdir(parents=True, exist_ok=True)
            for i, (measure, definition) in enumerate(MEASURES.items()):
                np.save(work / f"{stems[measure]}_gram_norm.npy", norms[i])
                save_models(args, cosines[i, 3 * m:3 * m + 3], model, root / subject / model, measure,
                            SUBJECT_CKA.format(n=len(listed) - 1) + ("" if measure == "cv" else ", both whitened"),
                            n_subjects=len(listed), n_windows=len(s.videos))
        log.info("[%d/%d] %s; elapsed %.0fs", pending.index(subject) + 1, len(pending), subject, time.time() - t0)
    executor.shutdown()
    if failed:
        raise RuntimeError(f"{len(failed)} subjects failed: {failed}")


def t_map(x):
    return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(len(x)) + 1e-30)


def fisher_z(values):
    return np.arctanh(np.clip(values.astype(np.float64), -0.999999, 0.999999))


def random_effects(values, names, correlations=True):
    """One-sample t maps over subjects (subjects, maps, grayordinates), on Fisher z for correlations; for the `cka` maps
    also the paired differences with the joint map."""
    z = fisher_z(values) if correlations else values.astype(np.float64)
    out = {f"t_{name}": t_map(z[:, i]) for i, name in enumerate(names)}
    if "cka_j" in names:
        joint = names.index("cka_j")
        for i, name in enumerate(names):
            if i != joint:
                difference = z[:, joint] - z[:, i]
                out[f"diff_j_minus_{name[4:]}"] = difference.mean(0)
                out[f"t_j_minus_{name[4:]}"] = t_map(difference)
    return out


RANDOM_EFFECTS = ("subjects as the random factor, on Fisher z = arctanh of the subject values: t_x = mean / standard error over "
                  "subjects (n - 1 degrees of freedom); in the models file diff_j_minus_x and t_j_minus_x are the mean and t of "
                  "the within-subject difference z_j - z_x")


def aggregate(args):
    from preprocess_individual import load_subjects

    listed = load_subjects(args.subjects_list)
    listed = listed[:args.limit] if args.limit else listed
    root = Path(args.output_dir) / args.fmri_suffix
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    whiten = np.load(whitener_path(args, root / "intermediate"))
    for model in args.models:
        components = tuple(load_embedding(args, name, kind, timing, run_trs) for name, kind in zip(component_models(args, model), "av"))
        cosines = component_cosines(components, load_embedding(args, model, "av", timing, run_trs), whiten)
        directory = root / "subject_mean" / model
        directory.mkdir(parents=True, exist_ok=True)
        for measure in MEASURES:
            stem = f"{map_stem(args, measure)}_{args.tag}"
            cka, semipartial, common = [], [], []
            for subject in listed:
                image = nib.load(root / "subjects" / subject / model / f"{stem}_models.dscalar.nii")
                assert list(image.header.get_axis(0).name) == COMPONENT_MAPS, f"{subject} holds other maps"
                cka.append(image.get_fdata(dtype=np.float32))
                semipartial.append(save_semipartial(args, cka[-1], cosines[measure], root / "subjects" / subject / model, stem))
                common.append(save_commonality(args, cka[-1], cosines[measure], root / "subjects" / subject / model, stem))
            for kind, names, x in (("models", COMPONENT_MAPS, np.stack(cka)), ("semipartial", SEMIPARTIAL_MAPS, np.stack(semipartial)),
                                   ("commonality", COMMONALITY_MAPS, np.stack(common))):
                for suffix, prefix, array in (("", "mean", x.mean(0)), ("_sem", "sem", x.std(0, ddof=1) / np.sqrt(len(listed)))):
                    save_cifti_multimap(array, [f"{prefix}_{name}" for name in names], args.template_cifti,
                                        str(directory / f"{stem}_{kind}{suffix}.dscalar.nii"))
                effects = random_effects(x, names, correlations=kind != "commonality")
                save_cifti_multimap(np.stack(list(effects.values())).astype(np.float32), list(effects), args.template_cifti,
                                    str(directory / f"{stem}_{kind}_random_effects.dscalar.nii"))
            write_provenance(directory / f"{stem}_provenance.json", {
                "n_subjects": len(listed), "subjects": listed, "tag": args.tag, "measure": measure,
                "mean": "mean over subjects of the subject maps", "sem": "standard deviation over subjects / sqrt(n_subjects)",
                "random_effects": RANDOM_EFFECTS + "; commonality maps are not correlations and use the raw values",
                "commonality": COMMONALITY, **semipartial_provenance(cosines[measure]),
            })
            log.info("%s %s: %d subjects", model, measure, len(listed))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    {"partitions": partitions, "run": run, "noncv": noncv, "subjects": subjects, "aggregate": aggregate,
     "commonality": commonality}[arguments.command](arguments)
