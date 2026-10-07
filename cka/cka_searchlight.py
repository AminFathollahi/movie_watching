"""Searchlight centered kernel alignment (CKA) between cortical responses and model embeddings.

    python cka/cka_searchlight.py run --models M [M ...] --output-dir DIR ...
    python cka/cka_searchlight.py subjects --models M [M ...] --raw-dir DIR --subjects-list FILE --output-dir DIR ...
    python cka/cka_searchlight.py aggregate --models M [M ...] --subjects-list FILE --output-dir DIR ...
    python cka/cka_searchlight.py noise-ceiling --raw-dir DIR --subjects-list FILE --cache-dir DIR --output-dir DIR ...

`run` writes the CKA maps of the joint, audio and video embeddings with the group-average responses,
and the semi-partial and commonality maps. `subjects` writes the same maps from each subject's own
responses, `aggregate` the mean and standard error of the subject maps and `noise-ceiling` the
bounds on the subject-mean CKA (see cka/README.md).
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

from cifti_io import load_cifti_data, save_cifti_map, save_cifti_multimap
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
    common.add_argument("--bin-sec", type=float, default=5.0)
    common.add_argument("--skip-sec", type=float, default=5.0)
    common.add_argument("--delay-sec", type=float, default=5.0)
    common.add_argument("--tr", type=float, default=1.0)

    individual = argparse.ArgumentParser(add_help=False)
    individual.add_argument("--subjects-list", required=True)
    individual.add_argument("--limit", type=int, default=0, help="Use only the first N listed subjects.")

    searchlight = argparse.ArgumentParser(add_help=False)
    searchlight.add_argument("--output-dir", required=True)
    searchlight.add_argument("--left-surface", required=True)
    searchlight.add_argument("--right-surface", required=True)
    searchlight.add_argument("--workbench", required=True)
    searchlight.add_argument("--geodesic-cache-dir", required=True)
    searchlight.add_argument("--k", type=int, default=100)
    searchlight.add_argument("--max-vertices", type=int, default=0, help="Evaluate only this many evenly spaced grayordinates.")

    maps = argparse.ArgumentParser(add_help=False, parents=[searchlight])
    maps.add_argument("--models", nargs="+", required=True, help="Joint embedding models; `{model}_av.npy` is used.")
    maps.add_argument("--audio-model", help="Model supplying `_a.npy` for every model (default: the model itself).")
    maps.add_argument("--video-model", help="Model supplying `_v.npy` for every model (default: the model itself).")
    maps.add_argument("--tag", default="unimodal_own", help="Inserted after the scaling in the audio, video and joint output names.")
    maps.add_argument("--embeddings-dir", required=True)
    maps.add_argument("--feature-scaling", choices=MODEL_NORMS, default=DEFAULT_MODEL_NORM)

    run = sub.add_parser("run", parents=[common, maps], help="Group-average CKA maps of the joint, audio and video embeddings of each model.")
    run.add_argument("--exclude-video-ids", default=REPEATED_CLIPS, help="Clips dropped for the `norepeats` diagnostic.")
    subjects = sub.add_parser("subjects", parents=[common, individual, maps], help="CKA maps of each subject.")
    subjects.add_argument("--raw-dir", required=True)
    sub.add_parser("aggregate", parents=[common, individual, maps], help="Mean and standard error of the subject maps.")
    ceiling = sub.add_parser("noise-ceiling", parents=[common, individual, searchlight], help="Lower and upper bounds of the subject-mean CKA.")
    ceiling.add_argument("--raw-dir", required=True)
    ceiling.add_argument("--cache-dir", required=True, help="Binned subject responses (float16, about 136 MB per subject).")
    return parser


def map_stem(args, method, suffix=""):
    config = searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec, method, args.feature_scaling)
    return f"cka_59k_{args.fmri_suffix}_{config}{suffix}"


def load_embedding(args, model, kind, timing, run_trs):
    folder = f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    return np.nan_to_num(process_model_embeddings(
        str(Path(args.embeddings_dir) / model / folder / f"{model}_{kind}.npy"), timing, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec,
        model_norm=args.feature_scaling).astype(np.float64))


def save_map(array, args, directory, method, name, suffix=""):
    stem = map_stem(args, method, suffix)
    directory.mkdir(parents=True, exist_ok=True)
    save_cifti_map(array, args.template_cifti, str(directory / f"{stem}_maps.dscalar.nii"), name)


NONCV = "CKA between the window-centered Gram matrix of the group-average responses and the component Gram matrix (noise-biased)"
SUBJECT_NONCV = ("CKA between the window-centered Gram matrix of one subject's responses, z-scored per run, and the component "
                 "Gram matrix (noise-biased)")
COMPONENT_MAPS = ["cka_a", "cka_v", "cka_j"]


def component_models(args, model):
    return args.audio_model or model, args.video_model or model


def write_provenance(path, entries):
    path.write_text(json.dumps(entries, indent=2) + "\n")


def save_models(args, cka, model, directory, definition, **entries):
    stem = f"{map_stem(args, 'noncv')}_{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    save_cifti_multimap(cka, COMPONENT_MAPS, args.template_cifti, str(directory / f"{stem}_models.dscalar.nii"))
    audio, video = component_models(args, model)
    write_provenance(directory / f"{stem}_provenance.json", {
        "audio_model": audio, "video_model": video, "joint_model": f"{model}_av", "tag": args.tag,
        "k": args.k, "bin_sec": args.bin_sec, "delay_sec": args.delay_sec, "feature_scaling": args.feature_scaling,
        "measure": "noncv", "cka": definition, **entries,
    })
    return stem


SEMIPARTIAL = ("sp_x: correlation of the brain matrix with the residual of the x matrix after least-squares regression on the "
               "other two component matrices; nothing is fitted to the brain")
SEMIPARTIAL_MAPS = ["sp_a", "sp_v", "sp_j"]
COMMONALITY = ("r2_S = c_S' R_SS^-1 c_S, the squared multiple correlation of the brain matrix with the component matrices in "
               "subset S (c: CKA values, R: cosines among the component matrices); unique_x, shared_*_only and shared_avj by "
               "inclusion-exclusion over the seven r2_S (commonality analysis, Seibold & McPhee 1979); shared_av = r2_a + r2_v - r2_av. "
               "r2_S is quadratic in the CKA values, so noise in them adds a positive bias that is larger for single subjects "
               "than for the group average")
COMMONALITY_MAPS = [f"r2_{k}" for k in ("a", "v", "j", "av", "aj", "vj", "avj")] + [
    "unique_a", "unique_v", "unique_j", "shared_av_only", "shared_aj_only", "shared_vj_only", "shared_avj", "shared_av"]
KINDS = {"models": COMPONENT_MAPS, "semipartial": SEMIPARTIAL_MAPS, "commonality": COMMONALITY_MAPS}


def semipartial_provenance(cosines):
    norms = kernels.semipartial_coefficients(cosines)[1]
    return dict(
        semipartial=SEMIPARTIAL, component_cosines={"order": "a, v, j", "matrix": cosines.tolist()},
        residual_norms=dict(zip(("sp_a", "sp_v", "sp_j"), norms.tolist())),
        zeroed_semipartial_maps=[f"sp_{c}" for c, n in zip("avj", norms) if n < kernels.RESIDUAL_FLOOR],
    )


def save_maps(args, results, grams, root, definition, **entries):
    """CKA, semi-partial and commonality maps of each model from `noncv_pass` results, under root/{model}."""
    for model in args.models:
        columns = np.stack([grams[(model, c)] for c in "avj"]).astype(np.float64)
        cosines = columns @ columns.T
        cka = np.stack([results[(model, c)] for c in "avj"])
        directory = root / model
        stem = save_models(args, cka, model, directory, definition, **entries, commonality=COMMONALITY,
                           **semipartial_provenance(cosines))
        save_cifti_multimap(kernels.semipartial(cka, cosines).astype(np.float32), SEMIPARTIAL_MAPS, args.template_cifti,
                            str(directory / f"{stem}_semipartial.dscalar.nii"))
        parts = kernels.commonality(cka, cosines)
        save_cifti_multimap(np.stack([parts[name] for name in COMMONALITY_MAPS]).astype(np.float32), COMMONALITY_MAPS,
                            args.template_cifti, str(directory / f"{stem}_commonality.dscalar.nii"))


def searchlights(args):
    ncols = kernels.neighbour_columns(args.template_cifti, args.left_surface, args.right_surface, args.workbench,
                                      args.geodesic_cache_dir, args.subject, args.k)
    n_vertices = ncols.shape[0]
    verts = np.linspace(0, n_vertices - 1, args.max_vertices).astype(int) if args.max_vertices else np.arange(n_vertices)
    return ncols, verts


def setup(args):
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    embeddings = {}
    for model in args.models:
        sources = (*zip(component_models(args, model), "av"), (model, "av"))
        embeddings.update({(model, c): load_embedding(args, name, kind, timing, run_trs) for c, (name, kind) in zip("avj", sources)})
    videos, _ = kernels.window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)
    assert all(len(x) == len(videos) for x in embeddings.values()), len(videos)
    ncols, verts = searchlights(args)
    return types.SimpleNamespace(root=Path(args.output_dir) / args.fmri_suffix, timing=timing, run_trs=run_trs, embeddings=embeddings,
                                 videos=videos, ncols=ncols, verts=verts, device=kernels.default_device())


def run(args):
    s = setup(args)
    fmri = load_cifti_data(str(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"))
    binned = preprocess_fmri(fmri, s.timing, s.run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    assert not np.isnan(binned).any() and len(binned) == len(s.videos), (binned.shape, len(s.videos))
    out = s.root / args.subject

    keep = ~np.isin(s.videos, args.exclude_video_ids.split(","))
    grams = {model: kernels.model_gram(s.embeddings[(model, "j")][keep]) for model in args.models}
    for model, array in kernels.noncv_pass(binned[keep], grams, s.ncols, s.verts, s.device).items():
        save_map(array, args, out / model / "diagnostics", "noncv", "cka", "_norepeats")

    grams = {key: kernels.model_gram(x) for key, x in s.embeddings.items()}
    save_maps(args, kernels.noncv_pass(binned, grams, s.ncols, s.verts, s.device), grams, out, NONCV, n_windows=len(s.videos))
    log.info("%d windows, %d without repeats", len(s.videos), keep.sum())


def load_subject(args, subject, timing, run_trs):
    """One subject's binned responses, z-scored per run; constant grayordinates set to zero."""
    from preprocess_individual import preprocess_subject

    data, _, subject_trs = preprocess_subject(subject, Path(args.raw_dir), args.tr, types.SimpleNamespace(sg_filter=False, psc=False, gsr=False))
    if not np.array_equal(subject_trs, run_trs):
        raise ValueError(f"run_trs {subject_trs.tolist()} != {run_trs.tolist()}")
    binned = preprocess_fmri(data, timing, subject_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    return np.nan_to_num(binned), int(np.isnan(binned).any(0).sum())


def prefetched(items, load):
    """Yield (item, result or exception) with the next item loading in the background."""
    executor = ThreadPoolExecutor(1)
    futures = deque(executor.submit(load, item) for item in items[:2])
    for i, item in enumerate(items):
        if i + 2 < len(items):
            futures.append(executor.submit(load, items[i + 2]))
        try:
            yield item, futures.popleft().result()
        except Exception as error:
            yield item, error
    executor.shutdown()


def subjects(args):
    from preprocess_individual import load_subjects

    s = setup(args)
    listed = load_subjects(args.subjects_list)
    grams = {key: kernels.model_gram(x) for key, x in s.embeddings.items()}
    root = s.root / "subjects"
    stem = f"{map_stem(args, 'noncv')}_{args.tag}"

    def done(subject):
        return all((root / subject / model / f"{stem}_commonality.dscalar.nii").exists() for model in args.models)

    pending = [subject for subject in (listed[:args.limit] if args.limit else listed) if not done(subject)]
    log.info("%d subjects listed, %d pending", len(listed), len(pending))
    failed, t0 = {}, time.time()
    for i, (subject, result) in enumerate(prefetched(pending, lambda subject: load_subject(args, subject, s.timing, s.run_trs))):
        if isinstance(result, Exception):
            failed[subject] = repr(result)
            log.warning("skipped %s: %r", subject, result)
            continue
        y, constant = result
        assert len(y) == len(s.videos), (y.shape, len(s.videos))
        save_maps(args, kernels.noncv_pass(y, grams, s.ncols, s.verts, s.device), grams, root / subject, SUBJECT_NONCV,
                  n_windows=len(s.videos), constant_grayordinates_zeroed=constant)
        log.info("[%d/%d] %s; %d constant grayordinates; elapsed %.0fs", i + 1, len(pending), subject, constant, time.time() - t0)
    if failed:
        raise RuntimeError(f"{len(failed)} subjects failed: {failed}")


def pool(values, names):
    """Mean and standard error over subjects of (subjects, maps, grayordinates); with the CKA maps also the fraction of
    subjects with cka_j above cka_a and above cka_v."""
    mean = {f"mean_{name}": x for name, x in zip(names, values.mean(0))}
    if "cka_j" in names:
        joint = values[:, names.index("cka_j")]
        mean.update({f"frac_j_gt_{c}": (joint > values[:, names.index(f"cka_{c}")]).mean(0) for c in "av"})
    sem = {f"sem_{name}": x for name, x in zip(names, values.std(0, ddof=1) / np.sqrt(len(values)))}
    return {"": mean, "_sem": sem}


POOLED = {"mean": "mean over subjects of the subject maps", "sem": "standard deviation over subjects (n - 1) / sqrt(n_subjects)",
          "frac_j_gt_x": "fraction of subjects with cka_j > cka_x"}


def aggregate(args):
    from preprocess_individual import load_subjects

    listed = load_subjects(args.subjects_list)
    listed = listed[:args.limit] if args.limit else listed
    root = Path(args.output_dir) / args.fmri_suffix
    stem = f"{map_stem(args, 'noncv')}_{args.tag}"
    for model in args.models:
        directory = root / "subject_mean" / model
        directory.mkdir(parents=True, exist_ok=True)
        folders = [root / "subjects" / subject / model for subject in listed]
        for kind, names in KINDS.items():
            values = []
            for folder in folders:
                image = nib.load(folder / f"{stem}_{kind}.dscalar.nii")
                assert list(image.header.get_axis(0).name) == names, f"{folder} holds other maps"
                values.append(image.get_fdata(dtype=np.float32))
            for suffix, maps in pool(np.stack(values), names).items():
                save_cifti_multimap(np.stack(list(maps.values())).astype(np.float32), list(maps), args.template_cifti,
                                    str(directory / f"{stem}_{kind}{suffix}.dscalar.nii"))
        first = json.loads((folders[0] / f"{stem}_provenance.json").read_text())
        first.pop("constant_grayordinates_zeroed", None)
        write_provenance(directory / f"{stem}_provenance.json", {**first, "n_subjects": len(listed), "subjects": listed, **POOLED})
        log.info("%s: %d subjects", model, len(listed))


CEILING = ("bounds on the mean over subjects of the subject CKA (Nili et al. 2014): with b_s the unit-norm, window-centered "
           "Gram matrix of subject s (responses binned and z-scored per run) and B the mean of the b_s, nc_upper = mean_s cos(b_s, B) "
           "and nc_lower = mean_s cos(b_s, mean of the other subjects' b_t)")


def noise_ceiling(args):
    from preprocess_individual import load_subjects

    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    n_windows = len(kernels.window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)[0])
    listed = load_subjects(args.subjects_list)
    listed = listed[:args.limit] if args.limit else listed
    config = f"delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def path(subject):
        return cache / f"{subject}_{args.fmri_suffix}_{config}.npy"

    pending = [subject for subject in listed if not path(subject).exists()]
    log.info("%d subjects, %d to cache", len(listed), len(pending))
    t0 = time.time()
    for i, (subject, result) in enumerate(prefetched(pending, lambda subject: load_subject(args, subject, timing, run_trs))):
        if isinstance(result, Exception):
            raise RuntimeError(f"{subject}: {result!r}")
        y, constant = result
        assert len(y) == n_windows, (y.shape, n_windows)
        np.save(cache / "partial.npy", np.ascontiguousarray(y.T, np.float16))
        os.replace(cache / "partial.npy", path(subject))
        log.info("[%d/%d] cached %s; %d constant grayordinates; elapsed %.0fs", i + 1, len(pending), subject, constant, time.time() - t0)

    ncols, verts = searchlights(args)
    t0 = time.time()
    lower, upper = kernels.noise_ceiling_pass([np.load(path(subject), mmap_mode="r") for subject in listed], ncols, verts,
                                              kernels.default_device())
    log.info("noise ceiling of %d subjects in %.0fs", len(listed), time.time() - t0)
    out = Path(args.output_dir) / args.fmri_suffix / "noise_ceiling"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cka_59k_{args.fmri_suffix}_k{args.k}_{config}_noncv_noise_ceiling"
    save_cifti_multimap(np.stack([lower, upper]), ["nc_lower", "nc_upper"], args.template_cifti, str(out / f"{stem}.dscalar.nii"))
    write_provenance(out / f"{stem}_provenance.json", {
        "k": args.k, "bin_sec": args.bin_sec, "delay_sec": args.delay_sec, "n_windows": n_windows, "noise_ceiling": CEILING,
        "n_subjects": len(listed), "subjects": listed, "responses": "stored as float16, computed in float32",
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    {"run": run, "subjects": subjects, "aggregate": aggregate, "noise-ceiling": noise_ceiling}[arguments.command](arguments)
