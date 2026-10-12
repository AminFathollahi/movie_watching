"""Group statistics over subjects of the per-subject Euclidean RSA maps of rsa/rdm_searchlight.py --subjects-list.

For each pair and baseline (aligned: the subject maps; minus_shifted: the subject maps minus those computed with each
run's brain windows circularly shifted by half the run), one file per kind, as for the group-average maps (models:
rho_x; semipartial: sp_x; partial: partial_x). For each map x it holds x_mean (mean over subjects), x_t (one-sample t
against 0), x_sigmap_uncorr and x_sigmap_fdr (sign of the mean times -log10 of the one-tailed p for a positive mean,
uncorrected and Benjamini-Hochberg corrected) and x_fdr_mask; for the maps named in --tfce also x_tfce, x_sigmap_tfce_fwe
and x_tfce_fwe_mask (threshold-free cluster enhancement of the Fisher-z maps with sign-flip permutations, family-wise
error at --alpha; Smith & Nichols 2009). TFCE maps already in an output file are kept when not recomputed.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap
from rsa.group_stats import _filter_clusters, _load_surface, _surface_adjacency, one_sample
from rsa.rdm_searchlight import KINDS, OWN_TAG, subject_config, subject_map_path
from rsa.tfce_groupstats import _compute_tfce_sigmap, tfce_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASELINES = {"aligned": "", "minus_shifted": "_minus_shifted"}
STATS = ["mean", "t", "sigmap_uncorr", "sigmap_fdr", "fdr_mask"]
TFCE_STATS = ["tfce", "sigmap_tfce_fwe", "tfce_fwe_mask"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="RSA output root holding subject_data/.")
    p.add_argument("--subjects-list", required=True)
    p.add_argument("--model", default="pe-av-small-16-frame")
    p.add_argument("--tags", nargs="+", default=[OWN_TAG, "mae", "text-asr"])
    p.add_argument("--baselines", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    p.add_argument("--fmri-suffix", default="raw")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=None)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--model-norm", choices=["zscore", "center"], default="center")
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--left-surface", required=True)
    p.add_argument("--right-surface", required=True)
    p.add_argument("--workbench", default=None, help="wb_command, for .border files of the masks.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--tfce", nargs="*", default=[], help="Maps (e.g. rho_j sp_j) to also test with TFCE.")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def write_borders(masks, out_stem, hemis, min_cluster_size, workbench):
    """One .border file per hemisphere with a border per mask (clusters of at least min_cluster_size), named after the
    mask; masks that are empty or cover more than 90% of the hemisphere are left out."""
    written = {}
    for tag, sl, vertices, n_full, surface, adj in hemis:
        columns = {}
        for name, mask in masks.items():
            m = _filter_clusters(mask[sl] > 0.5, adj, min_cluster_size)
            if 0 < m.sum() <= 0.9 * len(m):
                full = np.zeros(n_full, np.float32)
                full[vertices] = m
                columns[name] = full
        out = Path(f"{out_stem}_{tag}.border")
        out.unlink(missing_ok=True)
        if not columns:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            metric = Path(tmp) / "masks.func.gii"
            nib.save(GiftiImage(darrays=[GiftiDataArray(x, intent=0, datatype="NIFTI_TYPE_FLOAT32", meta={"Name": n})
                                         for n, x in columns.items()]), str(metric))
            subprocess.run([workbench, "-metric-rois-to-border", surface, str(metric), "clusters", str(out)],
                           check=True, capture_output=True, text=True)
        written[tag] = sorted(columns)
    return written


def main():
    args = parse_args()
    skip_sec = args.skip_sec if args.skip_sec is not None else args.bin_sec
    cfg = subject_config(args.k, args.delay_sec, args.bin_sec, skip_sec, args.model_norm)
    from preprocess_individual import load_subjects
    subjects = load_subjects(args.subjects_list)

    left_idx, right_idx = get_cortex_vertex_indices(get_bm_axis(args.template_cifti))
    n_left, n_grays = len(left_idx), len(left_idx) + len(right_idx)
    hemis = []
    for tag, surface, idx, sl in (("lh", args.left_surface, left_idx, slice(0, n_left)),
                                  ("rh", args.right_surface, right_idx, slice(n_left, n_grays))):
        faces, n_full = _load_surface(surface)
        hemis.append((tag, sl, np.asarray(idx), n_full, surface, _surface_adjacency(faces, np.asarray(idx), len(idx))))
    adj_lh, adj_rh = hemis[0][5], hemis[1][5]

    directory = Path(args.output_dir) / "groupstats" / args.model
    directory.mkdir(parents=True, exist_ok=True)

    def stack(tag, kind, shifted):
        """(maps of the kind, subjects, grayordinates)."""
        out = []
        for s in subjects:
            image = nib.load(str(subject_map_path(args.output_dir, s, args.model, tag, kind, cfg, args.fmri_suffix,
                                                  shifted)))
            assert list(image.header.get_axis(0).name[:3]) == KINDS[kind], (s, kind)
            out.append(np.asarray(image.dataobj[:3], dtype=np.float32))
        return np.ascontiguousarray(np.stack(out, axis=1))

    for tag in args.tags:
        for baseline in args.baselines:
            stem = directory / f"rsa_59k_{args.fmri_suffix}_{cfg}_{tag}{BASELINES[baseline]}"
            summary_path = Path(f"{stem}_summary.json")
            summary = json.loads(summary_path.read_text())["maps"] if summary_path.exists() else {}
            for kind, names in KINDS.items():
                path = Path(f"{stem}_{kind}.dscalar.nii")
                old = {}
                if path.exists():
                    image = nib.load(str(path))
                    old = dict(zip(image.header.get_axis(0).name, np.array(image.get_fdata(dtype=np.float32))))
                maps, fdr, tfce, nulls = {}, {}, {}, {}
                subject_maps = stack(tag, kind, False)
                if baseline == "minus_shifted":
                    subject_maps = subject_maps - stack(tag, kind, True)
                for name, x in zip(names, subject_maps):
                    stats, _ = one_sample(x, args.alpha)
                    maps.update({f"{name}_{k}": stats[k] for k in STATS})
                    fdr[name] = stats["fdr_mask"]
                    entry = summary.get(name, {})
                    entry.update(n_fdr=int(stats["fdr_mask"].sum()), mean_over_grayordinates=float(stats["mean"].mean()),
                                 fraction_positive_mean=float((stats["mean"] > 0).mean()))
                    if name in args.tfce:
                        log.info(f"{stem.name} {name}: TFCE, {args.n_permutations} permutations")
                        z = np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)).astype(np.float64)
                        t_stat, h0, thresh, mask = tfce_test(z, adj_lh, adj_rh, args.n_permutations, args.n_jobs,
                                                             args.alpha, args.min_cluster_size)
                        sig = (_compute_tfce_sigmap(t_stat, h0, stats["mean"], len(h0)) if len(h0)
                               else np.zeros(n_grays, np.float32))
                        maps.update({f"{name}_tfce": t_stat.astype(np.float32), f"{name}_sigmap_tfce_fwe": sig,
                                     f"{name}_tfce_fwe_mask": mask})
                        nulls[name] = h0
                        entry.update(tfce_fwe_threshold=thresh, n_tfce_fwe=int(mask.sum()),
                                     n_permutations=args.n_permutations)
                    elif f"{name}_tfce" in old:
                        maps.update({f"{name}_{k}": old[f"{name}_{k}"] for k in TFCE_STATS})
                    if f"{name}_tfce_fwe_mask" in maps:
                        tfce[name] = maps[f"{name}_tfce_fwe_mask"]
                    summary[name] = entry
                save_cifti_multimap(np.stack(list(maps.values())), list(maps), args.template_cifti, str(path))
                if nulls:
                    null_path = Path(f"{stem}_{kind}_tfce_null.npz")
                    kept = dict(np.load(null_path)) if null_path.exists() else {}
                    np.savez(null_path, **{**kept, **nulls})
                if args.workbench:
                    for test, masks in (("fdr", fdr), ("tfce", tfce)):
                        write_borders(masks, f"{stem}_{kind}_{test}", hemis, args.min_cluster_size, args.workbench)
                log.info(f"Saved {path.name}")
            summary_path.write_text(json.dumps({
                "model": args.model, "tag": tag, "baseline": baseline, "config": cfg,
                "n_subjects": len(subjects), "subjects": subjects, "alpha": args.alpha,
                "min_cluster_size": args.min_cluster_size, "maps": summary,
                "t": "one-sample t over subjects against 0 of the subject maps (Spearman correlations, no Fisher z)",
                "tfce": "TFCE (start 0, step 0.2) of the one-sample t of the Fisher-z subject maps, sign-flip "
                        "permutations, one-tailed, family-wise error from the maximum TFCE over grayordinates",
                "minus_shifted": "subject map minus the subject map with each run's brain windows circularly shifted "
                                 "by half the run",
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
