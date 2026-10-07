#!/usr/bin/env python3
"""Compare searchlight RSA, CKA and encoding maps on the group-average cortical axis.

    python reports/measure_comparison/compare_measures.py --models pe-av-small-16-frame nemotron_layer18_mp

Reads existing maps only (see reports/measure_comparison/README.md) and writes one markdown report.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, load_cifti_data
from cka.shared.kernels import neighbour_columns, window_index
from rsa.shared.naming import searchlight_config, searchlight_stem
from rsa.shared.rsa_utils import preprocess_fmri

PROJECT = ROOT.parent
GLASSER = "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
ENCODING_MODEL_MAPS = ("r2_a", "r2_v", "r2_j", "r2_av", "r2_avj")
SPLITS = ("loro",)
CORES = (("loro unique_j", "loro:unique_j", "loro:unique_j"),
         ("loro r2_avj", "loro:r2_avj", "loro:unique_j"))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=None, help="Default: bin-sec.")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--scaling", choices=("center", "zscore"), default="center")
    p.add_argument("--encoding-scaling", choices=("demean", "zscore"), default="demean")
    p.add_argument("--encoding-tag", default="unimodal_own")
    p.add_argument("--fmri-suffix", default="raw")
    p.add_argument("--subject", default="group_average")
    p.add_argument("--rsa-dir", default=str(PROJECT / "outputs/rsa"))
    p.add_argument("--cka-dir", default=str(PROJECT / "outputs/cka"))
    p.add_argument("--cka-tag", default="unimodal_own")
    p.add_argument("--encoding-dir", default=str(PROJECT / "outputs/encoding"))
    p.add_argument("--data-dir", default=str(PROJECT / "data"))
    p.add_argument("--geodesic-cache-dir", default=None, help="Default: {rsa-dir}/_geodesic_cache.")
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    p.add_argument("--core-percent", type=float, default=5.0)
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "measure_comparison.md"))
    return p


def md_table(title, header, rows):
    lines = [title, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def load_maps(path, axis):
    img = nib.load(str(path))
    found = img.header.get_axis(1)
    assert np.array_equal(found.name, axis.name) and np.array_equal(found.vertex, axis.vertex), f"axis mismatch: {path}"
    return dict(zip(img.header.get_axis(0).name, img.get_fdata().astype(np.float64)))


def map_paths(args, model):
    config = lambda method: searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec, method, args.scaling)
    rsa = lambda method, suffix="": f"{searchlight_stem(args.fmri_suffix, args.k, args.delay_sec, args.bin_sec, args.skip_sec, method, args.scaling)}{suffix}_maps.dscalar.nii"
    cka = lambda method, suffix="": f"cka_59k_{args.fmri_suffix}_{config(method)}{suffix}_maps.dscalar.nii"
    group = lambda base: Path(base) / args.fmri_suffix / args.subject
    rsa_dir, cka_dir = group(args.rsa_dir) / f"{model}_av", group(args.cka_dir) / model
    models = lambda method: f"cka_59k_{args.fmri_suffix}_{config(method)}_{args.cka_tag}_models.dscalar.nii"
    return {
        "corr-spearman": rsa_dir / rsa("spearman"),
        "euclid-spearman": rsa_dir / "diagnostics" / rsa("euclid-spearman"),
        "cka": cka_dir / models("noncv"),
        "corr-spearman_nr": rsa_dir / "diagnostics" / rsa("corr-spearman", "_norepeats"),
        "cka_nr": cka_dir / "diagnostics" / cka("noncv", "_norepeats"),
    }


def load_model_maps(args, model, axis):
    paths = map_paths(args, model)
    maps = {}
    for name, path in paths.items():
        found = load_maps(path, axis)
        maps[name] = found.get("cka_j", next(iter(found.values())))
    folder = Path(args.encoding_dir) / args.subject / model / f"delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    encoding = {}
    for split in SPLITS:
        stem = f"encoding_r2_{split}_{args.encoding_scaling}_{args.encoding_tag}"
        models, partition = load_maps(folder / f"{stem}_models.dscalar.nii", axis), load_maps(folder / f"{stem}_partition.dscalar.nii", axis)
        encoding.update({f"{split}:{name}": models[name] for name in ENCODING_MODEL_MAPS})
        encoding[f"{split}:unique_j"] = partition["unique_j"]
    return maps, encoding


def parcel_labels(path, axis):
    dlabel = nib.load(str(path))
    codes = dlabel.get_fdata()[0].astype(int)
    names = {k: v[0] for k, v in dlabel.header.get_axis(0).label[0].items()}
    dense = dlabel.header.get_axis(1)
    position = {key: i for i, key in enumerate(zip(dense.name, dense.vertex))}
    return np.array([codes[position[key]] for key in zip(axis.name, axis.vertex)]), names


def top_parcels(values, labels, names, largest=True, n=10):
    means = [(names[code], values[labels == code].mean()) for code in np.unique(labels) if code > 0]
    return sorted(means, key=lambda r: -r[1] if largest else r[1])[:n]


def neighbourhood_homogeneity(fmri, ncols, chunk=1024):
    n_windows, n_vertices = fmri.shape
    z = fmri - fmri.mean(0, keepdims=True)
    z = z / np.maximum(np.sqrt((z ** 2).mean(0, keepdims=True)), 1e-10)
    padded = np.concatenate([z, np.zeros((n_windows, 1), z.dtype)], 1)
    h = np.zeros(n_vertices)
    for start in range(0, n_vertices, chunk):
        cols = ncols[start:start + chunk]
        size = (cols >= 0).sum(1)
        total = padded[:, np.where(cols >= 0, cols, n_vertices)].sum(2).astype(np.float64)
        ok = size >= 2
        h[start:start + chunk][ok] = (((total ** 2).sum(0) - size * n_windows) / (np.maximum(size * (size - 1), 1) * n_windows))[ok]
    return h


def core_bins(core, ncols):
    count = (core[np.clip(ncols, 0, None)] & (ncols >= 0)).sum(1)
    return -(-count * 10 // ncols.shape[1])


def model_tables(model, maps, encoding, labels, names, ncols, h, n_windows, args):
    rsa = maps
    allm = {**rsa, **encoding}
    columns = list(allm)
    ranks = {k: rankdata(v) for k, v in allm.items()}
    n = len(labels)
    out = [f"\n## Model: {model}\n"]
    out.append(md_table(f"Table 1. Mean, 95th percentile and maximum of each map over {n} grayordinates.", ["map", "mean", "95th percentile", "max"],
                        [[k, f"{v.mean():.4f}", f"{np.percentile(v, 95):.4f}", f"{v.max():.4f}"] for k, v in rsa.items()]))
    out.append(md_table(f"Table 3a. Spatial Pearson correlation across {n} grayordinates of each map (row) with every RSA, CKA and encoding map (column; encoding = split:map).",
                        ["map"] + columns, [[a] + [f"{np.corrcoef(allm[a], allm[b])[0, 1]:.3f}" for b in columns] for a in rsa]))
    out.append(md_table("Table 3b. Same as 3a with Spearman rank correlation.",
                        ["map"] + columns, [[a] + [f"{np.corrcoef(ranks[a], ranks[b])[0, 1]:.3f}" for b in columns] for a in rsa]))
    tops = {k: top_parcels(v, labels, names) for k, v in rsa.items()}
    out.append(md_table("Table 4. Top 10 Glasser parcels by parcel mean for each map (parcel name and mean).", ["rank"] + list(rsa),
                        [[i + 1] + [f"{tops[k][i][0]} {tops[k][i][1]:.4f}" for k in rsa] for i in range(10)]))
    for title, core_key, unique_key in CORES:
        core = allm[core_key] >= np.percentile(allm[core_key], 100 - args.core_percent)
        bins = core_bins(core, ncols)
        rows = []
        for i in range(11):
            m = bins == i
            rows.append(["f = 0" if i == 0 else f"({(i - 1) / 10:.1f}, {i / 10:.1f}]", int(m.sum())]
                        + [f"{v[m].mean():.4f}" if m.any() else "NA" for v in rsa.values()] + [f"{allm[unique_key][m].mean():.4f}" if m.any() else "NA"])
        out.append(md_table(f"Table 5 ({title}). Ring diagnostic: core = top {args.core_percent:g}% of {core_key} ({int(core.sum())} grayordinates); f = fraction of the {ncols.shape[1]} searchlight neighbours in the core (medial-wall neighbours count as not in core); mean of each map and of {unique_key} per f bin.",
                            ["f bin", "n"] + list(rsa) + [unique_key], rows))
    out.append(md_table(f"Table 6a. Spatial correlation of each map with neighbourhood homogeneity h, the mean pairwise Pearson correlation of the {ncols.shape[1]} neighbours' binned time courses.",
                        ["map", "Pearson with h", "Spearman with h"],
                        [[k, f"{np.corrcoef(v, h)[0, 1]:.3f}", f"{np.corrcoef(ranks[k], rankdata(h))[0, 1]:.3f}"] for k, v in rsa.items()]))
    decile = np.floor(rankdata(h) / (n + 1) * 10).astype(int)
    out.append(md_table("Table 6b. Mean of each map per decile of h (decile 1 = lowest h).", ["h decile", "mean h", "n"] + list(rsa),
                        [[d + 1, f"{h[decile == d].mean():.4f}", int((decile == d).sum())] + [f"{v[decile == d].mean():.4f}" for v in rsa.values()] for d in range(10)]))
    rows = []
    for full, dropped in (("corr-spearman", "corr-spearman_nr"), ("cka", "cka_nr")):
        x, y = rsa[full], rsa[dropped]
        change = lambda f: f"{f(x):.4f} -> {f(y):.4f} ({f(y) - f(x):+.4f})"
        rows.append([full, f"{np.corrcoef(x, y)[0, 1]:.4f}", change(np.mean), change(lambda v: np.percentile(v, 95)), change(np.max)])
    out.append(md_table("Table 7. Repeated-clip effect: all windows against the windows with the four repeated clips dropped; spatial Pearson correlation of the two maps and the change in mean, 95th percentile and maximum.",
                        ["measure", "Pearson r, all vs dropped", "mean", "95th percentile", "max"], rows))
    return out


def difference_tables(args, axis, labels, names):
    out = []
    for a, b in itertools.combinations(args.models, 2):
        first, second = (load_model_maps(args, model, axis)[0] for model in (a, b))
        diffs = {"cka": first["cka"] - second["cka"]}
        out.append(f"\n## Model difference: {a} minus {b}\n")
        out.append(md_table(f"Table 8. Difference of the joint-embedding maps, {a} minus {b}; positive favours {a}. Mean, 5th and 95th percentile, number of grayordinates with a positive and a negative difference (n = {len(labels)}).",
                            ["map", "mean", "5th percentile", "95th percentile", "n > 0", "n < 0"],
                            [[k, f"{v.mean():.4f}", f"{np.percentile(v, 5):.4f}", f"{np.percentile(v, 95):.4f}", int((v > 0).sum()), int((v < 0).sum())] for k, v in diffs.items()]))
        for k, v in diffs.items():
            favour_a, favour_b = top_parcels(v, labels, names), top_parcels(v, labels, names, largest=False)
            out.append(md_table(f"Table 9 ({k}). Top 10 Glasser parcels by parcel mean of the difference, favouring {a} (highest) and {b} (lowest).", ["rank", f"{a} favoured", f"{b} favoured"],
                                [[i + 1, f"{favour_a[i][0]} {favour_a[i][1]:.4f}", f"{favour_b[i][0]} {favour_b[i][1]:.4f}"] for i in range(10)]))
    return out


def main():
    args = build_parser().parse_args()
    args.skip_sec = args.bin_sec if args.skip_sec is None else args.skip_sec
    data = Path(args.data_dir)
    average = data / "preprocessed/average_sub" / args.fmri_suffix
    template = average / f"group_average_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    axis = get_bm_axis(str(template))
    labels, names = parcel_labels(data / "HCP_S1200_GroupAvg_v1" / GLASSER, axis)
    timing = pd.read_csv(data / "movie_timing.csv")
    run_trs = np.load(average / f"group_average_{args.fmri_suffix}_run_trs.npy")
    fmri = preprocess_fmri(load_cifti_data(str(template)), timing, run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    n_windows = len(window_index(timing, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)[0])
    assert fmri.shape[0] == n_windows, (fmri.shape, n_windows)
    surfaces = data / "HCP_S1200_GroupAvg_v1/GroupAverage_59k"
    ncols = neighbour_columns(str(template), str(surfaces / "CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"),
                              str(surfaces / "CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"), args.workbench,
                              args.geodesic_cache_dir or str(Path(args.rsa_dir) / "_geodesic_cache"), args.subject, args.k)
    h = neighbourhood_homogeneity(fmri, ncols)
    report = [f"# Measure comparison: RSA, CKA and encoding (group average, bin {args.bin_sec:g} s, skip {args.skip_sec:g} s, delay {args.delay_sec:g} s, k = {args.k}, model {args.scaling}, encoding {args.encoding_scaling}, tag {args.encoding_tag})\n"]
    for model in args.models:
        maps, encoding = load_model_maps(args, model, axis)
        report += model_tables(model, maps, encoding, labels, names, ncols, h, n_windows, args)
    report += difference_tables(args, axis, labels, names)
    Path(args.output).write_text("\n".join(report))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
