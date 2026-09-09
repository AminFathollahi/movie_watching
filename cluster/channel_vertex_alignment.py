"""Correlate cortical vertex-cluster and transformer channel-cluster temporal profiles on the shared 626-bin movie timeline.

The two clusterings live on disjoint index sets (grayordinates vs. embedding
channels), so label overlap is meaningless. Each cluster's mean time series
on the common bin grid is the only comparable quantity: this script bins the
raw group-average dtseries the same way ``rsa``/``encoding`` scripts do,
reuses ``channel_timeseries_clustering.load_channel_timeseries`` for the
channel side, correlates every vertex-cluster profile against every
channel-cluster profile, and tests each pair against a circular-shift null
(movie time series are strongly autocorrelated, so an i.i.d. permutation null
would be invalid).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from io_cluster import load_group_average, GROUP_AVG_CIFTI, GROUP_AVG_TRS  # noqa: E402
from vertex_clustering import zscore_timeseries_inplace  # noqa: E402
from channel_timeseries_clustering import (  # noqa: E402
    FAMILIES, OUTPUT_DIR, EMBEDDINGS_DIR, TIMING_CSV, RUN_TRS,
    channel_model_selection_dir, load_channel_timeseries,
)
from rsa.shared.rsa_utils import preprocess_fmri, align_and_assert_bins  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPECTED_N_BINS = 626


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family", default="peav", choices=FAMILIES)
    parser.add_argument("--vertex-cifti", default=GROUP_AVG_CIFTI)
    parser.add_argument("--vertex-run-trs", default=GROUP_AVG_TRS)
    parser.add_argument("--run-trs", default=RUN_TRS, help="Channel-side run TR counts")
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--timing-csv", default=TIMING_CSV)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--vertex-model-selection-dir", default=None,
                        help="Defaults to outputs/cluster/group_average/_vertex/norm-zscore_raw")
    parser.add_argument("--channel-model-selection-dir", default=None,
                        help="Defaults to outputs/cluster/<family>/_channel_timeseries_model_selection"
                             "/norm-zscore_prepca50")
    parser.add_argument("--vertex-role", default="latent_best")
    parser.add_argument("--channel-role", default="latent_best")
    parser.add_argument("--vertex-reducer-tag", default=None,
                        help="Explicit reducer_tag override; requires --vertex-cluster-tag")
    parser.add_argument("--vertex-cluster-tag", default=None,
                        help="Explicit full_fit_cluster_tag override")
    parser.add_argument("--channel-reducer-tag", default=None,
                        help="Explicit reducer_tag override; requires --channel-cluster-tag")
    parser.add_argument("--channel-cluster-tag", default=None,
                        help="Explicit full_fit_cluster_tag override")
    parser.add_argument("--n-shifts", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def select_config(selected_csv: Path, role: str, reducer_tag: str | None,
                  cluster_tag: str | None) -> pd.Series:
    """Pick the winning row from a model-selection ``selected_clusterings.csv``."""
    table = pd.read_csv(selected_csv)
    if reducer_tag is not None or cluster_tag is not None:
        if reducer_tag is None or cluster_tag is None:
            raise ValueError("Both reducer_tag and cluster_tag must be given together")
        match = table[(table["reducer_tag"] == reducer_tag)
                      & (table["full_fit_cluster_tag"] == cluster_tag)]
        if match.empty:
            raise ValueError(f"No row for reducer_tag={reducer_tag} cluster_tag={cluster_tag} in {selected_csv}")
        return match.iloc[0]
    subset = table[table["selection_role"] == role]
    if subset.empty:
        raise ValueError(f"No rows with selection_role={role!r} in {selected_csv}")
    return subset.loc[subset["selection_score"].idxmax()]


def labels_path(model_selection_dir: Path, row: pd.Series, filename: str) -> Path:
    reducer_tag = str(row["reducer_tag"])
    cluster_tag = str(row["full_fit_cluster_tag"])
    return (model_selection_dir / "selected_maps" / reducer_tag
            / f"{reducer_tag}_{cluster_tag}" / filename)


def cluster_profiles(units_by_bins: np.ndarray, labels: np.ndarray,
                     exclude: tuple[int, ...] = (-1,)
                     ) -> tuple[np.ndarray, list[int], dict[int, int], int]:
    """Z-scored mean time series per cluster, excluding ``exclude`` label values.

    Channel labels use the sklearn convention (-1 = noise); vertex labels use
    ``vertex_clustering.expand_masked_labels``'s convention (0 = outside the
    stimulus mask, plus a reserved noise key) — pass ``exclude`` accordingly,
    see :func:`vertex_exclude_labels`.
    """
    if units_by_bins.shape[0] != labels.shape[0]:
        raise ValueError("labels length does not match unit rows")
    n_excluded = int(np.isin(labels, exclude).sum())
    cluster_ids = sorted(int(c) for c in np.unique(labels) if c not in exclude)
    if not cluster_ids:
        raise ValueError("No clusters found")
    profiles = np.stack(
        [units_by_bins[labels == cid].mean(axis=0) for cid in cluster_ids]
    ).astype(np.float32)
    counts = {cid: int((labels == cid).sum()) for cid in cluster_ids}
    zscore_timeseries_inplace(profiles)
    return profiles, cluster_ids, counts, n_excluded


def vertex_exclude_labels(labels_file: Path) -> tuple[int, ...]:
    """Vertex label values to exclude from clustering: 0 (outside the
    stimulus mask) plus the reserved HDBSCAN noise key recorded in the
    sibling ``spatial_report.json``, if any."""
    exclude = [0]
    report_file = labels_file.parent / "spatial_report.json"
    if report_file.exists():
        noise_label = json.loads(report_file.read_text()).get("noise_label")
        if noise_label is not None:
            exclude.append(int(noise_label))
    return tuple(exclude)


def _zscore_1d(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64) - v.astype(np.float64).mean()
    s = v.std()
    return (v / s if s > 0 else v).astype(np.float32)


def regress_out_global(profiles: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """Residualize each z-scored profile row against one zero-mean regressor, then re-zscore."""
    beta = (profiles @ regressor) / (regressor @ regressor)
    residual = (profiles - beta[:, None] * regressor[None, :]).astype(np.float32)
    zscore_timeseries_inplace(residual)
    return residual


def profile_correlation_summary(profiles: np.ndarray) -> dict[str, Any]:
    """Mean/max |off-diagonal correlation| among a side's own cluster profiles.

    Close to 1 means the clusters are temporally redundant (the partition
    barely differentiates in time), regardless of the sign of the relation.
    """
    n = profiles.shape[0]
    if n < 2:
        return {"mean_abs_off_diag": None, "max_abs_off_diag": None}
    corr = (profiles @ profiles.T) / profiles.shape[1]
    off = corr[~np.eye(n, dtype=bool)]
    return {"mean_abs_off_diag": float(np.mean(np.abs(off))), "max_abs_off_diag": float(np.max(np.abs(off)))}


def circular_shift_pvalues(vertex_profiles: np.ndarray, channel_profiles: np.ndarray,
                           n_shifts: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided circular-shift null on Pearson r between every profile pair.

    Both inputs are already z-scored per row (zero mean, unit population
    variance), so Pearson r reduces to a dot product over n_bins.
    """
    n_bins = vertex_profiles.shape[1]
    if channel_profiles.shape[1] != n_bins:
        raise ValueError("profile bin counts differ between sides")
    observed = (vertex_profiles @ channel_profiles.T) / n_bins
    abs_observed = np.abs(observed)
    rng = np.random.default_rng(random_state)
    exceed = np.zeros_like(observed, dtype=np.int64)
    for _ in range(n_shifts):
        shift = int(rng.integers(1, n_bins))
        shifted = np.roll(channel_profiles, shift, axis=1)
        null_r = (vertex_profiles @ shifted.T) / n_bins
        exceed += np.abs(null_r) >= abs_observed
    pvals = (exceed + 1) / (n_shifts + 1)
    return observed, pvals


def plot_heatmap(observed: np.ndarray, qvals: np.ndarray, vertex_ids: list[int],
                 channel_ids: list[int], out_path: Path, title: str, alpha: float) -> None:
    import matplotlib.pyplot as plt

    n_v, n_c = observed.shape
    figure, axis = plt.subplots(figsize=(max(6, 0.5 * n_c + 2), max(4, 0.5 * n_v + 2)), dpi=160)
    vmax = float(np.abs(observed).max()) or 1.0
    image = axis.imshow(observed, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_xticks(range(n_c))
    axis.set_xticklabels([f"ch{c}" for c in channel_ids], rotation=90)
    axis.set_yticks(range(n_v))
    axis.set_yticklabels([f"vx{v}" for v in vertex_ids])
    axis.set_xlabel("Channel cluster")
    axis.set_ylabel("Vertex cluster")
    for vi in range(n_v):
        for ci in range(n_c):
            marker = "*" if qvals[vi, ci] < alpha else ""
            axis.text(ci, vi, f"{observed[vi, ci]:.2f}{marker}", ha="center", va="center", fontsize=7)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Pearson r")
    axis.set_title(f"{title}\n* = BH-FDR q<{alpha}", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def _write_alignment_outputs(out_dir: Path, suffix: str, observed: np.ndarray, pvals: np.ndarray,
                             qvals: np.ndarray, vertex_ids: list[int], channel_ids: list[int],
                             vertex_counts: dict[int, int], channel_counts: dict[int, int],
                             tag: str, alpha: float) -> int:
    matrix_df = pd.DataFrame(
        observed,
        index=[f"vertex_cluster_{v}" for v in vertex_ids],
        columns=[f"channel_cluster_{c}" for c in channel_ids],
    )
    matrix_df.to_csv(out_dir / f"alignment_matrix{suffix}.csv")

    long_rows = []
    for vi, v_id in enumerate(vertex_ids):
        for ci, c_id in enumerate(channel_ids):
            long_rows.append({
                "vertex_cluster": v_id, "channel_cluster": c_id,
                "n_vertices": vertex_counts[v_id], "n_channels": channel_counts[c_id],
                "r": float(observed[vi, ci]), "p": float(pvals[vi, ci]), "q": float(qvals[vi, ci]),
            })
    pd.DataFrame(long_rows).sort_values("q").to_csv(out_dir / f"alignment_long{suffix}.csv", index=False)
    plot_heatmap(observed, qvals, vertex_ids, channel_ids,
                out_dir / f"alignment_heatmap{suffix}.png", tag + suffix, alpha)
    return int((qvals < alpha).sum())


def _config_tag(row: pd.Series, n_clusters: int, prefix: str) -> str:
    return f"{prefix}-{row['reduction_method']}{int(row['n_components'])}-{row['method']}{n_clusters}"


def _config_summary(row: pd.Series, ms_dir: Path, labels_file: Path,
                    n_clusters: int, n_noise: int) -> dict[str, Any]:
    return {
        "model_selection_dir": str(ms_dir),
        "reducer_tag": str(row["reducer_tag"]),
        "full_fit_cluster_tag": str(row["full_fit_cluster_tag"]),
        "reduction_method": str(row["reduction_method"]),
        "n_components": int(row["n_components"]),
        "cluster_method": str(row["method"]),
        "selection_role": str(row["selection_role"]),
        "selection_score": float(row["selection_score"]),
        "labels_file": str(labels_file),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


def run(args: argparse.Namespace) -> Path:
    vertex_ms_dir = Path(args.vertex_model_selection_dir) if args.vertex_model_selection_dir else (
        Path(args.output_dir) / "group_average" / "_vertex" / "norm-zscore_raw"
    )
    channel_ms_dir = Path(args.channel_model_selection_dir) if args.channel_model_selection_dir else (
        channel_model_selection_dir(args.output_dir, args.family)
    )

    vertex_row = select_config(vertex_ms_dir / "selected_clusterings.csv", args.vertex_role,
                               args.vertex_reducer_tag, args.vertex_cluster_tag)
    channel_row = select_config(channel_ms_dir / "selected_clusterings.csv", args.channel_role,
                                args.channel_reducer_tag, args.channel_cluster_tag)

    vertex_labels_file = labels_path(vertex_ms_dir, vertex_row, "spatial_vertex_labels.npy")
    channel_labels_file = labels_path(channel_ms_dir, channel_row, "channel_labels.npy")
    vertex_labels = np.load(vertex_labels_file)
    channel_labels = np.load(channel_labels_file)
    log.info("Vertex config: %s / %s", vertex_row["reducer_tag"], vertex_row["full_fit_cluster_tag"])
    log.info("Channel config: %s / %s", channel_row["reducer_tag"], channel_row["full_fit_cluster_tag"])

    log.info("Loading group-average dtseries: %s", args.vertex_cifti)
    X, vertex_run_trs, _ = load_group_average(args.vertex_cifti, args.vertex_run_trs)
    if X.shape[0] != vertex_labels.shape[0]:
        raise ValueError(
            f"Vertex label count {vertex_labels.shape[0]} != grayordinate count {X.shape[0]}"
        )
    timing_df = pd.read_csv(args.timing_csv)
    fmri_binned = preprocess_fmri(
        fmri_continuous=X, timing_df=timing_df, run_trs=vertex_run_trs,
        bin_sec=args.bin_sec, tr=args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    del X

    log.info("Loading channel time series for family: %s", args.family)
    channel_timeseries, _ = load_channel_timeseries(args)
    if channel_timeseries.shape[0] != channel_labels.shape[0]:
        raise ValueError(
            f"Channel label count {channel_labels.shape[0]} != channel count {channel_timeseries.shape[0]}"
        )

    align_and_assert_bins(fmri_binned, channel_timeseries.T)
    n_bins = fmri_binned.shape[0]
    if n_bins != EXPECTED_N_BINS:
        raise AssertionError(f"Expected {EXPECTED_N_BINS} movie bins, got {n_bins}")

    vertex_profiles, vertex_ids, vertex_counts, vertex_noise = cluster_profiles(
        fmri_binned.T, vertex_labels, exclude=vertex_exclude_labels(vertex_labels_file)
    )
    channel_profiles, channel_ids, channel_counts, channel_noise = cluster_profiles(
        channel_timeseries, channel_labels
    )
    log.info("Vertex clusters: %d (noise=%d)  Channel clusters: %d (noise=%d)",
             len(vertex_ids), vertex_noise, len(channel_ids), channel_noise)

    observed, pvals = circular_shift_pvalues(
        vertex_profiles, channel_profiles, args.n_shifts, args.random_state
    )
    qvals = false_discovery_control(pvals.ravel(), method="bh").reshape(pvals.shape)

    # global-signal control: regress the cortex-wide mean bin time series out of every
    # vertex- and channel-cluster profile, then re-test on the residuals
    cortex_global = _zscore_1d(fmri_binned.mean(axis=1))
    vertex_profiles_gc = regress_out_global(vertex_profiles, cortex_global)
    channel_profiles_gc = regress_out_global(channel_profiles, cortex_global)
    observed_gc, pvals_gc = circular_shift_pvalues(
        vertex_profiles_gc, channel_profiles_gc, args.n_shifts, args.random_state
    )
    qvals_gc = false_discovery_control(pvals_gc.ravel(), method="bh").reshape(pvals_gc.shape)

    tag = "_".join([
        _config_tag(vertex_row, len(vertex_ids), "v"),
        _config_tag(channel_row, len(channel_ids), "c"),
    ])
    out_dir = Path(args.output_dir) / args.family / "_channel_vertex_alignment" / tag
    if out_dir.exists() and not args.force and any(out_dir.iterdir()):
        log.info("Output already exists, use --force to overwrite: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sig = _write_alignment_outputs(out_dir, "", observed, pvals, qvals, vertex_ids, channel_ids,
                                     vertex_counts, channel_counts, tag, args.alpha)
    n_sig_gc = _write_alignment_outputs(out_dir, "_globalctrl", observed_gc, pvals_gc, qvals_gc,
                                        vertex_ids, channel_ids, vertex_counts, channel_counts,
                                        tag + " [global-signal controlled]", args.alpha)

    vertex_screen = profile_correlation_summary(vertex_profiles)
    channel_screen = profile_correlation_summary(channel_profiles)

    manifest = {
        "analysis": "channel_vertex_functional_alignment",
        "family": args.family,
        "vertex": _config_summary(vertex_row, vertex_ms_dir, vertex_labels_file,
                                  len(vertex_ids), vertex_noise),
        "channel": _config_summary(channel_row, channel_ms_dir, channel_labels_file,
                                   len(channel_ids), channel_noise),
        "vertex_temporal_differentiation": vertex_screen,
        "channel_temporal_differentiation": channel_screen,
        "n_bins": n_bins,
        "n_shifts": args.n_shifts,
        "random_state": args.random_state,
        "alpha": args.alpha,
        "n_pairs": int(observed.size),
        "n_pairs_fdr_sig": n_sig,
        "n_pairs_fdr_sig_globalctrl": n_sig_gc,
        "global_control": "cortex-wide mean bin time series (mean over all vertex/subcortical units) "
                          "regressed out of every vertex- and channel-cluster profile before re-testing",
        "null": "circular shift of channel-cluster profiles along the bin axis (nonzero offset)",
        "inputs": {
            "vertex_cifti": str(Path(args.vertex_cifti).resolve()),
            "vertex_run_trs": str(Path(args.vertex_run_trs).resolve()),
            "timing_csv": str(Path(args.timing_csv).resolve()),
            "channel_run_trs": str(Path(args.run_trs).resolve()),
            "embeddings_dir": str(Path(args.embeddings_dir).resolve()),
            "bin_sec": args.bin_sec, "skip_sec": args.skip_sec,
            "delay_sec": args.delay_sec, "tr": args.tr,
        },
    }
    (out_dir / "alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    log.info("Channel-vertex alignment complete: %s", out_dir)
    return out_dir


def demo() -> None:
    rng = np.random.default_rng(0)
    n_bins = 200
    shared = rng.normal(size=n_bins)
    unrelated = rng.normal(size=n_bins)

    vertex_units = np.stack(
        [shared + 0.1 * rng.normal(size=n_bins) for _ in range(3)]
        + [rng.normal(size=n_bins) for _ in range(2)]
    )
    vertex_labels = np.array([0, 0, 0, -1, -1])
    channel_units = np.stack(
        [shared + 0.1 * rng.normal(size=n_bins) for _ in range(4)]
        + [unrelated + 0.1 * rng.normal(size=n_bins) for _ in range(4)]
    )
    channel_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    v_profiles, v_ids, v_counts, v_noise = cluster_profiles(vertex_units, vertex_labels)
    c_profiles, c_ids, c_counts, c_noise = cluster_profiles(channel_units, channel_labels)
    assert v_profiles.shape == (1, n_bins) and v_ids == [0]
    assert v_noise == 2 and v_counts == {0: 3}
    assert c_profiles.shape == (2, n_bins) and c_ids == [0, 1]
    assert c_noise == 0

    r_direct = float(np.corrcoef(v_profiles[0], c_profiles[0])[0, 1])
    r_shortcut = float((v_profiles @ c_profiles.T)[0, 0] / n_bins)
    assert abs(r_direct - r_shortcut) < 1e-4, "dot-product shortcut must match np.corrcoef"

    observed, pvals = circular_shift_pvalues(v_profiles, c_profiles, n_shifts=1000, random_state=0)
    assert observed.shape == (1, 2)
    assert observed[0, 0] > 0.8, f"planted shared-signal pair should be strongly correlated: {observed[0, 0]}"
    assert pvals[0, 0] < 0.05, f"planted pair should survive the circular-shift null: p={pvals[0, 0]}"
    assert pvals[0, 1] > 0.05, f"unrelated pair should not be significant: p={pvals[0, 1]}"
    print(f"[demo] OK: observed={observed.ravel()} pvals={pvals.ravel()}")


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
