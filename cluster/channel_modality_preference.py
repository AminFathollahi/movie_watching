"""Audio/video preference of channel clusters, per clustering solution.

For each channel cluster of a screened clustering solution, tests whether its
channels' AV representation tracks the audio-only or video-only embedding more
closely (``d_c = atanh(r_av_a) - atanh(r_av_v)``, positive = audio-driven), and
the same axis restricted to single-modality-input AV passes (``clsav_from_a`` /
``clsav_from_v``). Reuses ``_profile_correlation``/``_processed_embedding``/
``_model_files`` from ``cf_modeling/channel_cca_analysis.py`` (same math as the
channel<->CCA-axis analysis there) and cluster labels loaded exactly as
``screen_temporal_differentiation.py`` loads them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from channel_vertex_alignment import parse_args as loader_args, labels_path  # noqa: E402
from cf_modeling.channel_cca_analysis import (  # noqa: E402
    MODEL_CONFIGS, _model_files, _processed_embedding, _profile_correlation,
)
from rsa.shared.rsa_utils import get_run_bin_counts  # noqa: E402

OUTPUT_DIR = Path("/home/amin/Research/Representation/Movie/outputs/cluster")
SCREEN_CSV = OUTPUT_DIR / "_channel_vertex_alignment_screen.csv"
FAMILIES = ("peav", "nemotron_layer18_mp")
R_CLIP = 1 - 1e-6


def _direct_path(root: Path, model: str, modality: str) -> Path:
    return root / model / "bin5s_skip5s" / f"{model}_{modality}.npy"


def load_family_embeddings(family: str) -> dict[str, np.ndarray] | None:
    """(n_bins, n_channels) av/a/v + clsav-ablation embeddings, binned and per-run
    z-scored the same way ``channel_timeseries_clustering.py`` binned the AV
    channels that produced the cluster labels. None if any file is missing."""
    args = loader_args(["--family", family])
    model = MODEL_CONFIGS[family]
    root = Path(args.embeddings_dir)
    intact_path, clsa_path, clsv_path = _model_files(root, model)
    paths = {
        "av": intact_path, "a": _direct_path(root, model, "a"), "v": _direct_path(root, model, "v"),
        "cls_a": clsa_path, "cls_v": clsv_path,
    }
    missing = {k: str(p) for k, p in paths.items() if not p.exists()}
    if missing:
        print(f"[{family}] missing embeddings, skipping family: {missing}")
        return None

    timing = pd.read_csv(args.timing_csv)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    run_bins = get_run_bin_counts(timing, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    embeddings = {k: _processed_embedding(p, args, run_bins) for k, p in paths.items()}
    n_channels = {k: v.shape[1] for k, v in embeddings.items()}
    if len(set(n_channels.values())) != 1:
        raise ValueError(f"[{family}] channel count mismatch across modalities: {n_channels}")
    return embeddings


def channel_stats(embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-channel r_av_a/r_av_v/d_c and their clsav analogues."""
    av, a, v = embeddings["av"], embeddings["a"], embeddings["v"]
    r_av_a = _profile_correlation(av, a)
    r_av_v = _profile_correlation(av, v)
    r_cls_a = _profile_correlation(av, embeddings["cls_a"])
    r_cls_v = _profile_correlation(av, embeddings["cls_v"])
    atanh = lambda r: np.arctanh(np.clip(r, -R_CLIP, R_CLIP))  # noqa: E731
    return pd.DataFrame({
        "channel_index": np.arange(av.shape[1]),
        "r_av_a": r_av_a, "r_av_v": r_av_v, "d_c": atanh(r_av_a) - atanh(r_av_v),
        "r_cls_a": r_cls_a, "r_cls_v": r_cls_v, "d_cls_c": atanh(r_cls_a) - atanh(r_cls_v),
    })


def cluster_preference_permutation(d: np.ndarray, in_mask: np.ndarray, n_permutations: int,
                                   rng: np.random.Generator) -> tuple[float, float]:
    """Two-sided permutation test: mean(d[in cluster]) - mean(d[outside cluster])
    against a null built by shuffling channel labels across the whole channel pool
    (noise channels included in "outside", excluded from "in")."""
    n_in, c = int(in_mask.sum()), d.size
    observed = float(d[in_mask].mean() - d[~in_mask].mean())
    perm_idx = np.argsort(rng.random((n_permutations, c)), axis=1)[:, :n_in]
    in_sums = d[perm_idx].sum(axis=1)
    total = d.sum()
    null = in_sums / n_in - (total - in_sums) / (c - n_in)
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    return observed, p


def bootstrap_ci_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size < 2:
        return float(values[0]), float(values[0])
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def evaluate_solution(channel_df: pd.DataFrame, labels: np.ndarray, n_permutations: int,
                      random_state: int, n_bootstrap: int) -> pd.DataFrame:
    d, d_cls = channel_df["d_c"].to_numpy(), channel_df["d_cls_c"].to_numpy()
    cluster_ids = sorted(int(c) for c in np.unique(labels) if c != -1)
    perm_rng = np.random.default_rng(random_state)
    boot_rng = np.random.default_rng(random_state + 1)
    rows = []
    for cid in cluster_ids:
        in_mask = labels == cid
        observed, p = cluster_preference_permutation(d, in_mask, n_permutations, perm_rng)
        ci_low, ci_high = bootstrap_ci_mean(d[in_mask], n_bootstrap, boot_rng)
        n_pos, n_neg = int((d[in_mask] > 0).sum()), int((d[in_mask] < 0).sum())
        sign_p = binomtest(n_pos, n_pos + n_neg, p=0.5).pvalue if (n_pos + n_neg) > 0 else np.nan
        sub = channel_df.loc[in_mask]
        rows.append({
            "cluster_id": cid, "n_channels": int(in_mask.sum()),
            "r_av_a_mean": float(sub["r_av_a"].mean()), "r_av_a_sd": float(sub["r_av_a"].std()),
            "r_av_v_mean": float(sub["r_av_v"].mean()), "r_av_v_sd": float(sub["r_av_v"].std()),
            "d_c_mean": float(d[in_mask].mean()), "d_c_sd": float(d[in_mask].std()),
            "d_c_ci_low": ci_low, "d_c_ci_high": ci_high,
            "r_cls_a_mean": float(sub["r_cls_a"].mean()), "r_cls_a_sd": float(sub["r_cls_a"].std()),
            "r_cls_v_mean": float(sub["r_cls_v"].mean()), "r_cls_v_sd": float(sub["r_cls_v"].std()),
            "d_cls_c_mean": float(d_cls[in_mask].mean()), "d_cls_c_sd": float(d_cls[in_mask].std()),
            "permutation_statistic": observed, "permutation_p": p,
            "sign_test_p": float(sign_p) if not np.isnan(sign_p) else np.nan,
            "sign_test_n_positive": n_pos, "sign_test_n_negative": n_neg,
        })
    table = pd.DataFrame(rows)
    table["permutation_q_bh"] = false_discovery_control(table["permutation_p"], method="bh")
    return table


def plot_cluster_preference(table: pd.DataFrame, title: str, out_path: Path, alpha: float) -> None:
    import matplotlib.pyplot as plt

    ordered = table.sort_values("d_c_mean").reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = ["#c0392b" if q < alpha else "#7f8c8d" for q in ordered["permutation_q_bh"]]
    figure, axis = plt.subplots(figsize=(6.5, max(3, 0.4 * len(ordered) + 1.5)), dpi=160)
    axis.errorbar(
        ordered["d_c_mean"], y,
        xerr=[ordered["d_c_mean"] - ordered["d_c_ci_low"], ordered["d_c_ci_high"] - ordered["d_c_mean"]],
        fmt="none", ecolor="#999999", capsize=2, zorder=2,
    )
    axis.scatter(ordered["d_c_mean"], y, c=colors, s=45, zorder=3)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_yticks(y)
    axis.set_yticklabels([f"cluster {c} (n={n})" for c, n in zip(ordered["cluster_id"], ordered["n_channels"])])
    axis.set_xlabel("mean d = atanh(r_av_a) - atanh(r_av_v)  (+ audio-driven, - video-driven)")
    axis.set_title(f"{title}\nred = BH-FDR q<{alpha} (channel bootstrap CI shown)", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def select_top_n_solutions(screen: pd.DataFrame, family: str, top_n: int) -> pd.DataFrame:
    subset = screen[(screen["side"] == "channel") & (screen["family"] == family) & (screen["n_clusters"] >= 3)]
    return subset.sort_values("mean_abs_off_diag").head(top_n)


def parse_cli(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--family", dest="families", action="append", choices=FAMILIES, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "channel_modality_preference"))
    args = parser.parse_args(argv)
    if not args.families:
        args.families = list(FAMILIES)
    return args


def main(argv=None) -> None:
    args = parse_cli(argv)
    out_root = Path(args.output_dir)
    screen = pd.read_csv(SCREEN_CSV)
    summary_rows = []

    for family in args.families:
        embeddings = load_family_embeddings(family)
        if embeddings is None:
            continue
        channel_df = channel_stats(embeddings)
        ms_dir = OUTPUT_DIR / family / "_channel" / "norm-zscore_raw"
        solutions = select_top_n_solutions(screen, family, args.top_n)

        for _, sol in solutions.iterrows():
            reducer_tag, cluster_tag = str(sol["reducer_tag"]), str(sol["cluster_tag"])
            lf = labels_path(ms_dir, pd.Series({"reducer_tag": reducer_tag, "full_fit_cluster_tag": cluster_tag}),
                             "channel_labels.npy")
            labels = np.load(lf)
            if labels.shape[0] != len(channel_df):
                print(f"[{family}] channel count mismatch for {reducer_tag}_{cluster_tag}: "
                     f"labels={labels.shape[0]} channels={len(channel_df)}, skipping")
                continue

            table = evaluate_solution(channel_df, labels, args.n_permutations, args.random_state, args.n_bootstrap)
            out_dir = out_root / family / f"{reducer_tag}_{cluster_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_dir / "cluster_preference.csv", index=False)

            channel_out = channel_df.copy()
            channel_out["cluster_label"] = labels
            channel_out.to_csv(out_dir / "channel_preference.csv", index=False)

            title = f"{family} | {reducer_tag} | {cluster_tag} | n_clusters={int(sol['n_clusters'])}"
            plot_cluster_preference(table, title, out_dir / "preference.png", args.alpha)

            table_summary = table.copy()
            table_summary["family"], table_summary["reducer_tag"], table_summary["cluster_tag"] = (
                family, reducer_tag, cluster_tag
            )
            table_summary["abs_d_c_mean"] = table_summary["d_c_mean"].abs()
            summary_rows.append(table_summary)

            n_sig = int((table["permutation_q_bh"] < args.alpha).sum())
            print(f"[{family}] {reducer_tag}_{cluster_tag}: {len(table)} clusters, "
                 f"{n_sig} significant @ q<{args.alpha} -> {out_dir}")

    if not summary_rows:
        print("No solutions evaluated.")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_rows, ignore_index=True).sort_values("abs_d_c_mean", ascending=False)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
