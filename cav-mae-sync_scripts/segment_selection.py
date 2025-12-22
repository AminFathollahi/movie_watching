"""
Post-process CAV-MAE Sync embeddings.

1) Load per-segment audio/video embeddings from embedding.py.
2) Build the cosine similarity matrix (audio rows, video columns) and save it.
3) Plot a heatmap and a diagonal vs off-diagonal summary plot.
4) Run one-sided t-tests per segment (diag > off-diagonal) and save significant indices.
5) Re-plot the diagonal/off-diagonal summary marking significant segments with stars.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
except Exception as e:  # pragma: no cover - environment-specific
    raise ImportError(
        "scipy is required for segment_selection.py (pip install scipy). "
        f"Import failed with: {e}"
    )


# ------------ Loading & similarity ------------

def load_embeddings(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio and video embeddings saved by embedding.py."""
    audio_path = emb_dir / "embeddings_audio.npy"
    video_path = emb_dir / "embeddings_video.npy"
    if not audio_path.exists() or not video_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings: expected {audio_path} and {video_path}. "
            "Run embedding.py first."
        )
    audio = np.load(audio_path)
    video = np.load(video_path)
    if audio.shape[0] != video.shape[0]:
        raise ValueError(
            f"Segment count mismatch: audio={audio.shape[0]} video={video.shape[0]}"
        )
    return audio, video


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a and b."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
    return a_norm @ b_norm.T


# ------------ Plots ------------

def plot_heatmap(sim: np.ndarray, out_path: Path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sim, origin="upper", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.xlabel("Video segment index")
    plt.ylabel("Audio segment index")
    plt.title("Audio vs. Video Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compute_diag_stats(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return diagonal, off-diagonal mean, std, and standard error per row."""
    diag = np.diag(sim)
    if sim.shape[0] > 1:
        n_off = sim.shape[1] - 1
        off_sum = sim.sum(axis=1) - diag
        off_mean = off_sum / n_off
        off_var = ((sim - diag[:, None]) ** 2).sum(axis=1) / n_off
        off_std = np.sqrt(off_var)
        off_se = off_std / np.sqrt(n_off)
    else:
        off_mean = np.zeros_like(diag)
        off_std = np.zeros_like(diag)
        off_se = np.zeros_like(diag)
    return diag, off_mean, off_std, off_se


def plot_diag_stats(diag: np.ndarray, off_mean: np.ndarray, off_std: np.ndarray, off_se: np.ndarray, out_path: Path):
    x = np.arange(diag.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(x, diag, label="diag (vid↔aud)", color="C0")
    plt.plot(x, off_mean, label="off-diag mean", color="C1")
    plt.fill_between(
        x,
        off_mean - off_se,
        off_mean + off_se,
        color="C1",
        alpha=0.3,
        label="off-diag ±1 se",
    )
    plt.xlabel("Segment index")
    plt.ylabel("Similarity")
    plt.title("Diagonal vs. Off-diagonal Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_diag_stats_flagged(
    diag: np.ndarray,
    off_mean: np.ndarray,
    off_std: np.ndarray,
    off_se: np.ndarray,
    sig_indices: List[int],
    out_path: Path,
):
    """Same as plot_diag_stats but marks significant segments with stars."""
    x = np.arange(diag.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(x, diag, label="diag (vid↔aud)", color="C0", marker="o", markersize=4, linewidth=1.5)
    plt.plot(x, off_mean, label="off-diag mean", color="C1")
    plt.fill_between(
        x,
        off_mean - off_se,
        off_mean + off_se,
        color="C1",
        alpha=0.3,
        label="off-diag ±1 se",
    )
    if sig_indices:
        plt.scatter(
            sig_indices,
            diag[sig_indices],
            marker="*",
            facecolor="black",
            edgecolor="k",
            s=100,
            zorder=3,
            label="significant (diag > off-diag, p<alpha)",
        )
    plt.xlabel("Segment index")
    plt.ylabel("Similarity")
    plt.title("Diagonal vs. Off-diagonal Similarity (significance flagged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------ Statistics ------------

def one_sided_ttest(off_vals: np.ndarray, diag_val: float) -> float:
    """
    One-sided t-test: H1 mean(off_vals) < diag_val.
    Returns p-value.
    """
    try:
        res = stats.ttest_1samp(off_vals, popmean=diag_val, alternative="less")
        return float(res.pvalue)
    except TypeError:
        res = stats.ttest_1samp(off_vals, popmean=diag_val)
        if np.isnan(res.statistic):
            return np.nan
        if res.statistic < 0:
            return float(res.pvalue / 2.0)
        return float(1.0 - res.pvalue / 2.0)


def find_significant(sim: np.ndarray, alpha: float) -> List[int]:
    """Return indices where diag > off-diagonal with one-sided t-test at alpha."""
    if sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Similarity matrix must be square; got {sim.shape}")
    n = sim.shape[0]
    if n < 2:
        return []

    diag = np.diag(sim)
    sig_indices: List[int] = []
    for i in range(n):
        off = np.concatenate([sim[i, :i], sim[i, i + 1 :]])
        if off.size == 0:
            continue
        p = one_sided_ttest(off, diag[i])
        if np.isnan(p):
            continue
        if p < alpha and diag[i] > off.mean():
            sig_indices.append(i)
    return sig_indices


# ------------ Main ------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot and test CAV-MAE audio/video similarities."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(Path(__file__).parent / "outputs_cavmae"),
        help="Directory containing embeddings_audio.npy and embeddings_video.npy.",
    )
    parser.add_argument(
        "--heatmap_path",
        type=str,
        default=None,
        help="Output path for the heatmap PNG (default: embeddings_dir/av_similarity_heatmap.png).",
    )
    parser.add_argument(
        "--diag_plot_path",
        type=str,
        default=None,
        help="Output path for the diagonal/off-diagonal plot PNG (default: embeddings_dir/diag_offdiag.png).",
    )
    parser.add_argument(
        "--diag_flagged_path",
        type=str,
        default=None,
        help="Output path for the significance-flagged plot PNG (default: embeddings_dir/diag_offdiag_flagged.png).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for one-sided t-test (diag > off-diagonal).",
    )
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir).expanduser().resolve()
    heatmap_path = (
        Path(args.heatmap_path).expanduser().resolve()
        if args.heatmap_path
        else emb_dir / "av_similarity_heatmap.png"
    )
    diag_plot_path = (
        Path(args.diag_plot_path).expanduser().resolve()
        if args.diag_plot_path
        else emb_dir / "diag_offdiag.png"
    )
    diag_flagged_path = (
        Path(args.diag_flagged_path).expanduser().resolve()
        if args.diag_flagged_path
        else emb_dir / "diag_offdiag_flagged.png"
    )

    audio, video = load_embeddings(emb_dir)
    sim = cosine_matrix(audio, video)

    sim_path = emb_dir / "av_similarity_matrix.npy"
    np.save(sim_path, sim.astype(np.float32))

    plot_heatmap(sim, heatmap_path)

    diag, off_mean, off_std, off_se = compute_diag_stats(sim)
    plot_diag_stats(diag, off_mean, off_std, off_se, diag_plot_path)

    sig_indices = find_significant(sim, args.alpha)
    plot_diag_stats_flagged(diag, off_mean, off_std, off_se, sig_indices, diag_flagged_path)

    out_npy = emb_dir / "significant_segments.npy"
    out_txt = emb_dir / "significant_segments.txt"
    np.save(out_npy, np.array(sig_indices, dtype=np.int64))
    with open(out_txt, "w") as f:
        f.write("# Segment indices (0-based) with diag > off-diag, p < {:.3f}\n".format(args.alpha))
        f.write(",".join(str(idx) for idx in sig_indices))

    print(f"Saved similarity matrix to {sim_path} with shape {sim.shape}")
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved diagonal/off-diagonal plot to {diag_plot_path}")
    print(f"Saved significance-flagged plot to {diag_flagged_path}")
    print(f"Found {len(sig_indices)} significant segments (alpha={args.alpha}); saved to {out_npy} and {out_txt}")


if __name__ == "__main__":
    main()
