import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.channel_modality_preference import channel_stats, cluster_preference_permutation


def _synthetic_embeddings(rng: np.random.Generator, n_bins: int = 200, n_channels: int = 40,
                          n_audio: int = 15) -> dict[str, np.ndarray]:
    a = rng.normal(size=(n_bins, n_channels))
    v = rng.normal(size=(n_bins, n_channels))
    av = rng.normal(size=(n_bins, n_channels)) * 0.05
    av[:, :n_audio] += a[:, :n_audio]
    av[:, n_audio:] += v[:, n_audio:]
    return {"av": av, "a": a, "v": v, "cls_a": a.copy(), "cls_v": v.copy()}


def test_channel_stats_recovers_known_audio_driven_channels():
    rng = np.random.default_rng(0)
    n_audio = 15
    embeddings = _synthetic_embeddings(rng, n_audio=n_audio)

    df = channel_stats(embeddings)

    assert (df["d_c"].to_numpy()[:n_audio] > 0).all()
    assert (df["d_c"].to_numpy()[n_audio:] < 0).all()


def test_cluster_permutation_detects_audio_driven_cluster():
    rng = np.random.default_rng(1)
    n_audio, n_channels = 15, 40
    embeddings = _synthetic_embeddings(rng, n_audio=n_audio, n_channels=n_channels)
    df = channel_stats(embeddings)
    d = df["d_c"].to_numpy()

    labels = np.zeros(n_channels, dtype=int)
    labels[n_audio:] = 1
    in_mask = labels == 0

    observed, p = cluster_preference_permutation(d, in_mask, n_permutations=2000,
                                                 rng=np.random.default_rng(2))

    assert observed > 0.5
    assert p < 0.01
