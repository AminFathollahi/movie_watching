"""
rsa/shared/model_registry.py
==============================
Central registry for model embeddings and partial RSA configurations.

All RSA scripts (run_partial_rsa.py, run_rdm_diagonal.py,
run_multimodal_decomposition.py) import from here so that model names,
modality codes, and partial-RSA run definitions are defined exactly once.

Embedding path convention (mirrors run_analysis.sh):
  {embeddings_dir}/{model}/{bin_sec_int}s/{model}_{modality}.npy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

# ── Default parameters ────────────────────────────────────────────────────────
BIN_SEC_DEFAULT: float = 5.0
DELAY_SEC_DEFAULT: float = 5.0
TR_DEFAULT: float = 1.0


# ── Path resolver ─────────────────────────────────────────────────────────────

def emb_path(embeddings_dir: str, model: str, modality: str,
             bin_sec: float) -> Path:
    """Resolve the .npy embedding file path for a given model/modality/bin.

    Parameters
    ----------
    embeddings_dir : root directory (e.g. outputs/model_embeddings)
    model          : model name string (e.g. "pe-av-small-16-frame")
    modality       : modality code "a", "v", or "av"
    bin_sec        : temporal bin width in seconds

    Returns
    -------
    Path — absolute or relative .npy path
    """
    bin_sec_int = int(bin_sec)
    return (Path(embeddings_dir) / model / f"{bin_sec_int}s"
            / f"{model}_{modality}.npy")


# ── Model catalogue ───────────────────────────────────────────────────────────
# Each entry documents the available modalities and a brief description.
# The "joint" key indicates whether the model produces a genuine multimodal
# joint embedding (True) vs. a unimodal model (False).

MODELS: dict[str, dict] = {
    "pe-av-small-16-frame": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Perception Encoder AV (small, 16-frame window) — joint audio-visual",
    },
    "pe-av-base-16-frame": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Perception Encoder AV (base, 16-frame window) — joint audio-visual",
    },
    "pe-av-large-16-frame": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Perception Encoder AV (large, 16-frame window) — joint audio-visual",
    },
    "cav-mae-sync": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "CAV-MAE (sync variant) — contrastive AV masked autoencoder",
    },
    "audiomae": {
        "modalities":   ["a"],
        "joint":        False,
        "description":  "AudioMAE — audio-only masked autoencoder",
    },
    "videomaev2-large": {
        "modalities":   ["v"],
        "joint":        False,
        "description":  "VideoMAEv2 Large — video-only masked autoencoder",
    },
    "wavlm-large": {
        "modalities":   ["a"],
        "joint":        False,
        "description":  "WavLM Large — audio self-supervised speech model",
    },
    "whisper-large-v3": {
        "modalities":   ["a"],
        "joint":        False,
        "description":  "Whisper Large v3 — audio/ASR encoder",
    },
    "pe-core-l14": {
        "modalities":   ["v"],
        "joint":        False,
        "description":  "Perception Encoder Core ViT-L/14 — vision-only encoder",
    },
}

# ── Recommended future models ─────────────────────────────────────────────────
RECOMMENDED_FUTURE_MODELS = """
Models suggested for download to strengthen the multimodal interaction analysis:

  pe-av-large-16-frame   — Larger PE-AV (same architecture, more parameters).
                           Tests whether the interaction signal scales with capacity.
                           Safetensors: facebook/perception-encoder (large variant).

  cav-mae-sync           — Already in registry; embeddings needed for additional
                           bin_sec values. Contrastive architecture vs. PE-AV's
                           generative one — tests architecture independence.

  imagebind              — Meta's model that aligns audio, video, and text in a
                           single semantic space.  If brain 'interaction regions'
                           align with imagebind but NOT with cav-mae, this implies
                           the interaction is language/semantic rather than
                           perceptual.  Safetensors: facebookresearch/ImageBind.
"""


# ── Partial RSA run configurations ────────────────────────────────────────────
# Each entry specifies a target model/modality and one or more nuisance
# model/modality pairs to regress out via banded ridge.

class PartialRSARun(NamedTuple):
    target:   tuple[str, str]          # (model, modality) for the target embedding
    nuisance: list[tuple[str, str]]    # [(model, modality), ...] nuisance embeddings
    label:    str                      # short string for file naming (no spaces)
    description: str                   # human-readable description


PARTIAL_RSA_RUNS: dict[str, PartialRSARun] = {
    # Run A: regress out independently-trained unimodal baselines (AudioMAE + VideoMAE)
    # from PE-AV joint.  Tests whether PE-AV encodes something beyond what two
    # separate specialist models capture.
    "run_A": PartialRSARun(
        target   = ("pe-av-small-16-frame", "av"),
        nuisance = [("audiomae", "a"), ("videomaev2-large", "v")],
        label    = "peav_av_partialout_audiomae_a+videomaev2_v",
        description = (
            "PE-AV (small 16-frame) AV joint embedding, controlling for "
            "AudioMAE (audio) + VideoMAEv2-Large (video) — tests cross-architecture "
            "unique variance."
        ),
    ),
    # Run B: regress out PE-AV's own unimodal decoders.  Tests whether the joint
    # embedding encodes cross-modal interactions beyond the simple union of its own
    # audio-only and video-only outputs.
    "run_B": PartialRSARun(
        target   = ("pe-av-small-16-frame", "av"),
        nuisance = [("pe-av-small-16-frame", "a"), ("pe-av-small-16-frame", "v")],
        label    = "peav_av_partialout_peav_a+peav_v",
        description = (
            "PE-AV (small 16-frame) AV joint embedding, controlling for its own "
            "audio-only and video-only outputs — tests within-architecture cross-modal "
            "interaction residual."
        ),
    ),
}


# ── Diagonal-masking configurations ──────────────────────────────────────────
# Default model/modality used by run_rdm_diagonal.py.

DIAGONAL_MASK_DEFAULT = {
    "model":    "pe-av-small-16-frame",
    "modality": "av",
    "bin_sec":  BIN_SEC_DEFAULT,
}


# ── Convenience helpers ────────────────────────────────────────────────────────

def validate_model(model: str, modality: str) -> None:
    """Raise ValueError if model or modality is not registered."""
    if model not in MODELS:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Registered models: {list(MODELS.keys())}"
        )
    if modality not in MODELS[model]["modalities"]:
        raise ValueError(
            f"Modality '{modality}' not available for model '{model}'. "
            f"Available: {MODELS[model]['modalities']}"
        )


def validate_run(run_name: str) -> PartialRSARun:
    """Return the PartialRSARun config, raising ValueError if unknown."""
    if run_name not in PARTIAL_RSA_RUNS:
        raise ValueError(
            f"Unknown partial RSA run '{run_name}'. "
            f"Available: {list(PARTIAL_RSA_RUNS.keys())}"
        )
    return PARTIAL_RSA_RUNS[run_name]


def check_embeddings_exist(embeddings_dir: str, model: str, modality: str,
                            bin_sec: float) -> Path:
    """Resolve embedding path and raise FileNotFoundError if missing."""
    path = emb_path(embeddings_dir, model, modality, bin_sec)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding not found: {path}\n"
            f"Expected: {embeddings_dir}/{model}/{int(bin_sec)}s/{model}_{modality}.npy"
        )
    return path
