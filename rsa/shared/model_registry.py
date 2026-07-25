"""
rsa/shared/model_registry.py
==============================
Central registry for model embeddings and partial RSA configurations.

All RSA scripts (partial_rsa.py, rdm_diagonal.py,
multimodal_decomposition.py) import from here so that model names,
modality codes, and partial-RSA run definitions are defined exactly once.

Embedding path convention (mirrors analysis.sh's repo-wide bin/skip naming):
  {embeddings_dir}/{model}/bin{bin_sec_int}s_skip{skip_sec_int}s/{model}_{modality}.npy
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
             bin_sec: float, skip_sec: float | None = None) -> Path:
    """Resolve the .npy embedding file path for a given model/modality/bin.

    Parameters
    ----------
    embeddings_dir : root directory (e.g. outputs/model_embeddings)
    model          : model name string (e.g. "pe-av-small-16-frame")
    modality       : modality code "a", "v", or "av"
    bin_sec        : temporal bin width in seconds
    skip_sec       : sliding-window stride in seconds (defaults to bin_sec)

    Returns
    -------
    Path — absolute or relative .npy path
    """
    bin_sec_int  = int(bin_sec)
    skip_sec_int = int(skip_sec) if skip_sec is not None else bin_sec_int
    return (Path(embeddings_dir) / model / f"bin{bin_sec_int}s_skip{skip_sec_int}s"
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
    "imagebind": {
        "modalities":   ["av"],
        "joint":        True,
        "description":  "ImageBind — aligned AV/text embedding space (2s bins only, no separable a/v)",
    },
    "omni3b_layer9_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Qwen2.5-Omni-3B thinker hidden state, layer 9 — early AV fusion",
    },
    "omni3b_layer18_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Qwen2.5-Omni-3B thinker hidden state, layer 18 — mid AV fusion",
    },
    "omni3b_layer27_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Qwen2.5-Omni-3B thinker hidden state, layer 27 — late AV fusion",
    },
    "topoomni_layer9_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni (cortical-sheet-regularized Qwen2.5-Omni-3B), layer 9",
    },
    "topoomni_layer18_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni (cortical-sheet-regularized Qwen2.5-Omni-3B), layer 18",
    },
    "topoomni_layer27_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni (cortical-sheet-regularized Qwen2.5-Omni-3B), layer 27",
    },
    "topoomni_layer9_sheet_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni cortical-sheet (topographic) code, layer 9",
    },
    "topoomni_layer18_sheet_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni cortical-sheet (topographic) code, layer 18",
    },
    "topoomni_layer27_sheet_mp": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "Topo-Omni cortical-sheet (topographic) code, layer 27",
    },
    # ── "_lasttoken" av variants: the last-sequence-position hidden state of the
    # SAME joint (audio+video) forward pass used for the plain "_av" above, rather
    # than a masked-token-position average. Unlike (a+v)/2 pooling, this is not a
    # deterministic function of "_a"/"_v" -- a genuinely emergent joint summary
    # (closer to how the Topo-Omni paper itself reads out per-clip stimulus
    # embeddings: "last-layer activation of the final token", Sec 4.7.2). av-only;
    # nuisance for the Move-1 integration contrast is the corresponding non-lasttoken
    # model's real (audio-only-pass / video-only-pass) _a/_v.
    "omni3b_layer9_lt":  {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 9, last-token joint-AV readout"},
    "omni3b_layer18_lt": {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 18, last-token joint-AV readout"},
    "omni3b_layer27_lt": {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 27, last-token joint-AV readout"},
    "topoomni_layer9_lt":  {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 9, last-token joint-AV readout"},
    "topoomni_layer18_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 18, last-token joint-AV readout"},
    "topoomni_layer27_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 27, last-token joint-AV readout"},
    "topoomni_layer9_sheet_lt":  {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni cortical sheet, layer 9, last-token joint-AV readout"},
    "topoomni_layer18_sheet_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni cortical sheet, layer 18, last-token joint-AV readout"},
    "topoomni_layer27_sheet_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni cortical sheet, layer 27, last-token joint-AV readout"},
    # ── nvidia/omni-embed-nemotron-3b ("NV-QwenOmni-Embed-3B-v1") -- a third
    # member of the Qwen2.5-Omni-3B-Thinker lineage (same base as omni3b/
    # topoomni above), but purpose-built as a contrastively-trained retrieval
    # EMBEDDING model: Talker removed, self-attention switched from causal to
    # BIDIRECTIONAL, and audio/video encoded as separate (non-interleaved)
    # streams even when jointly present. Unlike omni3b/topoomni, its official
    # "_av" readout (mean-pool over the full joint-forward-pass sequence,
    # L2-normalized) is a genuine joint summary from the start -- no (a+v)/2
    # circularity bug to fix (see nemotron_extract_intact.py). Layer 36 is
    # this model's TRUE FINAL layer / native trained embedding output;
    # layers 9/18/27 are supplementary depth-sweep probes for comparability
    # with omni3b/topoomni, not part of this checkpoint's training objective.
    "nemotron_layer9_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 9 — early AV fusion (depth-sweep probe)"},
    "nemotron_layer18_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 18 — mid AV fusion (depth-sweep probe)"},
    "nemotron_layer27_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 27 — late AV fusion (depth-sweep probe)"},
    "nemotron_layer36_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 36 (true final layer) — native contrastively-trained AV retrieval embedding"},
    # ── Penultimate-layer additions (thinker layer 35 of 36, second-to-last).
    # omni3b/topoomni use their own 0-indexed decoder_layers/cortical_adaptors
    # convention (layer{i}=index i directly), so penultimate is index 34,
    # named "layer34" -- NOT "layer35", which already denotes the pre-existing
    # (stale, un-migrated) FINAL layer for these two families. nemotron uses a
    # 1-indexed hidden_states convention where "layer35" is directly correct.
    # All three names refer to the SAME conceptual position (second-to-last of
    # 36 thinker layers). See *_extract_thinker_penultimate.py docstrings.
    "omni3b_layer34_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker hidden state, layer 34 (penultimate of 36) — late AV fusion"},
    "omni3b_layer34_lt": {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 34 (penultimate of 36), last-token joint-AV readout"},
    "topoomni_layer34_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 34 (penultimate of 36)"},
    "topoomni_layer34_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 34 (penultimate of 36), last-token joint-AV readout"},
    "topoomni_layer34_sheet_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni cortical-sheet (topographic) code, layer 34 (penultimate of 36)"},
    "topoomni_layer34_sheet_lt": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni cortical sheet, layer 34 (penultimate of 36), last-token joint-AV readout"},
    "nemotron_layer35_mp": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 35 (penultimate of 36) — late AV fusion (depth-sweep probe)"},
    "nemotron_layer35_lt": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 35 (penultimate of 36), last-token readout"},
    # ── Own-encoder (audio_tower/visual, pre-thinker-fusion) penultimate-layer
    # probes: genuinely separate-pass audio/video ENCODER representations, not
    # thinker hidden states -- see *_extract_encoder_penultimate.py. No "av"
    # readout (the two towers never see each other's tokens), so joint=False
    # like audiomae/videomaev2-large.
    "omni3b_encoder_penultimate": {"modalities": ["a", "v"], "joint": False,
        "description": "Qwen2.5-Omni-3B audio_tower/visual encoder, penultimate layer (pre-thinker-fusion, own-encoder space)"},
    "topoomni_encoder_penultimate": {"modalities": ["a", "v"], "joint": False,
        "description": "Topo-Omni audio_tower/visual encoder, penultimate layer (pre-thinker-fusion, own-encoder space)"},
    "nemotron_encoder_penultimate": {"modalities": ["a", "v"], "joint": False,
        "description": "Omni-Embed-Nemotron-3B audio_tower/visual encoder, penultimate layer (pre-thinker-fusion, own-encoder space)"},
    # ── Temporal-scramble binding control. Each bin's video paired with
    # a randomly permuted bin's audio (fixed seed) before extraction -- breaks
    # correct A-V temporal binding while preserving each modality's marginal
    # content. Same modalities as the intact model.
    "pe-av-small-16-frame_avscramble": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "PE-AV (small 16-frame), temporal-scramble binding control",
    },
    "cav-mae-sync_avscramble": {
        "modalities":   ["av", "a", "v"],
        "joint":        True,
        "description":  "CAV-MAE (sync variant), temporal-scramble binding control",
    },
    # omni3b/topoomni scramble: "_a"/"_v" here are REINDEXED copies of the real
    # unimodal embeddings (a_scrambled[i] = a_true[perm[i]], v unchanged) built by
    # build_scramble_unimodal_copies.py -- pure numpy, no re-inference needed,
    # since the true unimodal passes have no cross-modal tokens and cannot depend
    # on pairing. Only "_av" required a real (scrambled-pairing) forward pass.
    "omni3b_layer9_mp_avscramble":  {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 9, temporal-scramble binding control"},
    "omni3b_layer18_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 18, temporal-scramble binding control"},
    "omni3b_layer27_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 27, temporal-scramble binding control"},
    "topoomni_layer9_mp_avscramble":  {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 9, temporal-scramble binding control"},
    "topoomni_layer18_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 18, temporal-scramble binding control"},
    "topoomni_layer27_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 27, temporal-scramble binding control"},
    "nemotron_layer9_mp_avscramble":  {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 9, temporal-scramble binding control"},
    "nemotron_layer18_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 18, temporal-scramble binding control"},
    "nemotron_layer27_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 27, temporal-scramble binding control"},
    "nemotron_layer36_mp_avscramble": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 36 (native embedding), temporal-scramble binding control"},
    # "_lasttoken_avscramble": av-only (last-token joint readout under scrambled
    # pairing); nuisance for its integration run reuses the corresponding
    # "_avscramble" model's reindexed a/v (see _lasttoken_integration_run below).
    "omni3b_layer9_lt_avscramble":  {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 9, last-token readout, temporal-scramble control"},
    "omni3b_layer18_lt_avscramble": {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 18, last-token readout, temporal-scramble control"},
    "omni3b_layer27_lt_avscramble": {"modalities": ["av"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker, layer 27, last-token readout, temporal-scramble control"},
    "topoomni_layer9_lt_avscramble":  {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 9, last-token readout, temporal-scramble control"},
    "topoomni_layer18_lt_avscramble": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 18, last-token readout, temporal-scramble control"},
    "topoomni_layer27_lt_avscramble": {"modalities": ["av"], "joint": True,
        "description": "Topo-Omni thinker hidden state, layer 27, last-token readout, temporal-scramble control"},
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

# ── Native AV joint models ────────────────────────────────────────────────────
# Every genuine joint-fusion AV embedding we have, excluding the Move-3
# controls (_avscramble, _clsav_from_a/_v) which are binding/presence
# manipulations of a base model, not additional "real" AV models. Used to
# generalize the residual/partial-correlation analyses (linear_resid,
# partial_corr, projection_resid) across all native AV models instead of
# just pe-av-small-16-frame.

def _is_native_av(model: str) -> bool:
    info = MODELS[model]
    return (
        info["joint"]
        and "av" in info["modalities"]
        and "_avscramble" not in model
        and "_clsav_from_" not in model
    )


NATIVE_AV_MODELS: list[str] = [m for m in MODELS if _is_native_av(m)]

# ── Auto-derived residual pseudo-models ───────────────────────────────────────
# _linear_resid pseudo-models are generated on disk by
# notebooks/feature_extraction/compute_linear_residual_embeddings.py for every
# NATIVE_AV_MODELS entry (ridge residual after regressing out AudioMAE(a) +
# VideoMAEv2-Large(v)). Registered here purely for documentation/discoverability
# -- rsa/searchlight.py, rsa/partial_rsa.py, and encoding/encoding.py only need
# the .npy file to exist at the standard emb_path() location, not a MODELS entry.
for _m in list(NATIVE_AV_MODELS):
    MODELS[f"{_m}_av_linear_resid"] = {
        "modalities": ["av"], "joint": True,
        "description": (
            f"{_m} AV joint embedding, ridge residual after regressing out "
            f"AudioMAE(a) + VideoMAEv2-Large(v) -- unique variance beyond two "
            f"independent unimodal specialists (linear_resid)."
        ),
    }
del _m


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
    kind:     str = "cross_baseline"   # "cross_baseline" | "integration"
    # "integration" runs regress a native-AV model's OWN unimodal streams out of
    # its OWN joint embedding (the best-additive-combination contrast).
    # "cross_baseline" runs regress OTHER models' unimodal streams out instead.
    # Both kinds share one output convention (see partial_rsa.py's save block):
    # group_average/<label>/k{K}_delay{D}s_bin{B}s_skip{S}s_{method}/<file>.


def _own_unimodal_integration_run(model: str, modality: str = "av") -> PartialRSARun:
    """Best-additive integration contrast: model's own joint embedding, controlling
    for its own unimodal (a, v) streams -- the best-additive integration contrast.
    """
    return PartialRSARun(
        target   = (model, modality),
        nuisance = [(model, "a"), (model, "v")],
        label    = f"{model}_{modality}_INTEGRATION",
        description = (
            f"{model} {modality} joint embedding, controlling for its own "
            f"audio-only and video-only outputs — best-additive-combination "
            f"integration contrast (unique variance from fusion, holding "
            f"information content fixed)."
        ),
        kind = "integration",
    )


def _lasttoken_integration_run(base_model: str, modality: str = "av") -> PartialRSARun:
    """Like _own_unimodal_integration_run(), but the TARGET is the "_lt" (last-token)
    joint-AV readout (last sequence position of the joint forward pass -- a
    genuinely emergent summary, not a fixed function of the pooled a/v streams)
    while the NUISANCE bands are the corresponding "_mp" (mean-pool) model's real
    (separate audio-only-pass / video-only-pass) unimodal streams. `base_model` is
    the BARE identifier with neither suffix (e.g. "omni3b_layer9"). Only defined
    for models whose "_a"/"_v" come from genuinely separate unimodal forward
    passes (omni3b/topoomni layers, post the Move-1/4 unimodal re-extraction fix).
    """
    mp_model = f"{base_model}_mp"
    lt_model = f"{base_model}_lt"
    return PartialRSARun(
        target   = (lt_model, modality),
        nuisance = [(mp_model, "a"), (mp_model, "v")],
        label    = f"{lt_model}_{modality}_INTEGRATION",
        description = (
            f"{base_model} last-token joint-AV readout, controlling for the base "
            f"model's own (genuinely separate-pass) audio-only and video-only "
            f"outputs — integration contrast using a non-tautological joint "
            f"representation (see omni3b_extract_intact.py's docstring)."
        ),
        kind = "integration",
    )


def _lasttoken_scramble_integration_run(base_model: str, modality: str = "av") -> PartialRSARun:
    """Scrambled-pairing counterpart of _lasttoken_integration_run(): TARGET is the
    "_lt_avscramble" joint-AV readout (last-token of a forward pass fed
    video[i] + audio[perm[i]]); NUISANCE is the corresponding "_mp_avscramble" model's
    reindexed a/v (a[perm[i]], v[i] -- i.e. what was actually fed to the model),
    NOT the intact base model's true-paired a/v. `base_model` is the BARE identifier
    with neither suffix (e.g. "omni3b_layer9"). Mirrors the reasoning already used
    for the plain avscramble integration runs (_own_unimodal_integration_run on
    "{model}_mp_avscramble") -- nuisance always represents "what content was actually
    given to the model", so the residual isolates emergent fusion, not a mismatch
    artifact from comparing against the wrong nuisance pairing.
    """
    scramble_model = f"{base_model}_mp_avscramble"
    lt_model = f"{base_model}_lt_avscramble"
    return PartialRSARun(
        target   = (lt_model, modality),
        nuisance = [(scramble_model, "a"), (scramble_model, "v")],
        label    = f"{lt_model}_{modality}_INTEGRATION",
        description = (
            f"{base_model} last-token joint-AV readout under scrambled (mismatched) "
            f"audio-video pairing, controlling for the audio/video actually fed to "
            f"the model at each row (reindexed a[perm[i]], true v[i]) -- temporal-scramble "
            f"binding control counterpart of the intact lasttoken integration run."
        ),
        kind = "integration",
    )


def _dummy_integration_run(model: str, dummy_modality: str) -> PartialRSARun:
    """Dummy-modality counterpart of _own_unimodal_integration_run(): TARGET is the
    "_clsav_from_{a,v}" joint-AV readout (real ONE modality + a fixed content-free
    placeholder for the other); NUISANCE is only the corresponding REAL modality's
    own unimodal stream from `model` -- unlike the scramble case, the OTHER band
    can't be included as a nuisance regressor here (the placeholder is a constant
    stimulus repeated every row, so its RDM has zero variance and is degenerate for
    banded ridge). `model` is the model name whose bare "_a"/"_v" embeddings are the
    real modality actually fed alongside the placeholder (e.g. "pe-av-small-16-frame"
    or "omni3b_layer9_mp"). `dummy_modality` is "a" (dummy VIDEO + real AUDIO,
    i.e. "_clsav_from_a") or "v" (dummy AUDIO + real VIDEO, "_clsav_from_v").
    """
    dummy_model = f"{model}_clsav_from_{dummy_modality}"
    return PartialRSARun(
        target   = (dummy_model, "av"),
        nuisance = [(model, dummy_modality)],
        label    = f"{dummy_model}_av_INTEGRATION",
        description = (
            f"{model} joint-AV readout with one modality replaced by a fixed "
            f"content-free placeholder (real {dummy_modality} + dummy "
            f"{'video' if dummy_modality == 'a' else 'audio'}), controlling for "
            f"{model}'s own real {dummy_modality}-only output -- tests whether the "
            f"joint embedding carries anything beyond the one real modality present "
            f"(the modality-presence counterpart of the scramble binding control)."
        ),
        kind = "integration",
    )


def _lasttoken_dummy_integration_run(base_model: str, dummy_modality: str) -> PartialRSARun:
    """Dummy-modality counterpart of _lasttoken_scramble_integration_run(): TARGET is
    the "_lt_clsav_from_{a,v}" last-token joint-AV readout; NUISANCE is the
    corresponding "_mp_clsav_from_{a,v}" model's own real-modality stream (mirrors
    _dummy_integration_run's reasoning: the placeholder modality is a constant
    stimulus repeated every row, so it can't be a nuisance band). `base_model` is
    the BARE identifier with neither suffix (e.g. "omni3b_layer9").
    """
    mp_dummy_model = f"{base_model}_mp_clsav_from_{dummy_modality}"
    lt_dummy_model = f"{base_model}_lt_clsav_from_{dummy_modality}"
    return PartialRSARun(
        target   = (lt_dummy_model, "av"),
        nuisance = [(f"{base_model}_mp", dummy_modality)],
        label    = f"{lt_dummy_model}_av_INTEGRATION",
        description = (
            f"{base_model} last-token joint-AV readout with one modality replaced by "
            f"a fixed content-free placeholder (real {dummy_modality} + dummy "
            f"{'video' if dummy_modality == 'a' else 'audio'}), controlling for the "
            f"base model's own real {dummy_modality}-only output -- last-token "
            f"counterpart of _dummy_integration_run()."
        ),
        kind = "integration",
    )


def _cross_baseline_partial_corr_run(model: str, modality: str = "av") -> PartialRSARun:
    """Partial correlation between the target's RDM and the brain RDM,
    controlling for AudioMAE(a) + VideoMAEv2-Large(v) (two independent
    unimodal specialists, not the target's own unimodal streams). Tests
    cross-architecture unique variance for every model in NATIVE_AV_MODELS.
    Routed as "cross_baseline" kind since, unlike the Move-1 integration runs,
    nuisance here is NOT the target's own unimodal decoders.
    """
    return PartialRSARun(
        target   = (model, modality),
        nuisance = [("audiomae", "a"), ("videomaev2-large", "v")],
        label    = f"{model}_{modality}_partial_corr",
        description = (
            f"{model} {modality} joint embedding, controlling for "
            f"AudioMAE (audio) + VideoMAEv2-Large (video) -- cross-architecture "
            f"unique-variance partial correlation, generalized across all "
            f"NATIVE_AV_MODELS."
        ),
    )


PARTIAL_RSA_RUNS: dict[str, PartialRSARun] = {
    # PE-AV joint controlling for AudioMAE(a) + VideoMAEv2-Large(v) has no
    # standalone entry here: it is exactly "partial_corr_pe-av-small-16-frame",
    # produced below by _cross_baseline_partial_corr_run() via the
    # NATIVE_AV_MODELS sweep.
    #
    # Regress out PE-AV's own unimodal decoders. Tests whether the joint
    # embedding encodes cross-modal interactions beyond the simple union of its own
    # audio-only and video-only outputs -- the best-additive integration contrast
    # for the primary model (pe-av-small-16-frame). kind="integration" routes its
    # output to the group_average/<model>_av_INTEGRATION/ convention.
    "integration_pe-av-small-16-frame": PartialRSARun(
        target   = ("pe-av-small-16-frame", "av"),
        nuisance = [("pe-av-small-16-frame", "a"), ("pe-av-small-16-frame", "v")],
        label    = "pe-av-small-16-frame_av_INTEGRATION",
        description = (
            "PE-AV (small 16-frame) AV joint embedding, controlling for its own "
            "audio-only and video-only outputs — tests within-architecture cross-modal "
            "interaction residual (best-additive integration contrast, PRIMARY model)."
        ),
        kind = "integration",
    ),
    # Regress out specialist unimodal models from different architectures
    # (WavLM-Large for audio, PE-Core ViT-L/14 for vision). Tests whether PE-AV
    # captures something beyond strong specialist priors from independent model
    # families — a stricter cross-architecture control than the AudioMAE+VideoMAEv2
    # nuisance used by partial_corr_pe-av-small-16-frame.
    "cross_family_specialist_pe-av-small-16-frame": PartialRSARun(
        target   = ("pe-av-small-16-frame", "av"),
        nuisance = [("wavlm-large", "a"), ("pe-core-l14", "v")],
        label    = "peav_av_partialout_wavlm_a+pecore_v",
        description = (
            "PE-AV (small 16-frame) AV joint embedding, controlling for "
            "WavLM-Large (audio) + PE-Core ViT-L/14 (video) — strict cross-family "
            "specialist baseline."
        ),
    ),
    # ── Best-additive integration contrast, generalized across every
    # native-AV model that has separable _a/_v/_av embeddings at bin5s_skip5s.
    # (imagebind is 2s-only — no _a/_v at 5s — so it is intentionally excluded here.)
    "integration_cav-mae-sync": _own_unimodal_integration_run("cav-mae-sync"),
    "integration_omni3b_layer9":      _own_unimodal_integration_run("omni3b_layer9_mp"),
    "integration_omni3b_layer18":     _own_unimodal_integration_run("omni3b_layer18_mp"),
    "integration_omni3b_layer27":     _own_unimodal_integration_run("omni3b_layer27_mp"),
    "integration_topoomni_layer9":    _own_unimodal_integration_run("topoomni_layer9_mp"),
    "integration_topoomni_layer18":   _own_unimodal_integration_run("topoomni_layer18_mp"),
    "integration_topoomni_layer27":   _own_unimodal_integration_run("topoomni_layer27_mp"),
    # nemotron (omni-embed-nemotron-3b) -- genuinely joint "_av" from the start
    # (no (a+v)/2 placeholder ever existed), so no "_lasttoken" variant is needed.
    "integration_nemotron_layer9":    _own_unimodal_integration_run("nemotron_layer9_mp"),
    "integration_nemotron_layer18":   _own_unimodal_integration_run("nemotron_layer18_mp"),
    "integration_nemotron_layer27":   _own_unimodal_integration_run("nemotron_layer27_mp"),
    "integration_nemotron_layer36":   _own_unimodal_integration_run("nemotron_layer36_mp"),
    # ── "_lasttoken" alternative: genuinely emergent joint-AV readout (not a
    # fixed function of a/v), regressed against the corresponding base model's
    # real (separate-pass) unimodal streams. Requires the Move-1/4 unimodal
    # re-extraction fix (omni3b_extract_intact.py / topo_omni_extract_intact.py).
    "integration_omni3b_layer9_lasttoken":    _lasttoken_integration_run("omni3b_layer9"),
    "integration_omni3b_layer18_lasttoken":   _lasttoken_integration_run("omni3b_layer18"),
    "integration_omni3b_layer27_lasttoken":   _lasttoken_integration_run("omni3b_layer27"),
    "integration_topoomni_layer9_lasttoken":  _lasttoken_integration_run("topoomni_layer9"),
    "integration_topoomni_layer18_lasttoken": _lasttoken_integration_run("topoomni_layer18"),
    "integration_topoomni_layer27_lasttoken": _lasttoken_integration_run("topoomni_layer27"),
    # ── Intact integration run for the topoomni cortical-sheet variant,
    # matching its avscramble/dummy counterparts below.
    "integration_topoomni_layer9_sheet":    _own_unimodal_integration_run("topoomni_layer9_sheet_mp"),
    "integration_topoomni_layer18_sheet":   _own_unimodal_integration_run("topoomni_layer18_sheet_mp"),
    "integration_topoomni_layer27_sheet":   _own_unimodal_integration_run("topoomni_layer27_sheet_mp"),
    "integration_topoomni_layer9_sheet_lasttoken":  _lasttoken_integration_run("topoomni_layer9_sheet"),
    "integration_topoomni_layer18_sheet_lasttoken": _lasttoken_integration_run("topoomni_layer18_sheet"),
    "integration_topoomni_layer27_sheet_lasttoken": _lasttoken_integration_run("topoomni_layer27_sheet"),
    # ── Temporal-scramble binding control. Same own-unimodal integration
    # contrast, computed on the scrambled embeddings. BINDING MAP =
    # integration(intact) - integration(scrambled), computed downstream.
    "integration_pe-av-small-16-frame_avscramble": _own_unimodal_integration_run("pe-av-small-16-frame_avscramble"),
    "integration_cav-mae-sync_avscramble":         _own_unimodal_integration_run("cav-mae-sync_avscramble"),
    # ── Temporal-scramble binding control for omni3b/topoomni (the models
    # where the av=(a+v)/2 circularity fix originated), not just cav-mae-sync.
    "integration_omni3b_layer9_avscramble":    _own_unimodal_integration_run("omni3b_layer9_mp_avscramble"),
    "integration_omni3b_layer18_avscramble":   _own_unimodal_integration_run("omni3b_layer18_mp_avscramble"),
    "integration_omni3b_layer27_avscramble":   _own_unimodal_integration_run("omni3b_layer27_mp_avscramble"),
    "integration_topoomni_layer9_avscramble":  _own_unimodal_integration_run("topoomni_layer9_mp_avscramble"),
    "integration_topoomni_layer18_avscramble": _own_unimodal_integration_run("topoomni_layer18_mp_avscramble"),
    "integration_topoomni_layer27_avscramble": _own_unimodal_integration_run("topoomni_layer27_mp_avscramble"),
    "integration_nemotron_layer9_avscramble":  _own_unimodal_integration_run("nemotron_layer9_mp_avscramble"),
    "integration_nemotron_layer18_avscramble": _own_unimodal_integration_run("nemotron_layer18_mp_avscramble"),
    "integration_nemotron_layer27_avscramble": _own_unimodal_integration_run("nemotron_layer27_mp_avscramble"),
    "integration_nemotron_layer36_avscramble": _own_unimodal_integration_run("nemotron_layer36_mp_avscramble"),
    "integration_omni3b_layer9_lasttoken_avscramble":    _lasttoken_scramble_integration_run("omni3b_layer9"),
    "integration_omni3b_layer18_lasttoken_avscramble":   _lasttoken_scramble_integration_run("omni3b_layer18"),
    "integration_omni3b_layer27_lasttoken_avscramble":   _lasttoken_scramble_integration_run("omni3b_layer27"),
    "integration_topoomni_layer9_lasttoken_avscramble":  _lasttoken_scramble_integration_run("topoomni_layer9"),
    "integration_topoomni_layer18_lasttoken_avscramble": _lasttoken_scramble_integration_run("topoomni_layer18"),
    "integration_topoomni_layer27_lasttoken_avscramble": _lasttoken_scramble_integration_run("topoomni_layer27"),
    # ── Temporal-scramble control for the topoomni cortical-sheet variant,
    # matching the non-sheet readout's sweep above.
    "integration_topoomni_layer9_sheet_avscramble":  _own_unimodal_integration_run("topoomni_layer9_sheet_mp_avscramble"),
    "integration_topoomni_layer18_sheet_avscramble": _own_unimodal_integration_run("topoomni_layer18_sheet_mp_avscramble"),
    "integration_topoomni_layer27_sheet_avscramble": _own_unimodal_integration_run("topoomni_layer27_sheet_mp_avscramble"),
    "integration_topoomni_layer9_sheet_lasttoken_avscramble":  _lasttoken_scramble_integration_run("topoomni_layer9_sheet"),
    "integration_topoomni_layer18_sheet_lasttoken_avscramble": _lasttoken_scramble_integration_run("topoomni_layer18_sheet"),
    "integration_topoomni_layer27_sheet_lasttoken_avscramble": _lasttoken_scramble_integration_run("topoomni_layer27_sheet"),

    # ── Modality-presence control: dummy-modality (clsav_from_a/_v)
    # counterpart of the scramble integration runs above. Expectation (per
    # the AV-integration hypothesis): both this AND the avscramble integration
    # runs should show WEAKER alignment in true integration regions than the
    # intact integration run, while non-integration regions stay flat.
    "integration_pe-av-small-16-frame_clsav_from_a": _dummy_integration_run("pe-av-small-16-frame", "a"),
    "integration_pe-av-small-16-frame_clsav_from_v": _dummy_integration_run("pe-av-small-16-frame", "v"),
    "integration_omni3b_layer9_clsav_from_a":    _dummy_integration_run("omni3b_layer9_mp", "a"),
    "integration_omni3b_layer9_clsav_from_v":    _dummy_integration_run("omni3b_layer9_mp", "v"),
    "integration_omni3b_layer18_clsav_from_a":   _dummy_integration_run("omni3b_layer18_mp", "a"),
    "integration_omni3b_layer18_clsav_from_v":   _dummy_integration_run("omni3b_layer18_mp", "v"),
    "integration_omni3b_layer27_clsav_from_a":   _dummy_integration_run("omni3b_layer27_mp", "a"),
    "integration_omni3b_layer27_clsav_from_v":   _dummy_integration_run("omni3b_layer27_mp", "v"),
    "integration_topoomni_layer9_clsav_from_a":   _dummy_integration_run("topoomni_layer9_mp", "a"),
    "integration_topoomni_layer9_clsav_from_v":   _dummy_integration_run("topoomni_layer9_mp", "v"),
    "integration_topoomni_layer18_clsav_from_a":  _dummy_integration_run("topoomni_layer18_mp", "a"),
    "integration_topoomni_layer18_clsav_from_v":  _dummy_integration_run("topoomni_layer18_mp", "v"),
    "integration_topoomni_layer27_clsav_from_a":  _dummy_integration_run("topoomni_layer27_mp", "a"),
    "integration_topoomni_layer27_clsav_from_v":  _dummy_integration_run("topoomni_layer27_mp", "v"),
    "integration_topoomni_layer9_sheet_clsav_from_a":  _dummy_integration_run("topoomni_layer9_sheet_mp", "a"),
    "integration_topoomni_layer9_sheet_clsav_from_v":  _dummy_integration_run("topoomni_layer9_sheet_mp", "v"),
    "integration_topoomni_layer18_sheet_clsav_from_a": _dummy_integration_run("topoomni_layer18_sheet_mp", "a"),
    "integration_topoomni_layer18_sheet_clsav_from_v": _dummy_integration_run("topoomni_layer18_sheet_mp", "v"),
    "integration_topoomni_layer27_sheet_clsav_from_a": _dummy_integration_run("topoomni_layer27_sheet_mp", "a"),
    "integration_topoomni_layer27_sheet_clsav_from_v": _dummy_integration_run("topoomni_layer27_sheet_mp", "v"),
    "integration_nemotron_layer9_clsav_from_a":   _dummy_integration_run("nemotron_layer9_mp", "a"),
    "integration_nemotron_layer9_clsav_from_v":   _dummy_integration_run("nemotron_layer9_mp", "v"),
    "integration_nemotron_layer18_clsav_from_a":  _dummy_integration_run("nemotron_layer18_mp", "a"),
    "integration_nemotron_layer18_clsav_from_v":  _dummy_integration_run("nemotron_layer18_mp", "v"),
    "integration_nemotron_layer27_clsav_from_a":  _dummy_integration_run("nemotron_layer27_mp", "a"),
    "integration_nemotron_layer27_clsav_from_v":  _dummy_integration_run("nemotron_layer27_mp", "v"),
    "integration_nemotron_layer36_clsav_from_a":  _dummy_integration_run("nemotron_layer36_mp", "a"),
    "integration_nemotron_layer36_clsav_from_v":  _dummy_integration_run("nemotron_layer36_mp", "v"),

    # ── Last-token dummy counterparts (mirrors the lasttoken_avscramble block).
    "integration_omni3b_layer9_lasttoken_clsav_from_a":    _lasttoken_dummy_integration_run("omni3b_layer9", "a"),
    "integration_omni3b_layer9_lasttoken_clsav_from_v":    _lasttoken_dummy_integration_run("omni3b_layer9", "v"),
    "integration_omni3b_layer18_lasttoken_clsav_from_a":   _lasttoken_dummy_integration_run("omni3b_layer18", "a"),
    "integration_omni3b_layer18_lasttoken_clsav_from_v":   _lasttoken_dummy_integration_run("omni3b_layer18", "v"),
    "integration_omni3b_layer27_lasttoken_clsav_from_a":   _lasttoken_dummy_integration_run("omni3b_layer27", "a"),
    "integration_omni3b_layer27_lasttoken_clsav_from_v":   _lasttoken_dummy_integration_run("omni3b_layer27", "v"),
    "integration_topoomni_layer9_lasttoken_clsav_from_a":  _lasttoken_dummy_integration_run("topoomni_layer9", "a"),
    "integration_topoomni_layer9_lasttoken_clsav_from_v":  _lasttoken_dummy_integration_run("topoomni_layer9", "v"),
    "integration_topoomni_layer18_lasttoken_clsav_from_a": _lasttoken_dummy_integration_run("topoomni_layer18", "a"),
    "integration_topoomni_layer18_lasttoken_clsav_from_v": _lasttoken_dummy_integration_run("topoomni_layer18", "v"),
    "integration_topoomni_layer27_lasttoken_clsav_from_a": _lasttoken_dummy_integration_run("topoomni_layer27", "a"),
    "integration_topoomni_layer27_lasttoken_clsav_from_v": _lasttoken_dummy_integration_run("topoomni_layer27", "v"),
    "integration_topoomni_layer9_sheet_lasttoken_clsav_from_a":  _lasttoken_dummy_integration_run("topoomni_layer9_sheet", "a"),
    "integration_topoomni_layer9_sheet_lasttoken_clsav_from_v":  _lasttoken_dummy_integration_run("topoomni_layer9_sheet", "v"),
    "integration_topoomni_layer18_sheet_lasttoken_clsav_from_a": _lasttoken_dummy_integration_run("topoomni_layer18_sheet", "a"),
    "integration_topoomni_layer18_sheet_lasttoken_clsav_from_v": _lasttoken_dummy_integration_run("topoomni_layer18_sheet", "v"),
    "integration_topoomni_layer27_sheet_lasttoken_clsav_from_a": _lasttoken_dummy_integration_run("topoomni_layer27_sheet", "a"),
    "integration_topoomni_layer27_sheet_lasttoken_clsav_from_v": _lasttoken_dummy_integration_run("topoomni_layer27_sheet", "v"),
}

# ── Cross-baseline partial correlation, one entry per native AV model.
# name tag: _partial_corr.
PARTIAL_RSA_RUNS.update({
    f"partial_corr_{m}": _cross_baseline_partial_corr_run(m)
    for m in NATIVE_AV_MODELS
})


# ── Diagonal-masking configurations ──────────────────────────────────────────
# Default model/modality used by rdm_diagonal.py.

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
                            bin_sec: float, skip_sec: float | None = None) -> Path:
    """Resolve embedding path and raise FileNotFoundError if missing."""
    path = emb_path(embeddings_dir, model, modality, bin_sec, skip_sec)
    if not path.exists():
        skip_sec_int = int(skip_sec) if skip_sec is not None else int(bin_sec)
        raise FileNotFoundError(
            f"Embedding not found: {path}\n"
            f"Expected: {embeddings_dir}/{model}/bin{int(bin_sec)}s_skip{skip_sec_int}s/"
            f"{model}_{modality}.npy"
        )
    return path
