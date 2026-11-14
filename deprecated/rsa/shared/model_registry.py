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

from shared.model_registry import (  # noqa: F401
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    BARE_NAME_AV_MODELS,
    PARTIAL_RSA_RUNS,
    PartialRSARun,
    RESIDUALIZED_AV_MODELS,
    TR_DEFAULT,
    _dummy_integration_run,
    _lasttoken_dummy_integration_run,
    _lasttoken_integration_run,
    _lasttoken_scramble_integration_run,
    _own_unimodal_integration_run,
    check_embeddings_exist,
    emb_path,
    validate_run,
)


# ── Path resolver ─────────────────────────────────────────────────────────────


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
    # Legacy bare probes retained by the natural-AV group-average sweep. Their
    # A/V/AV embeddings share the thinker or sheet space.
    "omni3b_layer1": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker hidden state, legacy bare layer-1 probe"},
    "omni3b_layer35": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Qwen2.5-Omni-3B thinker hidden state, legacy bare layer-35 probe"},
    "topoomni_layer1": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, legacy bare layer-1 probe"},
    "topoomni_layer35": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni thinker hidden state, legacy bare layer-35 probe"},
    "topoomni_layer1_sheet": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni cortical sheet, legacy bare layer-1 probe"},
    "topoomni_layer35_sheet": {"modalities": ["av", "a", "v"], "joint": True,
        "description": "Topo-Omni cortical sheet, legacy bare layer-35 probe"},
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
    "nemotron_layer9_lt": {"modalities": ["av"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 9, last-token joint-AV readout"},
    "nemotron_layer18_lt": {"modalities": ["av"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 18, last-token joint-AV readout"},
    "nemotron_layer27_lt": {"modalities": ["av"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 27, last-token joint-AV readout"},
    "nemotron_layer36_lt": {"modalities": ["av"], "joint": True,
        "description": "Omni-Embed-Nemotron-3B, layer 36, last-token joint-AV readout"},
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
# generalize the residual/partial-correlation analyses (linear_resid_unimodal,
# partial_corr, projection_resid_own) across all native AV models instead of
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


# ── Descriptive AV comparison baselines ──────────────────────────────────────
# Group-average AV searchlight maps use three audio/video reference pairs:
#
#   own          PE-AV/CAV-MAE cls-a + cls-v; for the Omni families, the
#                pre-fusion audio/visual encoder probes (not thinker a/v).
#   unimodal     AudioMAE + VideoMAEv2.
#   text_aligned WavLM + PE-Core (the project's language/semantic-aligned
#                specialist pair).
#
# Shared by dependency scheduling and derived-map construction.
AV_DERIVED_COMMON_BASELINES: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "unimodal": (("audiomae", "a"), ("videomaev2-large", "v")),
    "text_aligned": (("wavlm-large", "a"), ("pe-core-l14", "v")),
}


def av_derived_baselines(
    model: str,
) -> dict[str, tuple[tuple[str, str], tuple[str, str]]] | None:
    """Return the three A/V reference pairs for a natural joint-AV model.

    ``None`` means that *model* is not one of the natural AV targets covered by
    the descriptive maps. In particular, scramble, dummy-modality, and
    residual pseudo-models are intentionally excluded.
    """
    excluded = (
        "_avscramble",
        "_clsav_from_",
        "_linear_resid",  # prefix match: also catches _unimodal/_own/_encoder variants
        "_projection_resid",  # prefix match: also catches the _own variant
    )
    if any(marker in model for marker in excluded):
        return None

    if model in {"pe-av-small-16-frame", "cav-mae-sync"}:
        own_model = model
    else:
        family = next(
            (name for name in ("omni3b", "topoomni", "nemotron")
             if model.startswith(f"{name}_layer")),
            None,
        )
        if family is None:
            return None
        own_model = f"{family}_encoder_penultimate"

    return {
        "own": ((own_model, "a"), (own_model, "v")),
        **AV_DERIVED_COMMON_BASELINES,
    }

# ── Auto-derived residual pseudo-models ───────────────────────────────────────
# _linear_resid_unimodal pseudo-models are generated on disk by
# notebooks/feature_extraction/compute_linear_residual_embeddings.py for the
# supported residual-analysis roster (ridge residual after regressing out
# AudioMAE(a) + VideoMAEv2-Large(v)). Registered here for discoverability
# -- rsa/searchlight.py and rsa/partial_rsa.py only need
# the .npy file to exist at the standard emb_path() location, not a MODELS entry.
for _m in dict.fromkeys([*NATIVE_AV_MODELS, *RESIDUALIZED_AV_MODELS]):
    MODELS[f"{_m}_av_linear_resid_unimodal"] = {
        "modalities": ["av"], "joint": True,
        "description": (
            f"{_m} AV joint embedding, ridge residual after regressing out "
            f"AudioMAE(a) + VideoMAEv2-Large(v) -- unique variance beyond two "
            f"independent unimodal specialists (linear_resid_unimodal)."
        ),
    }
del _m

# pe-av-small-16-frame_av_linear_resid_own: the one non-comprehension member of
# this family -- ridge residual of PE-AV's AV joint embedding after regressing
# out PE-AV's OWN audio and video streams (nuisance = own a/v, not the external
# AudioMAE/VideoMAEv2 pair above). Within-architecture unique variance; the
# ridge counterpart of PE-AV's own-stream *_av_projection_resid_own (same nuisance,
# geometric projection instead of ridge). Generated by
# compute_linear_residual_embeddings.py --nuisance pe-av-small-16-frame:a
# pe-av-small-16-frame:v --variant-suffix _own. PE-AV only -- not swept across
# RESIDUALIZED_AV_MODELS, so it does not belong in the comprehension above.
MODELS["pe-av-small-16-frame_av_linear_resid_own"] = {
    "modalities": ["av"], "joint": True,
    "description": (
        "pe-av-small-16-frame AV joint embedding, ridge residual after "
        "regressing out PE-AV's own audio and video streams -- "
        "within-architecture unique variance, the ridge counterpart of "
        "projection_resid_own (linear_resid_own)."
    ),
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

    # "integration" runs regress a native-AV model's OWN unimodal streams out of
    # its OWN joint embedding (the best-additive-combination contrast).
    # "cross_baseline" runs regress OTHER models' unimodal streams out instead.
    # Both kinds share one output convention (see partial_rsa.py's save block):
    # group_average/<label>/k{K}_delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}/<file>.


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


# ── Cross-baseline partial correlation, one entry per supported AV model.
# name tag: _partial_corr.
PARTIAL_RSA_RUNS.update({
    f"partial_corr_{m}": _cross_baseline_partial_corr_run(m)
    for m in dict.fromkeys([*NATIVE_AV_MODELS, *RESIDUALIZED_AV_MODELS])
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


