"""
topo_omni_extract.py
=====================
Standalone script version of topo_omni.ipynb -- extracts topo-omni (epfl-neuroai/topo-omni)
embeddings for RSA/encoding at the segment durations listed in SEGMENT_DURATIONS below.

Run with:
    conda run --no-capture-output -n topo_omni \
        python "notebooks/feature_extraction/topo_omni_extract.py"

Model
-----
epfl-neuroai/topo-omni: Qwen2.5-Omni-3B Thinker fine-tuned with a CorticalAdaptor spliced
after every audio/vision/text layer, projecting activations onto a shared 304x512
"cortical sheet" (see arXiv:2606.09770). Requires the cloned repo at TOPO_OMNI_REPO below
(custom modeling class -- the stock transformers class does not know about the adaptors).

Two embedding families are saved per quarter-layer, per segment, per modality mask:
    {MODEL_TAG}_layer{idx}            -- hidden / X_hat (post-adaptor residual stream,
                                          directly comparable to omni3b.ipynb's output)
    {MODEL_TAG}_layer{idx}_sheet      -- cortical sheet / Z (the topographic code itself)
Both come from the same forward pass via a hook on each chosen layer's CorticalAdaptor.

Source data
-----------
Uses segmented_stimulus/filtered/ -- pre-chunked mp4/wav files for all 18 videos.
Structure:
    Video{N}/Video{N}_chunks_{D}s/Video{N}_part_XXX.mp4
    Audio{N}/Audio{N}_chunks_{D}s/Audio{N}_part_XXX.wav
Video .mp4 files carry no embedded audio track; audio is loaded from the matching .wav.

Memory notes (12 GB GPU)
-------------------------
Model weights alone (~10.5 GB bf16) leave very little headroom. Two fixes were required
to avoid OOM on a normal forward pass, applied below:
  1. config.apply_spatial_loss = False -- skips the outer model's fused `unified_sheet`
     construction (not used here; we read per-layer cortical_sheet via hooks instead).
     This does NOT stop the audio/vision/text sub-modules from unconditionally building
     their OWN per-layer cortical_sheet tuples -- see fix 3.
  2. model.lm_head = nn.Identity() -- the stock forward always projects every token
     position to the full 151936-token vocabulary (>1 GB by itself); we never read
     `logits`, only hooked cortical_adaptor outputs.
  3. Local patch in the cloned repo (src/models/qwen2_5_omni.py, 3 one-line edits):
     the audio encoder, vision encoder, and text model each accumulate a growing tuple
     of every layer's cortical-sheet activity, keeping it all resident on GPU for the
     whole forward and only moving it to CPU right before returning. We patched
     `.detach().cpu()` onto each layer's append instead of only at the end -- same
     final values, just freed from GPU memory as we go instead of all at once at the end.

Both a `hidden`/`sheet` empirical layer-selection diagnostic (50 sampled segments, RDMs
+ Spearman-vs-last-layer table, saved as PNGs) and the full per-segment extraction loop
run every time this script is invoked; already-complete output durations are skipped.
"""

import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ.pop("SOCKS_PROXY", None)
os.environ.pop("socks_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT, DATA, OUTPUTS, MODELS  # noqa: E402

# topo-omni ships a *custom* modeling file (CorticalAdaptor modules spliced after every
# audio/vision/text layer). The stock transformers Qwen2_5OmniThinkerForConditionalGeneration
# class does not know about these adaptors -- loading these weights with the stock class
# would silently skip ~100 adaptor tensors as "unexpected keys" and run an untrained,
# incorrect forward pass (unlike the harmless token2wav/talker skip for vanilla omni3b).
# We therefore import the model class from the cloned repo's src/ directory instead.
# Both the repo root AND src/ must be on sys.path: entry points import `models.qwen2_5_omni`
# (expects src/ on path), but that file itself does `from src.models.spatial_utils import ...`
# (expects the repo ROOT on path, treating src/ as a namespace package) -- an inconsistency
# in the repo's own import style, not a typo; both paths are needed simultaneously.
TOPO_OMNI_REPO = ROOT / "Model Repos/topo-omni"
assert TOPO_OMNI_REPO.is_dir(), f"topo-omni repo not found at {TOPO_OMNI_REPO}"
sys.path.insert(0, str(TOPO_OMNI_REPO))
sys.path.insert(0, str(TOPO_OMNI_REPO / "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, CorticalAdaptor
from qwen_vl_utils import process_vision_info

import transformers
print(f"transformers version : {transformers.__version__}")
print(f"torch version        : {torch.__version__}")
print(f"CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM                 : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Download topo-omni to external HDD (skip if already present) ─────────────
EXTERNAL_HF_CACHE = str(MODELS / "hub")
TOPO_MODEL_ID     = "epfl-neuroai/topo-omni"

def _weights_present(ext_model_dir: Path) -> bool:
    snap_dir = ext_model_dir / "snapshots"
    if not snap_dir.exists():
        return False
    snaps = [s for s in snap_dir.iterdir() if s.is_dir()]
    if not snaps:
        return False
    snap = sorted(snaps)[-1]
    idx_path = snap / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        expected = set(idx["weight_map"].values())
        missing = [s for s in expected if not (snap / s).exists() or (snap / s).stat().st_size == 0]
        if missing:
            print(f"  Incomplete: {len(missing)}/{len(expected)} shards missing")
            return False
        print(f"  All {len(expected)} shards present in {snap.name[:12]}...")
        return True
    return any(f.suffix == ".safetensors" and "index" not in f.name for f in snap.iterdir())

ext_dir    = Path(EXTERNAL_HF_CACHE) / "models--epfl-neuroai--topo-omni"
sym_target = Path.home() / ".cache" / "huggingface" / "hub" / "models--epfl-neuroai--topo-omni"

if not _weights_present(ext_dir):
    print("Downloading epfl-neuroai/topo-omni ...")
    snapshot_download(TOPO_MODEL_ID, cache_dir=EXTERNAL_HF_CACHE,
                      ignore_patterns=["*.msgpack", "flax_model*"])
else:
    print(f"Weights already present: {ext_dir}")

if sym_target.exists() and not sym_target.is_symlink():
    shutil.rmtree(sym_target)
if not sym_target.exists():
    sym_target.symlink_to(ext_dir)
    print(f"Symlink created: {sym_target} -> {ext_dir}")
else:
    print(f"Symlink OK: {sym_target}")


# ── CONFIG ────────────────────────────────────────────────────────────────────
# All paths and parameters live here. Nothing is hardcoded elsewhere.

# transformers 4.57.6's cached_files() fails to resolve a bare hub-id string against
# a local-only cache with huggingface_hub 0.36.2 (LocalEntryNotFoundError even though
# the snapshot is fully present -- verified via hf_hub_download/try_to_load_from_cache,
# which both resolve it fine). The authors' own eval/extract scripts sidestep this
# entirely by always loading from a local checkpoint *directory*, never a hub id --
# so we resolve the local snapshot directory ourselves and pass that to from_pretrained.
MODEL_PATH = snapshot_download(TOPO_MODEL_ID, local_files_only=True)
print(f"Resolved local snapshot dir: {MODEL_PATH}")

DATA_BASE       = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE = OUTPUTS / "model_embeddings"
DIAGNOSTICS_DIR = EMBEDDINGS_BASE / "topoomni_diagnostics"

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# BIN_SEC: window duration in seconds; SKIP_SEC: stride (= BIN_SEC -> no overlap)
BIN_SEC  = 5.0
SKIP_SEC = 5.0

# Each tuple: (bin_sec, skip_sec, segments_root). Add (10.0, 10.0, DATA_BASE) for 10s runs.
SEGMENT_DURATIONS = [
    (5.0,  5.0,  DATA_BASE),
]
MODEL_TAG = "topoomni"   # used in all output filenames and directory names

# Audio sampling rate expected by WhisperFeatureExtractor inside this processor.
AUDIO_SR = 16000
# ─────────────────────────────────────────────────────────────────────────────
print(f"MODEL_TAG       : {MODEL_TAG}")
print(f"EMBEDDINGS_BASE : {EMBEDDINGS_BASE}")
EMBEDDINGS_BASE.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()

print("Loading topo-omni Thinker config ...")
model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_PATH, local_files_only=True)

# CRITICAL: without forcing is_training=False on every sub-config, CorticalAdaptor.__init__
# skips registering the W_pinv buffer, and the adaptor's eval-mode forward path
# (`else: W_pinv = self.W_pinv`) raises AttributeError.
model_config.audio_config.is_training = False
model_config.vision_config.is_training = False
model_config.text_config.is_training = False
model_config.apply_spatial_loss = False  # see module docstring, memory fix 1

print("Loading topo-omni Thinker weights ...")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    config=model_config,
    device_map={"": DEVICE},   # force all submodules onto one device
    torch_dtype=DTYPE,
    attn_implementation="sdpa",
    local_files_only=True,
)
model.eval()
model.lm_head = torch.nn.Identity()  # see module docstring, memory fix 2

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

_MODEL_DEVICE = next(model.parameters()).device
print(f"Thinker loaded on {_MODEL_DEVICE}.")

# ── Read layer count from config (never hardcode) ────────────────────────────
N_LAYERS = model.config.text_config.num_hidden_layers
QUARTER_LAYERS = sorted(set([
    1, 2, 4,
    N_LAYERS // 4,
    N_LAYERS // 2,
    3 * N_LAYERS // 4,
    N_LAYERS - 1,
]))
print(f"N_LAYERS       : {N_LAYERS}")
print(f"QUARTER_LAYERS : {QUARTER_LAYERS}")

# Only {N_LAYERS//4, N_LAYERS//2, 3*N_LAYERS//4} (9/18/27 for N_LAYERS=36) have a
# "_lasttoken"-now-"_lt" counterpart (extracted separately by topo_omni_extract_
# intact.py) -- those get the explicit "_mp" mean-pool tag so the two variants
# stay distinguishable. The other quarter-layer sweep probes (1, 2, 4, N_LAYERS-1)
# have no lasttoken counterpart, so there is nothing to disambiguate; they keep
# their bare "{MODEL_TAG}_layer{idx}" name unchanged.
MP_TAGGED_LAYERS = {N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4}


def _layer_tag(idx):
    return f"{MODEL_TAG}_layer{idx}_mp" if idx in MP_TAGGED_LAYERS else f"{MODEL_TAG}_layer{idx}"


def _sheet_tag(idx):
    return f"{MODEL_TAG}_layer{idx}_sheet_mp" if idx in MP_TAGGED_LAYERS else f"{MODEL_TAG}_layer{idx}_sheet"

# ── Verify token IDs from config (read dynamically, never hardcode) ───────────
AUDIO_TOKEN_ID = model.config.audio_token_id
VIDEO_TOKEN_ID = model.config.video_token_id
assert AUDIO_TOKEN_ID is not None, "model.config.audio_token_id is None -- check model config"
assert VIDEO_TOKEN_ID is not None, "model.config.video_token_id is None -- check model config"
print(f"audio_token_id : {AUDIO_TOKEN_ID}")
print(f"video_token_id : {VIDEO_TOKEN_ID}")

_proc_audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
_proc_video_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
assert _proc_audio_id == AUDIO_TOKEN_ID, (
    f"Audio token ID mismatch: config={AUDIO_TOKEN_ID}, processor={_proc_audio_id}"
)
assert _proc_video_id == VIDEO_TOKEN_ID, (
    f"Video token ID mismatch: config={VIDEO_TOKEN_ID}, processor={_proc_video_id}"
)
print("Token ID cross-check passed.")

torch.cuda.synchronize()
mem_gb = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory allocated after loading: {mem_gb:.2f} GB")


# ── TOKEN MASK UTILITIES ──────────────────────────────────────────────────────
# Identical convention to omni3b.ipynb: same tokenizer/vocab, same token IDs.

_masks_verified = False


def get_audio_token_mask(ids: torch.Tensor) -> torch.Tensor:
    """Bool mask (seq_len,) True at audio token positions."""
    return ids == AUDIO_TOKEN_ID


def get_video_token_mask(ids: torch.Tensor) -> torch.Tensor:
    """Bool mask (seq_len,) True at video token positions."""
    return ids == VIDEO_TOKEN_ID


def get_av_token_mask(ids: torch.Tensor) -> torch.Tensor:
    """Bool mask (seq_len,) True at audio OR video token positions."""
    return get_audio_token_mask(ids) | get_video_token_mask(ids)


def get_all_masks(input_ids: torch.Tensor) -> tuple:
    """Return (av_mask, a_mask, v_mask) as CPU bool tensors of shape (seq_len,).

    First call prints a verification report and asserts independently:
    - at least one audio token found
    - at least one video token found
    - |av_mask| == |a_mask| + |v_mask|  (IDs are disjoint, no double-counting)

    input_ids: (1, seq_len) or (seq_len,).
    """
    global _masks_verified
    ids     = input_ids.squeeze(0)
    a_mask  = get_audio_token_mask(ids)
    v_mask  = get_video_token_mask(ids)
    av_mask = a_mask | v_mask
    if not _masks_verified:
        n_a  = int(a_mask.sum())
        n_v  = int(v_mask.sum())
        n_av = int(av_mask.sum())
        n_t  = int((~av_mask).sum())
        print(f"[Mask verify]  audio_id={AUDIO_TOKEN_ID}  video_id={VIDEO_TOKEN_ID}")
        print(f"[Mask verify]  audio={n_a}  video={n_v}  av={n_av}  text={n_t}  total={len(ids)}")
        assert n_a  > 0, f"No audio tokens -- AUDIO_TOKEN_ID={AUDIO_TOKEN_ID} not in sequence"
        assert n_v  > 0, f"No video tokens -- VIDEO_TOKEN_ID={VIDEO_TOKEN_ID} not in sequence"
        assert n_av == n_a + n_v, "AV count != audio+video -- token ID overlap detected"
        print("[Mask verify]  all assertions passed.")
        _masks_verified = True
    return av_mask.cpu(), a_mask.cpu(), v_mask.cpu()


# ── DYNAMIC BATCH SIZE ────────────────────────────────────────────────────────
# Our segmented_stimulus .mp4 files carry NO embedded audio track (audio lives in
# matching .wav files), so we build the conversation with separate "video" and
# "audio" content entries and load the .wav via PyAV ourselves, rather than relying
# on the model's use_audio_in_video extraction (which assumes audio is embedded in
# the same video file).

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Decode audio from an mp4 or wav file via PyAV."""
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return np.zeros(target_sr, dtype=np.float32)
        native_sr = stream.sample_rate
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)  # stereo -> mono
            chunks.append(arr)
    if not chunks:
        return np.zeros(target_sr, dtype=np.float32)
    audio = np.concatenate(chunks)
    if native_sr != target_sr:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), native_sr, target_sr
        ).numpy()
    return audio


def _find_audio_path(video_path: Path) -> Path:
    """Video{N}/Video{N}_chunks_{dur}/Video{N}_part_{M}.mp4 -> matching Audio{...}.wav"""
    parts = [p.replace("Video", "Audio") for p in video_path.parts]
    wav_path = Path(*parts).with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    return video_path  # fall back to (silent) audio-less extraction from the mp4 itself


def _build_inputs_for_video(video_path: Path) -> tuple:
    """Build processor inputs for a single video+audio file."""
    audio_path = _find_audio_path(video_path)
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text",  "text":  "Describe what you see and hear."},
        ]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    _, frames = process_vision_info(msgs)
    audio_array = _load_audio(audio_path, AUDIO_SR)
    return text, frames, audio_array


def find_safe_batch_size(
    sample_video_path: Path,
    max_batch: int = 8,
) -> int:
    """Probe batch sizes 1..max_batch; return (largest safe - 1), minimum 1."""
    text_single, frames_single, audio_single = _build_inputs_for_video(sample_video_path)
    last_ok = 0
    for bs in range(1, max_batch + 1):
        torch.cuda.empty_cache()
        try:
            texts  = [text_single] * bs
            frames = frames_single * bs
            audios = np.stack([audio_single] * bs, axis=0) if bs > 1 else audio_single
            inputs = processor(
                text=texts,
                videos=frames,
                audio=audios,
                sampling_rate=AUDIO_SR,
                return_tensors="pt",
            ).to(_MODEL_DEVICE)
            with torch.inference_mode():
                _ = model(**inputs, output_hidden_states=False)
            last_ok = bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM at batch_size={bs}")
            break
        finally:
            torch.cuda.empty_cache()
    safe = max(1, last_ok - 1)
    print(f"Safe batch size: {safe}  (last successful: {last_ok})")
    return safe


# ── SINGLE SEGMENT EXTRACTION (hook-based, memory-efficient) ─────────────────
# Hooks are placed on model.model.cortical_adaptors[i] for each chosen layer i.
# CorticalAdaptor.forward returns (X_hat, Z):
#   X_hat -- post-adaptor residual stream (what actually continues to the next layer)
#   Z     -- cortical-sheet activation (same dimensionality; the topographic code)
# Both are captured from the SAME hook call and moved to CPU immediately, so peak
# GPU memory stays proportional to len(QUARTER_LAYERS), not N_LAYERS.
#
# Whisper padding trim logic identical to omni3b.ipynb.

_audio_trim_logged = False


def _find_cortical_adaptors(n_expected: int) -> torch.nn.ModuleList:
    """Locate the ModuleList of CorticalAdaptor instances in the text decoder stack."""
    direct = getattr(getattr(model, "model", None), "cortical_adaptors", None)
    if direct is not None and len(direct) == n_expected and isinstance(direct[0], CorticalAdaptor):
        return direct
    for _, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList)
                and len(mod) == n_expected
                and isinstance(mod[0], CorticalAdaptor)):
            return mod
    raise RuntimeError(f"Could not find CorticalAdaptor ModuleList of length {n_expected}")


_cortical_adaptors = _find_cortical_adaptors(N_LAYERS)
print(f"Cortical adaptors found: {len(_cortical_adaptors)}")


def extract_av_embeddings(video_path: Path, target_layers: list | None = None):
    """Extract hidden (X_hat) and sheet (Z) activations via forward hooks.

    Returns
    -------
    hidden_states : dict  {layer_idx -> tensor(1, seq_len, D) on CPU}  -- X_hat
    cortical_sheet : dict {layer_idx -> tensor(1, seq_len, D) on CPU}  -- Z
    av_mask, a_mask, v_mask : bool tensors (seq_len,) on CPU
    """
    global _audio_trim_logged
    if target_layers is None:
        target_layers = QUARTER_LAYERS

    text, frames, audio_array = _build_inputs_for_video(video_path)

    n_samples = len(audio_array)
    n_mel = int(np.ceil(n_samples / 160))
    n_valid_audio = min(int(np.ceil(n_mel / 2)), 1500)

    inputs = processor(
        text=[text],
        videos=frames,
        audio=audio_array,
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
    ).to(_MODEL_DEVICE)

    captured_hidden: dict[int, torch.Tensor] = {}
    captured_sheet:  dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(idx: int):
        def _hook(module, inp, out):
            x_hat, z = out
            captured_hidden[idx] = x_hat.detach().cpu()
            captured_sheet[idx]  = z.detach().cpu()
        return _hook

    for i in target_layers:
        handles.append(_cortical_adaptors[i].register_forward_hook(_make_hook(i)))

    try:
        with torch.inference_mode():
            model(**inputs, output_hidden_states=False)
    finally:
        for h in handles:
            h.remove()

    _, a_mask_full, v_mask = get_all_masks(inputs["input_ids"])
    del inputs
    torch.cuda.empty_cache()

    a_positions = a_mask_full.nonzero(as_tuple=True)[0]
    n_clip = min(n_valid_audio, len(a_positions))
    a_mask = torch.zeros_like(a_mask_full)
    a_mask[a_positions[:n_clip]] = True
    av_mask = a_mask | v_mask

    if not _audio_trim_logged:
        print(f"[Audio trim]  valid={n_clip}/{len(a_positions)} audio tokens "
              f"(n_samples={n_samples}, n_mel={n_mel}, n_valid_audio={n_valid_audio})")
        _audio_trim_logged = True

    return captured_hidden, captured_sheet, av_mask, a_mask, v_mask


# ── MEAN POOL EMBEDDING ───────────────────────────────────────────────────────

def get_mean_pool_embedding(
    hidden_states: dict,
    mask: torch.Tensor,
    layer_idx: int,
) -> np.ndarray:
    """Mean pool hidden states at positions where mask is True.

    Works identically for the `hidden_states` (X_hat) dict or the `cortical_sheet`
    (Z) dict -- both have the same (1, seq_len, D) shape per layer.

    Returns np.ndarray of shape (hidden_dim,). Returns zeros if mask is all-False.
    """
    if layer_idx not in hidden_states:
        available = sorted(hidden_states.keys())
        raise KeyError(f"layer_idx={layer_idx} not in captured layers {available}")
    hs = hidden_states[layer_idx]  # (1, seq_len, D) on CPU
    if not mask.any():
        return np.zeros(hs.shape[-1], dtype=np.float32)
    return hs[0, mask, :].mean(dim=0).float().numpy()


# ── EMPIRICAL LAYER SELECTION ─────────────────────────────────────────────────
# Sample 50 segments from the primary bin duration, extract QUARTER_LAYERS
# embeddings for BOTH hidden (X_hat) and sheet (Z), compute RDMs, and compare
# Spearman r against each variant's own last-layer RDM. Saves PNGs + sample .npy
# files under DIAGNOSTICS_DIR / EMBEDDINGS_BASE respectively.

_dur_int  = int(BIN_SEC)
_skip_int = int(SKIP_SEC)
_chunk_suffix = (f"_av_chunks_{_dur_int}s" if _skip_int == _dur_int
                 else f"_av_chunks_{_dur_int}s_skip{_skip_int}s")
all_segs = natsorted(list(DATA_BASE.rglob(f"*{_chunk_suffix}/*.mp4")), key=lambda p: p.name)
assert len(all_segs) > 0, f"No {BIN_SEC}s AV .mp4 files found under {DATA_BASE}"
sample_size = min(50, len(all_segs))
random.seed(42)
sample_paths = random.sample(all_segs, sample_size)
print(f"Sampling {sample_size} / {len(all_segs)} {_dur_int}s segments for layer selection.")
print(f"Probing layers: {QUARTER_LAYERS}")

hidden_embeddings = {idx: [] for idx in QUARTER_LAYERS}
sheet_embeddings  = {idx: [] for idx in QUARTER_LAYERS}
failed_sample = []

for vp in tqdm(sample_paths, desc="Layer selection sample"):
    try:
        hs, sheet, av_mask, a_mask, v_mask = extract_av_embeddings(vp)
        for idx in QUARTER_LAYERS:
            e_a = get_mean_pool_embedding(hs, a_mask, idx)
            e_v = get_mean_pool_embedding(hs, v_mask, idx)
            hidden_embeddings[idx].append((e_a + e_v) / 2.0)

            s_a = get_mean_pool_embedding(sheet, a_mask, idx)
            s_v = get_mean_pool_embedding(sheet, v_mask, idx)
            sheet_embeddings[idx].append((s_a + s_v) / 2.0)
        del hs, sheet
        torch.cuda.empty_cache()
    except Exception as e:
        failed_sample.append((vp.name, repr(e)))

if failed_sample:
    print(f"\n{len(failed_sample)} segments failed during sampling:")
    for name, err in failed_sample:
        print(f"  {name}: {err}")


def _compute_rdms(embeddings_by_layer):
    rdms = {}
    for idx, embs_list in embeddings_by_layer.items():
        if len(embs_list) == 0:
            print(f"Layer {idx}: no embeddings collected -- skipping RDM.")
            continue
        embs = np.array(embs_list)
        rdms[idx] = squareform(pdist(embs, metric="cosine"))
    return rdms


def _print_spearman_table(rdms, label):
    if N_LAYERS - 1 not in rdms:
        print(f"[{label}] last-layer RDM unavailable -- skipping Spearman table.")
        return
    last_layer_rdm = rdms[N_LAYERS - 1].flatten()
    print(f"\nSpearman r vs last-layer RDM ({label}):")
    for idx in QUARTER_LAYERS:
        if idx not in rdms:
            print(f"  Layer {idx:3d}: skipped (no embeddings)")
            continue
        r, _ = spearmanr(rdms[idx].flatten(), last_layer_rdm)
        tag = " (last)" if idx == N_LAYERS - 1 else ""
        print(f"  Layer {idx:3d}: r = {r:.3f}{tag}")


hidden_rdms = _compute_rdms(hidden_embeddings)
sheet_rdms  = _compute_rdms(sheet_embeddings)
_print_spearman_table(hidden_rdms, "hidden / X_hat")
_print_spearman_table(sheet_rdms,  "sheet / Z")

# ── Dynamic RDM grid (hidden variant; primary omni3b-comparable signal) ───────
valid_layers = [idx for idx in QUARTER_LAYERS if idx in hidden_rdms]
n_show = len(valid_layers)
if n_show > 0:
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_visible(False)
    for ax, idx in zip(axes_flat, valid_layers):
        ax.set_visible(True)
        im = ax.imshow(hidden_rdms[idx], vmin=0, vmax=1, cmap="hot_r")
        ax.set_title(f"Layer {idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{MODEL_TAG} (hidden/X_hat) -- layer RDMs (N={sample_size} segments)")
    plt.tight_layout()
    out_png = DIAGNOSTICS_DIR / f"{MODEL_TAG}_layer_rdms_bin{_dur_int}s.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved layer RDM grid: {out_png}")

# ── Save sample embeddings (both variants) ────────────────────────────────────
for idx in QUARTER_LAYERS:
    if hidden_embeddings[idx]:
        arr = np.array(hidden_embeddings[idx])
        layer_dir = EMBEDDINGS_BASE / _layer_tag(idx)
        layer_dir.mkdir(parents=True, exist_ok=True)
        out_path = layer_dir / f"{_layer_tag(idx)}_sample_av_{_dur_int}s.npy"
        np.save(out_path, arr)
        print(f"Saved {out_path.name}  shape={arr.shape}")
    if sheet_embeddings[idx]:
        arr = np.array(sheet_embeddings[idx])
        layer_dir = EMBEDDINGS_BASE / _sheet_tag(idx)
        layer_dir.mkdir(parents=True, exist_ok=True)
        out_path = layer_dir / f"{_sheet_tag(idx)}_sample_av_{_dur_int}s.npy"
        np.save(out_path, arr)
        print(f"Saved {out_path.name}  shape={arr.shape}")


# ── FULL EXTRACTION LOOP ──────────────────────────────────────────────────────
# Loops over SEGMENT_DURATIONS and extracts quarter-layer mean-pool embeddings
# for three modality masks (_av/_a/_v) x two representation variants per segment:
#   {MODEL_TAG}_layer{idx}            -- hidden / X_hat (comparable to omni3b)
#   {MODEL_TAG}_layer{idx}_sheet      -- cortical sheet / Z (the topographic code)
#
# Output structure (matches the MODEL_NAME:modalities convention in analysis.sh):
#   EMBEDDINGS_BASE/{MODEL_TAG}_layer{idx}/{out_tag}/{MODEL_TAG}_layer{idx}_{sfx}.npy
#   EMBEDDINGS_BASE/{MODEL_TAG}_layer{idx}_sheet/{out_tag}/{MODEL_TAG}_layer{idx}_sheet_{sfx}.npy
#
# Skip logic: if all outputs exist for a duration, skip it.

for bin_sec, skip_sec, segments_dir in SEGMENT_DURATIONS:
    dur_int  = int(bin_sec)
    skip_int = int(skip_sec)
    chunk_suffix = (f"_av_chunks_{dur_int}s" if skip_int == dur_int
                    else f"_av_chunks_{dur_int}s_skip{skip_int}s")
    mp4_files = natsorted(
        list(segments_dir.rglob(f"*{chunk_suffix}/*.mp4")),
        key=lambda p: p.name,
    )
    out_tag = f"bin{dur_int}s_skip{skip_int}s"
    if len(mp4_files) == 0:
        print(f"[{out_tag}] No AV .mp4 files found -- skipping.")
        continue

    expected = [
        EMBEDDINGS_BASE / _layer_tag(idx) / out_tag / f"{_layer_tag(idx)}_{sfx}.npy"
        for idx in QUARTER_LAYERS for sfx in ("av", "a", "v")
    ] + [
        EMBEDDINGS_BASE / _sheet_tag(idx) / out_tag / f"{_sheet_tag(idx)}_{sfx}.npy"
        for idx in QUARTER_LAYERS for sfx in ("av", "a", "v")
    ]
    if all(p.exists() for p in expected):
        print(f"[{out_tag}] All {len(expected)} outputs exist -- skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"{MODEL_TAG} -- {out_tag}  ({len(mp4_files)} segments)")
    print(f"{'='*60}")

    _batch_size = find_safe_batch_size(mp4_files[0])
    print(f"Using batch_size={_batch_size} (one-by-one with safe probe).")

    hidden_av = {idx: [] for idx in QUARTER_LAYERS}
    hidden_a  = {idx: [] for idx in QUARTER_LAYERS}
    hidden_v  = {idx: [] for idx in QUARTER_LAYERS}
    sheet_av  = {idx: [] for idx in QUARTER_LAYERS}
    sheet_a   = {idx: [] for idx in QUARTER_LAYERS}
    sheet_v   = {idx: [] for idx in QUARTER_LAYERS}
    failed    = []

    for vp in tqdm(mp4_files, desc=f"{MODEL_TAG} {out_tag}"):
        try:
            hs, sheet, av_mask, a_mask, v_mask = extract_av_embeddings(vp)
            for idx in QUARTER_LAYERS:
                hidden_av[idx].append((get_mean_pool_embedding(hs, a_mask, idx)
                                        + get_mean_pool_embedding(hs, v_mask, idx)) / 2.0)
                hidden_a[idx].append(get_mean_pool_embedding(hs, a_mask, idx))
                hidden_v[idx].append(get_mean_pool_embedding(hs, v_mask, idx))

                sheet_av[idx].append((get_mean_pool_embedding(sheet, a_mask, idx)
                                       + get_mean_pool_embedding(sheet, v_mask, idx)) / 2.0)
                sheet_a[idx].append(get_mean_pool_embedding(sheet, a_mask, idx))
                sheet_v[idx].append(get_mean_pool_embedding(sheet, v_mask, idx))
            del hs, sheet
            torch.cuda.empty_cache()
        except Exception as e:
            err_str = repr(e)
            failed.append((vp.name, err_str))
            print(f"\n  FAILED {vp.name}: {err_str}")

    for idx in QUARTER_LAYERS:
        layer_dir = EMBEDDINGS_BASE / _layer_tag(idx) / out_tag
        layer_dir.mkdir(parents=True, exist_ok=True)
        for sfx, emb_list in [("av", hidden_av[idx]), ("a", hidden_a[idx]), ("v", hidden_v[idx])]:
            arr = np.array(emb_list)
            out_path = layer_dir / f"{_layer_tag(idx)}_{sfx}.npy"
            np.save(out_path, arr)
            print(f"  saved {out_path.relative_to(EMBEDDINGS_BASE)}  shape={arr.shape}")

        sheet_dir = EMBEDDINGS_BASE / _sheet_tag(idx) / out_tag
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for sfx, emb_list in [("av", sheet_av[idx]), ("a", sheet_a[idx]), ("v", sheet_v[idx])]:
            arr = np.array(emb_list)
            out_path = sheet_dir / f"{_sheet_tag(idx)}_{sfx}.npy"
            np.save(out_path, arr)
            print(f"  saved {out_path.relative_to(EMBEDDINGS_BASE)}  shape={arr.shape}")

    if failed:
        print(f"\n  Failed segments ({len(failed)}):")
        for name, err in failed:
            print(f"    {name}: {err}")
    else:
        print(f"  All {len(mp4_files)} segments extracted successfully.")

    del hidden_av, hidden_a, hidden_v, sheet_av, sheet_a, sheet_v
    gc.collect()
    torch.cuda.empty_cache()

print("\nFull extraction complete.")


# ── SANITY CHECKS ──────────────────────────────────────────────────────────────
# Load the saved last-layer av / a / v embeddings for the primary duration,
# for both the hidden (X_hat) and sheet (Z) variants. RDM comparison saved as PNG.

_out_tag    = f"bin{int(BIN_SEC)}s_skip{int(SKIP_SEC)}s"
_last_layer = N_LAYERS - 1

for variant, dirname, fprefix in [
    ("hidden", f"{MODEL_TAG}_layer{_last_layer}", f"{MODEL_TAG}_layer{_last_layer}"),
    ("sheet",  f"{MODEL_TAG}_layer{_last_layer}_sheet", f"{MODEL_TAG}_layer{_last_layer}_sheet"),
]:
    _layer_dir = EMBEDDINGS_BASE / dirname / _out_tag
    for sfx in ("av", "a", "v"):
        fpath = _layer_dir / f"{fprefix}_{sfx}.npy"
        if not fpath.exists():
            print(f"[{variant}] missing: {fpath}")
            continue
        print(f"\n[{variant}] Loading: {fpath}")
        embs = np.load(fpath)
        print(f"  Shape : {embs.shape}")
        print(f"  Mean  : {embs.mean():.4f}  Std: {embs.std():.4f}  "
              f"Min: {embs.min():.4f}  Max: {embs.max():.4f}")
        if embs.std() < 0.01:
            print(f"  WARNING: {variant}/{sfx} embeddings may be degenerate")

# RDM from first 100 segments of the av embeddings, both variants side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (variant, dirname, fprefix) in zip(axes, [
    ("hidden", f"{MODEL_TAG}_layer{_last_layer}", f"{MODEL_TAG}_layer{_last_layer}"),
    ("sheet",  f"{MODEL_TAG}_layer{_last_layer}_sheet", f"{MODEL_TAG}_layer{_last_layer}_sheet"),
]):
    av_path = EMBEDDINGS_BASE / dirname / _out_tag / f"{fprefix}_av.npy"
    if not av_path.exists():
        ax.set_visible(False)
        continue
    embs_av = np.load(av_path)
    n_check = min(100, len(embs_av))
    rdm_check = squareform(pdist(embs_av[:n_check], metric="cosine"))
    im = ax.imshow(rdm_check, vmin=0, vmax=1, cmap="hot_r")
    ax.set_title(f"{MODEL_TAG} layer{_last_layer} ({variant}) -- {_out_tag} AV RDM (N={n_check})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
out_png = DIAGNOSTICS_DIR / f"{MODEL_TAG}_sanity_rdm_{_out_tag}.png"
plt.savefig(out_png, dpi=120)
plt.close(fig)
print(f"Saved sanity RDM: {out_png}")

print("\nSanity check complete.")
