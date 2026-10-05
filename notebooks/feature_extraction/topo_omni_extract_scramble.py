"""
notebooks/feature_extraction/topo_omni_extract_scramble.py
=============================================================
Topo-Omni analogue of omni3b_extract_scramble.py -- Move 3 (temporal-scramble
binding control) for topoomni_layer{9,18,27}. Extracts BOTH the hidden
(X_hat) stream, used as a whole-embedding diff-study input, AND the cortical
sheet (Z), used as the scored/sheet model in the scramble-design in-silico
localizer (rsa/topoomni_av_separability_localizer.py --design scramble).

Unimodal "_a"/"_v" (hidden) do NOT need re-extraction: since
topo_omni_extract_intact.py they come from genuinely separate audio-only /
video-only forward passes and cannot depend on which audio was paired with
which video. Only the JOINT forward pass depends on pairing, so this script
re-runs only that pass with video[i] paired against a RANDOMLY PERMUTED
audio[perm(i)] (fixed seed 42, same convention as the other scramble
scripts), saving:
  topoomni_layer{N}_mp_avscramble_av.npy                -- hidden masked-pool (a+v)/2, scrambled pairing
  topoomni_layer{N}_lt_avscramble_av.npy                -- hidden last-token, scrambled pairing
  topoomni_layer{N}_sheet_mp_avscramble_av.npy          -- sheet masked-pool (a+v)/2, scrambled pairing
  topoomni_layer{N}_sheet_lt_avscramble_av.npy          -- sheet last-token, scrambled pairing

Run with:
    conda run --no-capture-output -n topo_omni \
        python "notebooks/feature_extraction/topo_omni_extract_scramble.py"
"""

import gc
import os

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from natsort import natsorted
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT, DATA, OUTPUTS  # noqa: E402

TOPO_OMNI_REPO = ROOT / "Model Repos/topo-omni"
assert TOPO_OMNI_REPO.is_dir(), f"topo-omni repo not found at {TOPO_OMNI_REPO}"
sys.path.insert(0, str(TOPO_OMNI_REPO))
sys.path.insert(0, str(TOPO_OMNI_REPO / "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, CorticalAdaptor
from qwen_vl_utils import process_vision_info

TOPO_MODEL_ID   = "epfl-neuroai/topo-omni"
DATA_BASE       = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE = OUTPUTS / "model_embeddings"
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
TARGET_LAYERS   = [9, 18, 27]
MODEL_TAG       = "topoomni"
AUDIO_SR        = 16000
SCRAMBLE_SEED   = 42

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def main():
    MODEL_PATH = snapshot_download(TOPO_MODEL_ID, local_files_only=True)
    print(f"Resolved local snapshot dir: {MODEL_PATH}")

    torch.cuda.empty_cache()
    gc.collect()

    print("Loading topo-omni Thinker config ...")
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    model_config.audio_config.is_training = False
    model_config.vision_config.is_training = False
    model_config.text_config.is_training = False
    model_config.apply_spatial_loss = False

    print("Loading topo-omni Thinker weights ...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, config=model_config, device_map={"": DEVICE},
        torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    model.lm_head = torch.nn.Identity()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    _MODEL_DEVICE = next(model.parameters()).device
    print(f"Thinker loaded on {_MODEL_DEVICE}.")

    N_LAYERS = model.config.text_config.num_hidden_layers
    print(f"N_LAYERS={N_LAYERS}  TARGET_LAYERS={TARGET_LAYERS}")

    AUDIO_TOKEN_ID = model.config.audio_token_id
    VIDEO_TOKEN_ID = model.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

    def _find_cortical_adaptors(n_expected):
        direct = getattr(getattr(model, "model", None), "cortical_adaptors", None)
        if direct is not None and len(direct) == n_expected and isinstance(direct[0], CorticalAdaptor):
            return direct
        for _, mod in model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and len(mod) == n_expected
                    and isinstance(mod[0], CorticalAdaptor)):
                return mod
        raise RuntimeError(f"Could not find CorticalAdaptor ModuleList of length {n_expected}")

    cortical_adaptors = _find_cortical_adaptors(N_LAYERS)
    print(f"Cortical adaptors found: {len(cortical_adaptors)}")

    def _load_audio(path, target_sr):
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return np.zeros(target_sr, dtype=np.float32)
            native_sr = stream.sample_rate
            chunks = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray().astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                chunks.append(arr)
        if not chunks:
            return np.zeros(target_sr, dtype=np.float32)
        audio = np.concatenate(chunks)
        if native_sr != target_sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), native_sr, target_sr).numpy()
        return audio

    def _find_audio_path(video_path):
        parts = [p.replace("Video", "Audio") for p in video_path.parts]
        wav_path = Path(*parts).with_suffix(".wav")
        return wav_path if wav_path.exists() else video_path

    def _build_joint_inputs(video_path, audio_path):
        """Joint audio+video message, with an EXPLICIT (possibly mismatched) audio_path."""
        content = [
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text", "text": "Describe what you see and hear."},
        ]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, frames = process_vision_info(msgs)
        audio_array = _load_audio(audio_path, AUDIO_SR)
        return text, frames, audio_array

    def extract_joint(video_path, audio_path, target_layers=TARGET_LAYERS):
        """Joint AV forward pass with an explicit (video, audio) pair.
        Returns (pooled_hidden, lasttoken_hidden, pooled_sheet, lasttoken_sheet)
        dicts keyed by layer idx.

        The HIDDEN (X_hat) pooling below intentionally keeps its original,
        UNTRIMMED audio mask (no n_valid_audio clipping) to leave the
        already-computed/used topoomni_layer{N}_mp_avscramble_av.npy values
        unchanged on rerun. The new SHEET (Z) pooling uses the SAME
        n_valid_audio-trimmed audio mask as every other sheet computation in
        this repo (topo_omni_extract.py, topo_omni_extract_dummy_modality.py),
        since the sheet is compared directly against the real/dummy sheet
        arrays in the localizer and must use a consistent pooling convention."""
        text, frames, audio_array = _build_joint_inputs(video_path, audio_path)
        n_samples = len(audio_array)
        n_mel = int(np.ceil(n_samples / 160))
        n_valid_audio = min(int(np.ceil(n_mel / 2)), 1500)

        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)
        captured_hidden = {}
        captured_sheet = {}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                x_hat, z = out
                captured_hidden[idx] = x_hat.detach().cpu()
                captured_sheet[idx] = z.detach().cpu()
            return _hook

        for i in target_layers:
            handles.append(cortical_adaptors[i].register_forward_hook(_make_hook(i)))
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()

        ids = inputs["input_ids"].squeeze(0).cpu()
        a_mask_full = (ids == AUDIO_TOKEN_ID)
        v_mask = (ids == VIDEO_TOKEN_ID)
        del inputs
        torch.cuda.empty_cache()

        a_positions = a_mask_full.nonzero(as_tuple=True)[0]
        n_clip = min(n_valid_audio, len(a_positions))
        a_mask_trimmed = torch.zeros_like(a_mask_full)
        a_mask_trimmed[a_positions[:n_clip]] = True

        def _pool(hs, mask):
            if mask.any():
                return hs[0, mask, :].mean(dim=0).float().numpy()
            return np.zeros(hs.shape[-1], dtype=np.float32)

        pooled_hidden, lasttoken_hidden = {}, {}
        pooled_sheet, lasttoken_sheet = {}, {}
        for idx in target_layers:
            hs = captured_hidden[idx]  # (1, seq_len, D)
            zs = captured_sheet[idx]
            pooled_hidden[idx] = (_pool(hs, a_mask_full) + _pool(hs, v_mask)) / 2.0
            lasttoken_hidden[idx] = hs[0, -1, :].float().numpy()
            pooled_sheet[idx] = (_pool(zs, a_mask_trimmed) + _pool(zs, v_mask)) / 2.0
            lasttoken_sheet[idx] = zs[0, -1, :].float().numpy()
        return pooled_hidden, lasttoken_hidden, pooled_sheet, lasttoken_sheet

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    audio_paths = [_find_audio_path(vp) for vp in all_segs]
    rng = np.random.default_rng(SCRAMBLE_SEED)
    perm = rng.permutation(len(audio_paths))
    n_fixed = int((perm == np.arange(len(audio_paths))).sum())
    print(f"[SCRAMBLE_AV] Permuted {len(audio_paths)} audio segments "
          f"(seed={SCRAMBLE_SEED}, {n_fixed} incidental self-pairs)")
    audio_paths_scrambled = [audio_paths[i] for i in perm]

    _limit = os.environ.get("TOPOOMNI_SCRAMBLE_LIMIT")
    if _limit:
        n = int(_limit)
        all_segs = all_segs[:n]
        audio_paths_scrambled = audio_paths_scrambled[:n]
        print(f"TOPOOMNI_SCRAMBLE_LIMIT set -- truncated to {n} segments (smoke test).")

    D = 2048
    results_hidden = {idx: [] for idx in TARGET_LAYERS}
    results_hidden_lt = {idx: [] for idx in TARGET_LAYERS}
    results_sheet = {idx: [] for idx in TARGET_LAYERS}
    results_sheet_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp, ap in tqdm(list(zip(all_segs, audio_paths_scrambled)), desc="Scrambled joint extraction"):
        try:
            pooled_hidden, lasttoken_hidden, pooled_sheet, lasttoken_sheet = extract_joint(vp, ap)
            for idx in TARGET_LAYERS:
                results_hidden[idx].append(pooled_hidden[idx])
                results_hidden_lt[idx].append(lasttoken_hidden[idx])
                results_sheet[idx].append(pooled_sheet[idx])
                results_sheet_lt[idx].append(lasttoken_sheet[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_hidden[idx].append(np.zeros(D, dtype=np.float32))
                results_hidden_lt[idx].append(np.zeros(D, dtype=np.float32))
                results_sheet[idx].append(np.zeros(D, dtype=np.float32))
                results_sheet_lt[idx].append(np.zeros(D, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    def _save(name: str, rows: list) -> None:
        out_dir = EMBEDDINGS_BASE / name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array(rows, dtype=np.float32)
        np.save(out_dir / f"{name}_av.npy", arr)
        print(f"[{name}] saved shape={arr.shape} -> {out_dir}")

    for idx in TARGET_LAYERS:
        _save(f"{MODEL_TAG}_layer{idx}_mp_avscramble", results_hidden[idx])
        _save(f"{MODEL_TAG}_layer{idx}_lt_avscramble", results_hidden_lt[idx])
        _save(f"{MODEL_TAG}_layer{idx}_sheet_mp_avscramble", results_sheet[idx])
        _save(f"{MODEL_TAG}_layer{idx}_sheet_lt_avscramble", results_sheet_lt[idx])

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
