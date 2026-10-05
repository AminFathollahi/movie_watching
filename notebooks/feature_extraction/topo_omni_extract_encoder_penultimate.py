"""
notebooks/feature_extraction/topo_omni_extract_encoder_penultimate.py
==========================================================================
Topo-Omni analogue of omni3b_extract_encoder_penultimate.py -- see that
script's docstring for the full rationale. Extracts the PENULTIMATE-LAYER
hidden state of Topo-Omni's own audio/video ENCODERS (audio_tower.layers,
visual.blocks; 32 layers each, penultimate = index 30), as opposed to the
existing topoomni_layer{9,18,27}(_sheet) embeddings, which are THINKER
hidden states read out after both towers are already fused.

Loaded via the STOCK transformers Qwen2_5OmniThinkerForConditionalGeneration
class (not Topo-Omni's custom CorticalAdaptor-patched subclass): verified by
inspection that the checkpoint's cortical_adaptors live as separate
ModuleList attributes (audio_tower.cortical_adaptors, visual.cortical_adaptors,
model.cortical_adaptors) parallel to -- not interleaved into --
audio_tower.layers / visual.blocks, so the stock class's forward() computes
those towers identically; it just doesn't know about the extra adaptor
attributes (harmless "unused weights" at load time). No need for the
custom repo's Python path or CorticalAdaptor class here.

Single joint (audio+video) forward pass captures both towers at once (they
don't attend across modalities or into the thinker).

Output convention (shared across every topoomni_layer*_mp/_lt/_sheet readout):
    {embeddings_dir}/topoomni_encoder_penultimate/bin5s_skip5s/topoomni_encoder_penultimate_{a,v}.npy

Run with:
    conda run --no-capture-output -n topo_omni \
        python "notebooks/feature_extraction/topo_omni_extract_encoder_penultimate.py"
"""

import gc
import os
from pathlib import Path

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import av
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from natsort import natsorted
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ── Config ────────────────────────────────────────────────────────────────
TOPO_MODEL_ID     = "epfl-neuroai/topo-omni"
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS  # noqa: E402

DATA_BASE         = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE   = OUTPUTS / "model_embeddings"
DEVICE            = "cuda"
DTYPE             = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
PENULTIMATE_IDX   = 30  # 0-indexed; both towers have 32 layers
MODEL_TAG         = "topoomni_encoder_penultimate"
AUDIO_SR          = 16000
FEATURE_DIM       = 1280

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _pool(hs: torch.Tensor) -> torch.Tensor:
    """Mean-pool a captured hidden state over its sequence dimension, robust
    to either a (seq, dim) or (1, seq, dim) hook output convention."""
    if hs.dim() == 3:
        hs = hs[0]
    return hs.mean(dim=0)


def main():
    MODEL_PATH = snapshot_download(TOPO_MODEL_ID, local_files_only=True)
    print(f"Resolved local snapshot dir: {MODEL_PATH}")

    torch.cuda.empty_cache()
    gc.collect()

    print("Loading topo-omni Thinker (stock transformers class) ...")
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, config=model_config, device_map={"": DEVICE},
        torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    _MODEL_DEVICE = next(model.parameters()).device
    print(f"Thinker loaded on {_MODEL_DEVICE}.")

    n_audio_layers = len(model.audio_tower.layers)
    n_video_layers = len(model.visual.blocks)
    print(f"audio_tower.layers={n_audio_layers}  visual.blocks={n_video_layers}  "
          f"PENULTIMATE_IDX={PENULTIMATE_IDX}")

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

    def _build_joint_inputs(video_path):
        audio_path = _find_audio_path(video_path)
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

    def extract_encoder_penultimate(video_path):
        text, frames, audio_array = _build_joint_inputs(video_path)
        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)

        captured = {}

        def _hook(name):
            def _fn(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                captured[name] = hs.detach()
            return _fn

        h_a = model.audio_tower.layers[PENULTIMATE_IDX].register_forward_hook(_hook("a"))
        h_v = model.visual.blocks[PENULTIMATE_IDX].register_forward_hook(_hook("v"))
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            h_a.remove()
            h_v.remove()
        del inputs
        torch.cuda.empty_cache()

        pooled_a = _pool(captured["a"]).float().cpu().numpy()
        pooled_v = _pool(captured["v"]).float().cpu().numpy()
        return pooled_a, pooled_v

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("TOPOOMNI_ENCPEN_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"TOPOOMNI_ENCPEN_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    results_a, results_v, failed = [], [], []
    for vp in tqdm(all_segs, desc="Encoder-penultimate extraction"):
        try:
            pa, pv = extract_encoder_penultimate(vp)
            results_a.append(pa)
            results_v.append(pv)
        except Exception as e:
            failed.append((vp.name, repr(e)))
            results_a.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            results_v.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    out_dir = EMBEDDINGS_BASE / MODEL_TAG / f"bin{dur_int}s_skip{skip_int}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_a = np.array(results_a, dtype=np.float32)
    arr_v = np.array(results_v, dtype=np.float32)
    np.save(out_dir / f"{MODEL_TAG}_a.npy", arr_a)
    np.save(out_dir / f"{MODEL_TAG}_v.npy", arr_v)
    print(f"[{MODEL_TAG}] saved a={arr_a.shape} v={arr_v.shape} -> {out_dir}")
    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
