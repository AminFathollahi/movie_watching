"""
notebooks/feature_extraction/nemotron_extract_encoder_penultimate.py
=========================================================================
Omni-Embed-Nemotron-3B analogue of omni3b_extract_encoder_penultimate.py --
see that script's docstring for the full rationale. Extracts the
PENULTIMATE-LAYER hidden state of Nemotron's own audio/video ENCODERS
(audio_tower.layers, visual.blocks; 32 layers each, penultimate = index 30),
as opposed to the existing nemotron_layer{9,18,27,36} embeddings, which are
BidirectQwen2_5OmniThinkerTextModel hidden states read out after both towers
are already fused.

Verified empirically: nemotron's audio_tower/visual are the SAME
Qwen2_5OmniAudioEncoder / Qwen2_5OmniVisionEncoder classes used by
omni3b/topoomni (same base architecture per the model card), so the same
hook targets and pooling logic apply -- only the checkpoint weights differ
(separately contrastively trained).

Uses a single joint "av" forward pass (both towers populated at once, no
cross-modal attention between them). Requires HF_HOME pointed at this
checkpoint's actual cache location (see MODEL_PATH note below) -- it is not
under the default cache used by omni3b/topoomni.

Output convention (shared across every nemotron_layer*_mp/_lt readout):
    {embeddings_dir}/nemotron_encoder_penultimate/bin5s_skip5s/nemotron_encoder_penultimate_{a,v}.npy

Run with:
    HF_HOME=<models folder> \
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/nemotron_extract_encoder_penultimate.py"
"""

import gc
import os
from pathlib import Path

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS, MODELS  # noqa: E402

# force-set (not setdefault): the avtransformer conda env's activation script
# exports HF_HOME as the literal unexpanded string "~/.cache/huggingface",
# so setdefault() silently no-ops and the wrong cache dir gets used.
os.environ["HF_HOME"] = str(MODELS)
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import av
import numpy as np
import torch
import torchaudio
from natsort import natsorted
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH        = "nvidia/omni-embed-nemotron-3b"
DATA_BASE         = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE   = OUTPUTS / "model_embeddings"
DEVICE            = "cuda"
DTYPE             = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
PENULTIMATE_IDX   = 30  # 0-indexed; both towers have 32 layers
MODEL_TAG         = "nemotron_encoder_penultimate"
AUDIO_SR          = 16000
FEATURE_DIM       = 1280
DOC_PREFIX        = "passage: "

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


def _pool(hs: torch.Tensor) -> torch.Tensor:
    """Mean-pool a captured hidden state over its sequence dimension, robust
    to either a (seq, dim) or (1, seq, dim) hook output convention."""
    if hs.dim() == 3:
        hs = hs[0]
    return hs.mean(dim=0)


def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading omni-embed-nemotron-3b ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True,
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}.")

    n_audio_layers = len(model.audio_tower.layers)
    n_video_layers = len(model.visual.blocks)
    print(f"audio_tower.layers={n_audio_layers}  visual.blocks={n_video_layers}  "
          f"PENULTIMATE_IDX={PENULTIMATE_IDX}")

    def _find_audio_path(video_path):
        parts = [p.replace("Video", "Audio") for p in video_path.parts]
        wav_path = Path(*parts).with_suffix(".wav")
        return wav_path if wav_path.exists() else video_path

    def _build_messages(video_path):
        content = [
            {"type": "text", "text": DOC_PREFIX},
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(_find_audio_path(video_path))},
        ]
        return [{"role": "user", "content": content}]

    def extract_encoder_penultimate(video_path):
        messages = _build_messages(video_path)
        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(messages, use_audio_in_video=False)

        kwargs = dict(text=text, images=images, videos=videos, audio=audio,
                      return_tensors="pt", text_kwargs=TEXT_KWARGS,
                      videos_kwargs=VIDEOS_KWARGS, audio_kwargs=AUDIO_KWARGS)
        batch = processor(**kwargs)
        batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}

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
                model(**batch, output_hidden_states=False)
        finally:
            h_a.remove()
            h_v.remove()
        del batch
        torch.cuda.empty_cache()

        pooled_a = _pool(captured["a"]).float().cpu().numpy()
        pooled_v = _pool(captured["v"]).float().cpu().numpy()
        return pooled_a, pooled_v

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("NEMOTRON_ENCPEN_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"NEMOTRON_ENCPEN_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

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
