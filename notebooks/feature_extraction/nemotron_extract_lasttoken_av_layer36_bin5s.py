"""
notebooks/feature_extraction/nemotron_extract_lasttoken_av_layer36_bin5s.py
============================================================================
One-off backfill: nemotron_extract_intact.py's working tree currently runs
at BIN_SEC=SKIP_SEC=2.0 (not 5.0, the granularity every other script and the
localizer/RSA pipeline uses), for reasons unrelated to this backfill. Rather
than touch that file's bin size, this script re-extracts ONLY what the
localizer actually needs from it that is otherwise missing: the intact joint
"av" last-token embedding for nemotron's true final layer (36), at bin5s_
skip5s (626 segments) -- matching TopoOmni's own ad-hoc probe pooling
(hidden_states[-1][:, -1, :]) as an alternative driver to nemotron's official
mean-pool readout (nemotron_layer36_mp_av.npy, already on disk at bin5s_skip5s).

Run with:
    conda run --no-capture-output -n avtransformer \\
        python notebooks/feature_extraction/nemotron_extract_lasttoken_av_layer36_bin5s.py
"""

import gc
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS, MODELS  # noqa: E402

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
os.environ["HF_HOME"] = str(MODELS)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info
from natsort import natsorted

MODEL_PATH      = "nvidia/omni-embed-nemotron-3b"
DATA_BASE       = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE = OUTPUTS / "model_embeddings"
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
LAYER           = 36
MODEL_TAG       = "nemotron"
DOC_PREFIX      = "passage: "

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


def _find_audio_path(video_path):
    parts = [p.replace("Video", "Audio") for p in video_path.parts]
    wav_path = Path(*parts).with_suffix(".wav")
    return wav_path if wav_path.exists() else video_path


def main():
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    model_name = f"{MODEL_TAG}_layer{LAYER}_lt"
    out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
    out_path = out_dir / f"{model_name}_av.npy"

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

    N_LAYERS = model.config.text_config.num_hidden_layers
    assert LAYER == N_LAYERS, f"expected final layer {N_LAYERS}, got {LAYER}"

    def extract_av_lasttoken(video_path):
        content = [
            {"type": "text", "text": DOC_PREFIX},
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(_find_audio_path(video_path))},
        ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(messages, use_audio_in_video=False)

        batch = processor(text=text, images=images, videos=videos, audio=audio,
                           return_tensors="pt", text_kwargs=TEXT_KWARGS,
                           videos_kwargs=VIDEOS_KWARGS, audio_kwargs=AUDIO_KWARGS)
        batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states[LAYER].float()  # (1, seq_len, D)
        # Matches TopoOmni's own ad-hoc probe pooling (last real token, L2-normalized).
        lt = F.normalize(hs[:, -1, :], dim=-1)[0].cpu().numpy()

        del batch, out
        torch.cuda.empty_cache()
        return lt

    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    results, failed = [], []
    for vp in tqdm(all_segs, desc="nemotron layer36 lasttoken av (bin5s)"):
        try:
            results.append(extract_av_lasttoken(vp))
        except Exception as e:
            failed.append((vp.name, repr(e)))
            results.append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(results, dtype=np.float32)
    np.save(out_path, arr)
    print(f"[{model_name}] saved shape={arr.shape} -> {out_path}")
    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
