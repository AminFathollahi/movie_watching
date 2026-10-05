"""
notebooks/feature_extraction/nemotron_extract_lasttoken_av_bin5s_backfill.py
=============================================================================
One-off backfill, sibling of nemotron_extract_lasttoken_av_layer36_bin5s.py:
that script only covered layer 36 (the layer the localizer/diff-study needed
first). Task #10's diff-study searchlight sweep then found layers 9/18/27
lasttoken were also missing at bin5s_skip5s (nemotron_extract_intact.py's
working tree runs at bin2s_skip2s) -- this fills in those 3 remaining layers
in one pass (single forward per video, all 3 layers read from the same
hidden_states tuple, like nemotron_extract_intact.py's extract()).

Run with:
    conda run --no-capture-output -n avtransformer \\
        python notebooks/feature_extraction/nemotron_extract_lasttoken_av_bin5s_backfill.py
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
TARGET_LAYERS   = [9, 18, 27]  # 36 already backfilled separately
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
    assert max(TARGET_LAYERS) < N_LAYERS, f"expected layers < {N_LAYERS}, got {TARGET_LAYERS}"

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
        lt = {}
        for idx in TARGET_LAYERS:
            hs = out.hidden_states[idx].float()  # (1, seq_len, D)
            # Matches TopoOmni's own ad-hoc probe pooling (last real token, L2-normalized).
            lt[idx] = F.normalize(hs[:, -1, :], dim=-1)[0].cpu().numpy()

        del batch, out
        torch.cuda.empty_cache()
        return lt

    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    results = {idx: [] for idx in TARGET_LAYERS}
    failed = []
    for vp in tqdm(all_segs, desc="nemotron layers 9/18/27 lasttoken av (bin5s)"):
        try:
            lt = extract_av_lasttoken(vp)
        except Exception as e:
            failed.append((vp.name, repr(e)))
            lt = {idx: np.zeros(2048, dtype=np.float32) for idx in TARGET_LAYERS}
        for idx in TARGET_LAYERS:
            results[idx].append(lt[idx])

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        model_name = f"{MODEL_TAG}_layer{idx}_lt"
        out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_av.npy"
        arr = np.array(results[idx], dtype=np.float32)
        np.save(out_path, arr)
        print(f"[{model_name}] saved shape={arr.shape} -> {out_path}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
