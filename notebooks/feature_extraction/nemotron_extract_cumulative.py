"""
notebooks/feature_extraction/nemotron_extract_cumulative.py
==============================================================
Progressive/cumulative-context variant of nemotron_extract_intact.py: for
video N's segment i (of the standard non-overlapping bin5s_skip5s grid,
i = 1..floor(duration/5)), the model sees NOT just that 5s slice but
everything from the start of the video through the end of segment i
(seconds [0, 5*i)) -- i.e. segment 1 gets 5s of context, segment 2 gets 10s,
..., the last segment of a video gets the (nearly) full video. Segment
COUNT and ordering are identical to bin5s_skip5s (626 total across 18
videos) so this is a drop-in comparison against the standard intact
extraction in RSA -- only the INPUT CONTENT per segment differs.

Cumulative clips are built with ffmpeg stream copy (-c copy, always
starting at t=0 so no keyframe-seek inaccuracy) into a temp file, extracted,
then deleted -- avoids storing ~18 videos worth of quadratic-in-segment-count
duplicated media on disk.

GPU feasibility (see conversation / fork investigation, empirically tested
on an RTX 5070 Ti Laptop, 12GB):
  - Model weights alone (bf16) = 9.53GB. Headroom for activations on a 12GB
    card runs out around 60-70s of cumulative context (with
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) -- NOT enough for the
    longest videos here (up to 258s / 51 segments).
  - Per explicit user decision: do NOT quantize the model to work around
    this. This script is written for a >=24GB GPU (lab 4090), where
    weights (9.53GB) + activations for the longest cumulative clip
    (~258s, extrapolated ~10GB) fit comfortably (~20GB of 24GB).
  - Per-segment OOM is still caught and the segment is zero-filled (same
    convention as nemotron_extract_intact.py's `failed` list) so a run on a
    smaller GPU degrades gracefully instead of crashing outright.

Smoke-test on a small GPU: set NEMOTRON_CUMULATIVE_LIMIT=N to only process
the first N cumulative segments (short-duration ones), e.g. to sanity-check
correctness locally before the full run on the 4090.

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/nemotron_extract_cumulative.py
"""

import gc
import os
import subprocess
import tempfile

os.environ["HF_HOME"] = "/home/amin/hf_models"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info
from natsort import natsorted

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH      = "nvidia/omni-embed-nemotron-3b"
DATA_BASE       = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC         = 5.0   # segment grid (same as bin5s_skip5s); cumulative duration for
                         # segment i is BIN_SEC * i, NOT BIN_SEC itself.
TARGET_LAYERS   = [9, 18, 27, 36]
MODEL_TAG       = "nemotron"
DOC_PREFIX      = "passage: "

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


def _video_indices() -> list[int]:
    indices = []
    for d in natsorted(DATA_BASE.glob("Video*")):
        try:
            indices.append(int(d.name.replace("Video", "")))
        except ValueError:
            continue
    return indices


def _probe_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def _build_cumulative_segments() -> list[tuple[int, int, float, Path, Path]]:
    """Returns (video_idx, segment_i, cumulative_duration, video_path, audio_path)
    for every segment on the standard bin5s_skip5s grid, in the same
    video-major, segment-minor order as nemotron_extract_intact.py's
    natsorted glob (so output arrays line up with existing bin5s_skip5s
    fMRI timing)."""
    segments = []
    for idx in _video_indices():
        video_path = DATA_BASE / f"Video{idx}" / f"Video{idx}.mp4"
        audio_path = DATA_BASE / f"Audio{idx}" / f"Audio{idx}.m4a"
        duration = _probe_duration(video_path)
        n_segments = int(math.floor(duration / BIN_SEC))
        for i in range(1, n_segments + 1):
            cum_dur = min(BIN_SEC * i, duration)
            segments.append((idx, i, cum_dur, video_path, audio_path))
    return segments


def _slice_cumulative(video_path: Path, audio_path: Path, cum_dur: float, out_path: Path) -> None:
    cmd = ["ffmpeg", "-y",
           "-t", str(cum_dur), "-i", str(video_path),
           "-t", str(cum_dur), "-i", str(audio_path),
           "-c:v", "copy", "-c:a", "copy",
           "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


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

    N_LAYERS = model.config.text_config.num_hidden_layers
    assert max(TARGET_LAYERS) == N_LAYERS, f"expected final layer {N_LAYERS}, got {TARGET_LAYERS}"

    def _build_messages(clip_path, modality):
        content = [{"type": "text", "text": DOC_PREFIX}]
        if modality in ("video", "av"):
            content.append({"type": "video", "video": str(clip_path)})
        if modality in ("audio", "av"):
            content.append({"type": "audio", "audio": str(clip_path)})
        return [{"role": "user", "content": content}]

    def extract(clip_path, modality, target_layers=TARGET_LAYERS):
        messages = _build_messages(clip_path, modality)
        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(messages, use_audio_in_video=False)

        kwargs = dict(text=text, images=images, videos=videos, audio=audio,
                      return_tensors="pt", text_kwargs=TEXT_KWARGS,
                      videos_kwargs=VIDEOS_KWARGS, audio_kwargs=AUDIO_KWARGS)
        batch = processor(**kwargs)
        batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True)
        hidden_states = out.hidden_states
        attention_mask = batch["attention_mask"]

        pooled, lasttoken = {}, {}
        for idx in target_layers:
            hs = hidden_states[idx].float()
            masked = hs.masked_fill(~attention_mask[..., None].bool(), 0.0)
            p = masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            p = F.normalize(p, dim=-1)
            pooled[idx] = p[0].cpu().numpy()
            lt = F.normalize(hs[:, -1, :], dim=-1)
            lasttoken[idx] = lt[0].cpu().numpy()

        del batch, out, hidden_states
        torch.cuda.empty_cache()
        return pooled, lasttoken

    segments = _build_cumulative_segments()
    assert len(segments) > 0, f"No videos found under {DATA_BASE}"
    print(f"Built {len(segments)} cumulative segments across {len(_video_indices())} videos.")

    _limit = os.environ.get("NEMOTRON_CUMULATIVE_LIMIT")
    if _limit:
        segments = segments[: int(_limit)]
        print(f"NEMOTRON_CUMULATIVE_LIMIT set -- truncated to {len(segments)} segments (smoke test).")

    results_a  = {idx: [] for idx in TARGET_LAYERS}
    results_v  = {idx: [] for idx in TARGET_LAYERS}
    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_a_lt  = {idx: [] for idx in TARGET_LAYERS}
    results_v_lt  = {idx: [] for idx in TARGET_LAYERS}
    results_av_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="nemotron_cumulative_"))
    try:
        for vidx, i, cum_dur, video_path, audio_path in tqdm(segments, desc="nemotron cumulative a/v/av"):
            clip_path = tmp_dir / f"Video{vidx}_cum_{i:04d}.mp4"
            tag = f"Video{vidx}_part_{i:04d} (cum={cum_dur:.1f}s)"
            try:
                _slice_cumulative(video_path, audio_path, cum_dur, clip_path)
                pooled_a,  lt_a  = extract(clip_path, "audio")
                pooled_v,  lt_v  = extract(clip_path, "video")
                pooled_av, lt_av = extract(clip_path, "av")
                for idx in TARGET_LAYERS:
                    results_a[idx].append(pooled_a[idx])
                    results_v[idx].append(pooled_v[idx])
                    results_av[idx].append(pooled_av[idx])
                    results_a_lt[idx].append(lt_a[idx])
                    results_v_lt[idx].append(lt_v[idx])
                    results_av_lt[idx].append(lt_av[idx])
            except Exception as e:
                failed.append((tag, repr(e)))
                for idx in TARGET_LAYERS:
                    results_a[idx].append(np.zeros(2048, dtype=np.float32))
                    results_v[idx].append(np.zeros(2048, dtype=np.float32))
                    results_av[idx].append(np.zeros(2048, dtype=np.float32))
                    results_a_lt[idx].append(np.zeros(2048, dtype=np.float32))
                    results_v_lt[idx].append(np.zeros(2048, dtype=np.float32))
                    results_av_lt[idx].append(np.zeros(2048, dtype=np.float32))
                torch.cuda.empty_cache()
            finally:
                clip_path.unlink(missing_ok=True)
    finally:
        tmp_dir.rmdir()

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        model_name = f"{MODEL_TAG}_layer{idx}_mp_cumulative"
        out_dir = EMBEDDINGS_BASE / model_name / "bin5s_skip5s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr_a  = np.array(results_a[idx], dtype=np.float32)
        arr_v  = np.array(results_v[idx], dtype=np.float32)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        np.save(out_dir / f"{model_name}_a.npy", arr_a)
        np.save(out_dir / f"{model_name}_v.npy", arr_v)
        np.save(out_dir / f"{model_name}_av.npy", arr_av)
        print(f"[{model_name}] saved a={arr_a.shape} v={arr_v.shape} av={arr_av.shape} -> {out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt_cumulative"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / "bin5s_skip5s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_a_lt  = np.array(results_a_lt[idx], dtype=np.float32)
        arr_v_lt  = np.array(results_v_lt[idx], dtype=np.float32)
        arr_av_lt = np.array(results_av_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_a.npy", arr_a_lt)
        np.save(lt_out_dir / f"{lt_model_name}_v.npy", arr_v_lt)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_av_lt)
        print(f"[{lt_model_name}] saved a={arr_a_lt.shape} v={arr_v_lt.shape} av={arr_av_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(segments)} segments failed.")


if __name__ == "__main__":
    main()
