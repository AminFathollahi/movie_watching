"""
notebooks/feature_extraction/nemotron_extract_dummy_modality.py
===================================================================
Nemotron analogue of topo_omni_extract_dummy_modality.py / pe_av_extract_
dummy_modality.py -- extracts nvidia/omni-embed-nemotron-3b's own joint
embedding with one real modality and one fixed content-free dummy
substitute, so Nemotron can serve as an independent clustering driver/judge
for the 3-condition (av / clsav_from_a / clsav_from_v) in-silico AV-
integration localizer (rsa/topoomni_av_separability_localizer.py --design
dummy), alongside its existing role as a driver for the scramble design
(nemotron_layer{N}_mp_avscramble.npy, from nemotron_extract_scramble.py).

--dummy-modality a: dummy VIDEO + real AUDIO -> nemotron_layer{N}_mp_clsav_from_a
--dummy-modality v: dummy AUDIO + real VIDEO -> nemotron_layer{N}_mp_clsav_from_v

Layers: 9, 18, 27, 36 (36 = the true final layer / native trained embedding),
matching nemotron_extract_scramble.py's TARGET_LAYERS.

Also saves a last-token variant (nemotron_layer{N}_lt_clsav_from_{a,v}),
matching TopoOmni's own ad-hoc probe pooling (topo-discover/
extract_video_embeddings.py: hidden_states[-1][:, -1, :]) as an alternative
driver to the mean-pool default -- see nemotron_extract_intact.py.

Dummy stimuli (shared across every dummy-modality extraction in this repo):
  data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4
  data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/nemotron_extract_dummy_modality.py --dummy-modality a
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/nemotron_extract_dummy_modality.py --dummy-modality v
"""

import argparse
import gc
import os

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
os.environ["HF_HOME"] = "/home/amin/hf_models"
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
DATA_BASE       = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DUMMY_VIDEO     = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4")
DUMMY_AUDIO     = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
TARGET_LAYERS   = [9, 18, 27, 36]
MODEL_TAG       = "nemotron"
DOC_PREFIX      = "passage: "

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


def find_all_segments(data_base: Path, bin_sec: float, skip_sec: float) -> list[Path]:
    dur_int, skip_int = int(bin_sec), int(skip_sec)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    return natsorted(list(data_base.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)


def _find_audio_path(video_path: Path) -> Path:
    parts = [p.replace("Video", "Audio") for p in video_path.parts]
    wav_path = Path(*parts).with_suffix(".wav")
    return wav_path if wav_path.exists() else video_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dummy-modality", required=True, choices=["a", "v"], dest="dummy_modality",
                   help="'a' = dummy VIDEO + real AUDIO; 'v' = dummy AUDIO + real VIDEO")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test: only process first N segments")
    args = p.parse_args()

    assert DUMMY_VIDEO.exists(), f"missing {DUMMY_VIDEO}"
    assert DUMMY_AUDIO.exists(), f"missing {DUMMY_AUDIO}"

    tag = f"clsav_from_{args.dummy_modality}"
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    out_paths = {idx: EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_mp_{tag}"
                 / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_mp_{tag}_av.npy"
                 for idx in TARGET_LAYERS}
    out_paths.update({
        f"{idx}_lt": EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_lt_{tag}"
        / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_lt_{tag}_av.npy"
        for idx in TARGET_LAYERS
    })
    if not args.force and all(pth.exists() for pth in out_paths.values()):
        print(f"All outputs already exist for tag={tag} -- skipping (use --force to overwrite).")
        return

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
    assert max(TARGET_LAYERS) == N_LAYERS

    def _build_dummy_messages(video_path: Path):
        real_audio_path = _find_audio_path(video_path)
        if args.dummy_modality == "a":
            vid_for_msg, aud_for_msg = DUMMY_VIDEO, real_audio_path  # dummy VIDEO + real AUDIO
        else:
            vid_for_msg, aud_for_msg = video_path, DUMMY_AUDIO       # dummy AUDIO + real VIDEO
        content = [
            {"type": "text", "text": DOC_PREFIX},
            {"type": "video", "video": str(vid_for_msg)},
            {"type": "audio", "audio": str(aud_for_msg)},
        ]
        return [{"role": "user", "content": content}]

    def extract_dummy_av(video_path: Path, target_layers=TARGET_LAYERS):
        messages = _build_dummy_messages(video_path)
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
            pooled_vec = masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            pooled_vec = F.normalize(pooled_vec, dim=-1)
            pooled[idx] = pooled_vec[0].cpu().numpy()
            # Matches TopoOmni's own ad-hoc probe pooling (last real token,
            # L2-normalized) -- see nemotron_extract_intact.py.
            lt = F.normalize(hs[:, -1, :], dim=-1)
            lasttoken[idx] = lt[0].cpu().numpy()

        del batch, out, hidden_states
        torch.cuda.empty_cache()
        return pooled, lasttoken

    all_segs = find_all_segments(DATA_BASE, BIN_SEC, SKIP_SEC)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments. dummy_modality={args.dummy_modality} (tag={tag})")

    if args.limit:
        all_segs = all_segs[: args.limit]
        print(f"--limit set -- truncated to {len(all_segs)} segments (smoke test).")

    D = 2048
    results = {idx: [] for idx in TARGET_LAYERS}
    results_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp in tqdm(all_segs, desc=f"Nemotron dummy-modality ({tag}) extraction"):
        try:
            pooled, lasttoken = extract_dummy_av(vp)
            for idx in TARGET_LAYERS:
                results[idx].append(pooled[idx])
                results_lt[idx].append(lasttoken[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results[idx].append(np.zeros(D, dtype=np.float32))
                results_lt[idx].append(np.zeros(D, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        model_name = f"{MODEL_TAG}_layer{idx}_mp_{tag}"
        out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array(results[idx], dtype=np.float32)
        out_path = out_dir / f"{model_name}_av.npy"
        np.save(out_path, arr)
        print(f"[{model_name}] saved shape={arr.shape} -> {out_path}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt_{tag}"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_lt = np.array(results_lt[idx], dtype=np.float32)
        lt_out_path = lt_out_dir / f"{lt_model_name}_av.npy"
        np.save(lt_out_path, arr_lt)
        print(f"[{lt_model_name}] saved shape={arr_lt.shape} -> {lt_out_path}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
