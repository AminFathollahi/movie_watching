"""
notebooks/feature_extraction/nemotron_extract_scramble.py
==========================================================
Move 3 (temporal-scramble binding control) for nvidia/omni-embed-nemotron-3b
-- companion to omni3b_extract_scramble.py / topo_omni_extract_scramble.py.

Unimodal "_a"/"_v" embeddings do NOT need re-extraction here: since they come
from genuinely separate forward passes with no cross-modal tokens present
(nemotron_extract_intact.py), they cannot depend on which audio was paired
with which video, and are reindexed (not re-inferred) by
build_scramble_unimodal_copies.py. Only the JOINT (audio+video) forward pass
depends on pairing, so this script re-runs only that pass, with video[i]
paired against a randomly permuted audio[perm(i)] (fixed seed 42, identical
convention to every other scramble script in this repo).

Layers: 9, 18, 27, 36 (36 = the true final layer / native trained embedding).

Also saves a last-token variant (nemotron_layer{N}_lt_avscramble),
matching TopoOmni's own ad-hoc probe pooling (topo-discover/
extract_video_embeddings.py: hidden_states[-1][:, -1, :]) as an alternative
driver to the mean-pool default -- see nemotron_extract_intact.py.

Run with:
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/nemotron_extract_scramble.py"
"""

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

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
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
BIN_SEC, SKIP_SEC = 5.0, 5.0
TARGET_LAYERS   = [9, 18, 27, 36]
MODEL_TAG       = "nemotron"
AUDIO_SR        = 16000
DOC_PREFIX      = "passage: "
SCRAMBLE_SEED   = 42

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


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
    assert max(TARGET_LAYERS) == N_LAYERS

    def _find_audio_path(video_path):
        parts = [p.replace("Video", "Audio") for p in video_path.parts]
        wav_path = Path(*parts).with_suffix(".wav")
        return wav_path if wav_path.exists() else video_path

    def _build_joint_messages(video_path, audio_path):
        content = [
            {"type": "text", "text": DOC_PREFIX},
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(audio_path)},
        ]
        return [{"role": "user", "content": content}]

    def extract_joint(video_path, audio_path, target_layers=TARGET_LAYERS):
        messages = _build_joint_messages(video_path, audio_path)
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
            # Matches TopoOmni's own ad-hoc probe pooling (last real token,
            # L2-normalized) -- see nemotron_extract_intact.py.
            lt = F.normalize(hs[:, -1, :], dim=-1)
            lasttoken[idx] = lt[0].cpu().numpy()

        del batch, out, hidden_states
        torch.cuda.empty_cache()
        return pooled, lasttoken

    # ── Segment list + scrambled pairing (identical convention to omni3b_extract_scramble.py) ──
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

    _limit = os.environ.get("NEMOTRON_SCRAMBLE_LIMIT")
    if _limit:
        n = int(_limit)
        all_segs = all_segs[:n]
        audio_paths_scrambled = audio_paths_scrambled[:n]
        print(f"NEMOTRON_SCRAMBLE_LIMIT set -- truncated to {n} segments (smoke test).")

    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_av_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp, ap in tqdm(list(zip(all_segs, audio_paths_scrambled)), desc="nemotron scrambled joint extraction"):
        try:
            pooled_av, lt_av = extract_joint(vp, ap)
            for idx in TARGET_LAYERS:
                results_av[idx].append(pooled_av[idx])
                results_av_lt[idx].append(lt_av[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_av[idx].append(np.zeros(2048, dtype=np.float32))
                results_av_lt[idx].append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        av_model_name = f"{MODEL_TAG}_layer{idx}_mp_avscramble"
        av_out_dir = EMBEDDINGS_BASE / av_model_name / f"bin{dur_int}s_skip{skip_int}s"
        av_out_dir.mkdir(parents=True, exist_ok=True)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        np.save(av_out_dir / f"{av_model_name}_av.npy", arr_av)
        print(f"[{av_model_name}] saved scrambled av={arr_av.shape} -> {av_out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt_avscramble"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_av_lt = np.array(results_av_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_av_lt)
        print(f"[{lt_model_name}] saved scrambled lasttoken av={arr_av_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
