"""
notebooks/feature_extraction/nemotron_extract_intact.py
==========================================================
Extraction for nvidia/omni-embed-nemotron-3b ("NV-QwenOmni-Embed-3B-v1"),
a purpose-built multimodal EMBEDDING model (not a generic chat/generation
model like omni3b/topo-omni). It is built on the same base as omni3b/
topo-omni -- the Qwen2.5-Omni-3B Thinker -- but with (a) the Talker module
removed, (b) the text decoder's self-attention switched from causal to
BIDIRECTIONAL (BidirectQwen2_5OmniThinkerTextModel), and (c) a contrastively
trained mean-pooling + L2-normalize readout at the final decoder layer
(config: 1_Pooling/config.json -- pooling_mode="mean", include_prompt=true).
Per the model card, audio and video are encoded independently (no TMRoPE
interleaving) even when both are present in one input.

This checkpoint has an official joint (av) readout recipe -- mean-pool over
the full sequence (all modality tokens together) of a joint audio+video
forward pass -- so, unlike the (a+v)/2 average of omni3b/topo-omni, "_av" is
not a function of "_a" and "_v": it is a real joint forward pass, pooled over
all tokens (audio + video + text). "_a"/"_v" are separate forward passes (no
cross-modal tokens in the sequence), as in omni3b_extract_intact.py, so the
integration contrast (av regressed on [a, v]) is not tautological.

Layers extracted: 9, 18, 27 (comparable probe depths to omni3b/topo-omni)
plus 36 -- the TRUE FINAL layer (config.text_config.num_hidden_layers = 36),
which is this model's own NATIVE, contrastively-trained embedding output
(hidden_states[-1] in the model card's official usage example). Layer 36 is
the layer used for this model's "native embedding" entries in
shared/model_registry.py; layers 9/18/27 are supplementary depth-sweep
diagnostics only (the checkpoint was not trained for a depth sweep).

A "_lt" (last-token) variant is also produced (nemotron_layer{N}_lt_{a,v,av},
same target layers). TopoOmni's probe-extraction script (topo-discover/
extract_video_embeddings.py: hidden_states[-1][:, -1, :]) uses last-token
pooling at the final layer, not nemotron's official mean-pool recipe, so both
readouts are available as drivers of the in-silico localizer.

Prompted with the model's own "passage: " document prefix (config_sentence_
transformers.json: prompts.document = "passage: "), since we are embedding
stimulus content (documents), not queries.

Run with:
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/nemotron_extract_intact.py"
"""

import argparse
import gc
import json
import os
import shutil

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.paths import DATA, OUTPUTS, MODELS  # noqa: E402

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
os.environ["HF_HOME"] = os.environ.get("MODELS_HOME", str(MODELS))
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
from pe_av_extract_intact import _chunk_suffix, find_chunk_pairs

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH      = "nvidia/omni-embed-nemotron-3b"
DATA_BASE       = Path(os.environ.get("STIMULUS_DIR", str(DATA / "segmented_stimulus/filtered")))
EMBEDDINGS_BASE = Path(os.environ.get("EMBEDDINGS_BASE", str(OUTPUTS / "model_embeddings")))
DEVICE          = os.environ.get("NEMOTRON_DEVICE", "cuda")
DTYPE           = torch.bfloat16
BIN_SEC = float(os.environ.get("BIN_SEC", "2.0"))
SKIP_SEC = float(os.environ.get("SKIP_SEC", str(BIN_SEC)))
TARGET_LAYERS   = [9, 18, 27, 35, 36]
CAT_LAYERS      = [1, 9, 18, 27, 36]
MODEL_TAG       = "nemotron"
AUDIO_SR        = 16000
DOC_PREFIX      = "passage: "

VIDEOS_KWARGS = {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False}
TEXT_KWARGS   = {"truncation": True, "padding": True, "max_length": 204800}
AUDIO_KWARGS  = {"max_length": 2048000}


def token_stats(n):
    return {"min": int(n.min()), "mean": float(n.mean()), "max": int(n.max())}


def save_new(path, array):
    if path.exists():
        old = np.load(path)
        diff = float(np.abs(old - array).max()) if old.shape == array.shape else float("inf")
        print(f"  exists, not written: {path.name} max abs difference {diff:.3g}")
    else:
        np.save(path, array)


def save_mpcat(segs, mean_a, mean_v, halves, dur_int, skip_int):
    """mean_a, mean_v: {layer: (n, D)} unimodal mean-pool vectors; halves: {layer: (audio (n, D), video (n, D))}."""
    bin_dir = f"bin{dur_int}s_skip{skip_int}s"
    counts = np.array(halves["counts"])
    for layer in CAT_LAYERS:
        audio, video = halves[layer]
        scale_a = float(np.sqrt((audio ** 2).sum(axis=1).mean()))
        scale_v = float(np.sqrt((video ** 2).sum(axis=1).mean()))
        source = f"{MODEL_TAG}_layer{layer}_mp"
        src_dir = EMBEDDINGS_BASE / source / bin_dir
        src_dir.mkdir(parents=True, exist_ok=True)
        for mod, array in (("a", mean_a[layer]), ("v", mean_v[layer])):
            save_new(src_dir / f"{source}_{mod}.npy", array.astype(np.float32))
        name = f"{MODEL_TAG}_layer{layer}_mpcat"
        out_dir = EMBEDDINGS_BASE / name / bin_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        joint = np.concatenate([audio / scale_a, video / scale_v], axis=1).astype(np.float32)
        np.save(out_dir / f"{name}_av.npy", joint)
        np.save(out_dir / f"{name}_tokens.npy", counts)
        for mod in "av":
            shutil.copyfile(src_dir / f"{source}_{mod}.npy", out_dir / f"{name}_{mod}.npy")
        info = {"model": MODEL_PATH, "dtype": "bfloat16", "layer": layer, "n_windows": len(segs),
                "pooling": "joint pass; mean over audio tokens | mean over video tokens, each half divided by its RMS norm",
                "scale_audio": scale_a, "scale_video": scale_v,
                "audio_tokens": token_stats(counts[:, 0]), "video_tokens": token_stats(counts[:, 1]),
                "a_v_copied_from": source}
        (out_dir / f"{name}_layers.json").write_text(json.dumps(info, indent=1))
        print(f"[{name}] saved av={joint.shape} -> {out_dir}\n{json.dumps(info)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["mp", "mpcat"], default="mp",
                        help="mp: mean over all tokens (layers 9-36, a/v/av). mpcat: layers 1, 9, 18, 27, 36; "
                             "J = [mean over audio tokens | mean over video tokens] of the joint pass, each half scaled "
                             "to unit RMS norm; also writes the audio-only and video-only mp vectors that are missing.")
    pooling = parser.parse_args().pooling
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
    print(f"N_LAYERS={N_LAYERS}  TARGET_LAYERS={TARGET_LAYERS}")
    assert max(TARGET_LAYERS) == N_LAYERS, f"expected final layer {N_LAYERS}, got {TARGET_LAYERS}"
    tokenizer = processor.tokenizer
    AUDIO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    VIDEO_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|VIDEO|>")

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

    def _build_messages(pair, modality):
        """pair: (muted video mp4, wav); modality: 'audio', 'video', or 'av'."""
        video_path, audio_path = pair
        content = [{"type": "text", "text": DOC_PREFIX}]
        if modality in ("video", "av"):
            content.append({"type": "video", "video": str(video_path)})
        if modality in ("audio", "av"):
            content.append({"type": "audio", "audio": str(audio_path)})
        return [{"role": "user", "content": content}]

    def forward(pair, modality):
        messages = _build_messages(pair, modality)
        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        audio, images, videos = process_mm_info(messages, use_audio_in_video=False)

        kwargs = dict(text=text, images=images, videos=videos, audio=audio,
                      return_tensors="pt", text_kwargs=TEXT_KWARGS,
                      videos_kwargs=VIDEOS_KWARGS, audio_kwargs=AUDIO_KWARGS)
        batch = processor(**kwargs)
        batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True)
        return batch, out

    def extract_cat(pair, layers):
        batch, out = forward(pair, "av")
        ids = batch["input_ids"][0]
        masks = [ids == AUDIO_TOKEN_ID, ids == VIDEO_TOKEN_ID]
        assert all(m.any() for m in masks), f"{pair[0].name}: missing audio or video tokens"
        halves = {layer: [out.hidden_states[layer][0][m].float().mean(dim=0).cpu().numpy() for m in masks]
                  for layer in layers}
        counts = [int(m.sum()) for m in masks]
        del batch, out
        torch.cuda.empty_cache()
        return halves, counts

    def extract(pair, modality, target_layers=TARGET_LAYERS):
        batch, out = forward(pair, modality)
        hidden_states = out.hidden_states  # tuple, len N_LAYERS+1, index i = output of layer i
        attention_mask = batch["attention_mask"]

        pooled, lasttoken = {}, {}
        for idx in target_layers:
            hs = hidden_states[idx].float()  # (1, seq_len, D)
            masked = hs.masked_fill(~attention_mask[..., None].bool(), 0.0)
            p = masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            p = F.normalize(p, dim=-1)
            pooled[idx] = p[0].cpu().numpy()
            # Matches TopoOmni's own ad-hoc probe pooling (last real token,
            # L2-normalized) -- computed from the SAME hidden_states, no extra
            # forward pass.
            lt = F.normalize(hs[:, -1, :], dim=-1)
            lasttoken[idx] = lt[0].cpu().numpy()

        del batch, out, hidden_states
        torch.cuda.empty_cache()
        return pooled, lasttoken

    # ── Segment list: muted video + lossless wav, same windows/order as pe_av_extract_intact.py ──
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    all_segs = find_chunk_pairs(DATA_BASE, BIN_SEC, SKIP_SEC)
    n_videos = sum(
        len(list(d.glob("*_part_*.mp4")))
        for d in DATA_BASE.glob(f"Video*/Video*_chunks_{_chunk_suffix(BIN_SEC, SKIP_SEC)}")
        if "_av_" not in d.name
    )
    assert all_segs, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    assert len(all_segs) == n_videos, f"{n_videos - len(all_segs)} muted clips have no matching wav"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("NEMOTRON_UNIMODAL_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"NEMOTRON_UNIMODAL_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    if pooling == "mpcat":
        mean_a, mean_v = {k: [] for k in CAT_LAYERS}, {k: [] for k in CAT_LAYERS}
        audio, video, counts = {k: [] for k in CAT_LAYERS}, {k: [] for k in CAT_LAYERS}, []
        for pair in tqdm(all_segs, desc="nemotron mpcat extraction"):
            pooled_a, _ = extract(pair, "audio", CAT_LAYERS)
            pooled_v, _ = extract(pair, "video", CAT_LAYERS)
            halves, n = extract_cat(pair, CAT_LAYERS)
            counts.append(n)
            for k in CAT_LAYERS:
                mean_a[k].append(pooled_a[k])
                mean_v[k].append(pooled_v[k])
                audio[k].append(halves[k][0])
                video[k].append(halves[k][1])
        stack = lambda d: {k: np.stack(v) for k, v in d.items()}
        joint = {k: (np.stack(audio[k]), np.stack(video[k])) for k in CAT_LAYERS} | {"counts": counts}
        save_mpcat(all_segs, stack(mean_a), stack(mean_v), joint, dur_int, skip_int)
        return

    results_a  = {idx: [] for idx in TARGET_LAYERS}
    results_v  = {idx: [] for idx in TARGET_LAYERS}
    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_a_lt  = {idx: [] for idx in TARGET_LAYERS}
    results_v_lt  = {idx: [] for idx in TARGET_LAYERS}
    results_av_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for pair in tqdm(all_segs, desc="nemotron a/v/av extraction"):
        try:
            pooled_a,  lt_a  = extract(pair, "audio")
            pooled_v,  lt_v  = extract(pair, "video")
            pooled_av, lt_av = extract(pair, "av")
            for idx in TARGET_LAYERS:
                results_a[idx].append(pooled_a[idx])
                results_v[idx].append(pooled_v[idx])
                results_av[idx].append(pooled_av[idx])
                results_a_lt[idx].append(lt_a[idx])
                results_v_lt[idx].append(lt_v[idx])
                results_av_lt[idx].append(lt_av[idx])
        except Exception as e:
            failed.append((pair[0].name, repr(e)))
            for idx in TARGET_LAYERS:
                results_a[idx].append(np.zeros(2048, dtype=np.float32))
                results_v[idx].append(np.zeros(2048, dtype=np.float32))
                results_av[idx].append(np.zeros(2048, dtype=np.float32))
                results_a_lt[idx].append(np.zeros(2048, dtype=np.float32))
                results_v_lt[idx].append(np.zeros(2048, dtype=np.float32))
                results_av_lt[idx].append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        model_name = f"{MODEL_TAG}_layer{idx}_mp"
        out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr_a  = np.array(results_a[idx], dtype=np.float32)
        arr_v  = np.array(results_v[idx], dtype=np.float32)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        np.save(out_dir / f"{model_name}_a.npy", arr_a)
        np.save(out_dir / f"{model_name}_v.npy", arr_v)
        np.save(out_dir / f"{model_name}_av.npy", arr_av)
        print(f"[{model_name}] saved a={arr_a.shape} v={arr_v.shape} av={arr_av.shape} -> {out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_a_lt  = np.array(results_a_lt[idx], dtype=np.float32)
        arr_v_lt  = np.array(results_v_lt[idx], dtype=np.float32)
        arr_av_lt = np.array(results_av_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_a.npy", arr_a_lt)
        np.save(lt_out_dir / f"{lt_model_name}_v.npy", arr_v_lt)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_av_lt)
        print(f"[{lt_model_name}] saved a={arr_a_lt.shape} v={arr_v_lt.shape} av={arr_av_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
