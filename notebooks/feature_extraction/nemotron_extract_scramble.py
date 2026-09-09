"""Extract Nemotron joint embeddings under controlled audio-video pairings."""

import argparse
import gc
import os

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
# /home/amin/hf_models is a symlink through EXTERNAL_USB, which is not
# attached on this machine; trust_remote_code's dynamic-module cache needs a
# writable HF_HOME regardless of where the model weights resolve from (see
# _resolve_model_path() below), so point it at the mounted ADATA mirror.
os.environ["HF_HOME"] = "/media/amin/ADATA HD710 PRO/hf_models"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from qwen_omni_utils import process_mm_info
from natsort import natsorted

from av_pairing import factorial_contrast, fold_confined_pairing, fold_reference_pairing

# ── Config ────────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/omni-embed-nemotron-3b"
# ~/.cache/huggingface/hub/models--nvidia--omni-embed-nemotron-3b is a dead
# symlink on this machine: HF_HOME above resolves through
# /home/amin/hf_models -> /media/amin/EXTERNAL_USB/..., and that drive is
# not attached. A full mirror of the same snapshot sits on the ADATA HD710
# PRO drive, which IS mounted -- resolve the snapshot dir directly instead
# of trusting HF_HOME resolution (same fix as
# topo_omni_extract_full_sheet.py's _resolve_model_path()).
_HF_SNAPSHOT_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--nvidia--omni-embed-nemotron-3b",
    Path("/media/amin/ADATA HD710 PRO/hf_models/hub/models--nvidia--omni-embed-nemotron-3b"),
]


def _resolve_model_path() -> str:
    for base in _HF_SNAPSHOT_CANDIDATES:
        snap_dir = base / "snapshots"
        if snap_dir.is_dir():
            for child in sorted(snap_dir.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    return str(child)
    return snapshot_download(MODEL_ID, local_files_only=True)


MODEL_PATH      = _resolve_model_path()
DATA_BASE       = Path(os.environ.get(
    "MOVIE_SEGMENTED_DIR",
    "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus/filtered",
))
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
TIMING_CSV      = Path("/home/amin/Research/Representation/Movie/data/movie_timing.csv")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out-run", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--factorial-run", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=SCRAMBLE_SEED)
    parser.add_argument("--timing-csv", type=Path, default=TIMING_CSV)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.held_out_run is not None and args.factorial_run is not None:
        parser.error("--held-out-run and --factorial-run are mutually exclusive")
    if args.factorial_run is not None:
        output_suffix = f"interaction_run{args.factorial_run}_seed{args.seed}"
    elif args.held_out_run is None:
        output_suffix = "avscramble"
    else:
        output_suffix = f"avmismatch_run{args.held_out_run}_seed{args.seed}"
    expected_outputs = []
    for layer in TARGET_LAYERS:
        for pooling in ("mp", "lt"):
            name = f"{MODEL_TAG}_layer{layer}_{pooling}_{output_suffix}"
            root = EMBEDDINGS_BASE / name / f"bin{int(BIN_SEC)}s_skip{int(SKIP_SEC)}s"
            expected_outputs.append(root / f"{name}_av.npy")
            if args.held_out_run is not None or args.factorial_run is not None:
                expected_outputs.append(root / "pairing_manifest.csv")
    if not args.force and all(path.exists() for path in expected_outputs):
        print("All requested outputs already exist; skipping.")
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
    pairing_manifest = None
    reference_indices = None
    if args.factorial_run is not None:
        reference_indices, pairing_manifest = fold_reference_pairing(
            all_segs, pd.read_csv(args.timing_csv), args.factorial_run, args.seed,
        )
        audio_paths_scrambled = [audio_paths[index] for index in reference_indices]
    elif args.held_out_run is not None:
        audio_paths_scrambled, pairing_manifest = fold_confined_pairing(
            all_segs, audio_paths, pd.read_csv(args.timing_csv),
            args.held_out_run, args.seed,
        )
        print(
            f"[AV_MISMATCH] Confined cross-clip pairings to held-out run "
            f"{args.held_out_run} (seed={args.seed})"
        )
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(audio_paths))
        n_fixed = int((perm == np.arange(len(audio_paths))).sum())
        print(f"[SCRAMBLE_AV] Permuted {len(audio_paths)} audio segments "
              f"(seed={args.seed}, {n_fixed} incidental self-pairs)")
        audio_paths_scrambled = [audio_paths[i] for i in perm]

    reference_video_paths = all_segs
    reference_audio_paths = audio_paths
    _limit = os.environ.get("NEMOTRON_SCRAMBLE_LIMIT")
    if _limit:
        n = int(_limit)
        all_segs = all_segs[:n]
        audio_paths_scrambled = audio_paths_scrambled[:n]
        if reference_indices is not None:
            reference_indices = reference_indices[:n]
        if pairing_manifest is not None:
            pairing_manifest = pairing_manifest.iloc[:n].copy()
        print(f"NEMOTRON_SCRAMBLE_LIMIT set -- truncated to {n} segments (smoke test).")

    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_av_lt = {idx: [] for idx in TARGET_LAYERS}
    reverse_av = {idx: [] for idx in TARGET_LAYERS}
    reverse_av_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp, ap in tqdm(list(zip(all_segs, audio_paths_scrambled)), desc="nemotron scrambled joint extraction"):
        try:
            pooled_av, lt_av = extract_joint(vp, ap)
            pooled_reverse = lt_reverse = None
            row_index = len(results_av[TARGET_LAYERS[0]])
            if reference_indices is not None:
                ref = int(reference_indices[row_index])
                pooled_reverse, lt_reverse = extract_joint(
                    reference_video_paths[ref], reference_audio_paths[row_index],
                )
            for idx in TARGET_LAYERS:
                results_av[idx].append(pooled_av[idx])
                results_av_lt[idx].append(lt_av[idx])
                if pooled_reverse is not None and lt_reverse is not None:
                    reverse_av[idx].append(pooled_reverse[idx])
                    reverse_av_lt[idx].append(lt_reverse[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_av[idx].append(np.zeros(2048, dtype=np.float32))
                results_av_lt[idx].append(np.zeros(2048, dtype=np.float32))
                if reference_indices is not None:
                    reverse_av[idx].append(np.zeros(2048, dtype=np.float32))
                    reverse_av_lt[idx].append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        av_model_name = f"{MODEL_TAG}_layer{idx}_mp_{output_suffix}"
        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt_{output_suffix}"
        av_out_dir = EMBEDDINGS_BASE / av_model_name / f"bin{dur_int}s_skip{skip_int}s"
        av_out_dir.mkdir(parents=True, exist_ok=True)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        arr_av_lt = np.array(results_av_lt[idx], dtype=np.float32)
        if reference_indices is not None:
            intact_name = f"{MODEL_TAG}_layer{idx}_mp"
            intact_lt_name = f"{MODEL_TAG}_layer{idx}_lt"
            intact_root = EMBEDDINGS_BASE / intact_name / f"bin{dur_int}s_skip{skip_int}s"
            intact_lt_root = EMBEDDINGS_BASE / intact_lt_name / f"bin{dur_int}s_skip{skip_int}s"
            intact = np.load(intact_root / f"{intact_name}_av.npy")
            intact_lt = np.load(intact_lt_root / f"{intact_lt_name}_av.npy")
            arr_av = factorial_contrast(
                intact, reference_indices, arr_av, np.asarray(reverse_av[idx]),
            )
            arr_av_lt = factorial_contrast(
                intact_lt, reference_indices, arr_av_lt, np.asarray(reverse_av_lt[idx]),
            )
            pairing_manifest["representation_space"] = "l2_normalized_hidden_probe"
        np.save(av_out_dir / f"{av_model_name}_av.npy", arr_av)
        if pairing_manifest is not None:
            pairing_manifest.to_csv(av_out_dir / "pairing_manifest.csv", index=False)
        print(f"[{av_model_name}] saved scrambled av={arr_av.shape} -> {av_out_dir}")

        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_av_lt)
        if pairing_manifest is not None:
            pairing_manifest.to_csv(lt_out_dir / "pairing_manifest.csv", index=False)
        print(f"[{lt_model_name}] saved scrambled lasttoken av={arr_av_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
