"""Extract PE-AV joint embeddings under controlled audio-video pairings."""

import argparse
import gc
import os
import re
from pathlib import Path

# Must run before any transformers/huggingface_hub import -- HF_HOME is read
# at import time, so setting it later has no effect on cache resolution.
os.environ["HF_HOME"] = "/media/amin/ADATA HD710 PRO/hf_models"

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from transformers import AutoModel, AutoProcessor

from av_pairing import factorial_contrast, fold_confined_pairing, fold_reference_pairing

MODEL_ID    = "facebook/pe-av-small-16-frame"
MODEL_NAME  = "pe-av-small-16-frame"
DATA_BASE   = Path(os.environ.get(
    "MOVIE_SEGMENTED_DIR",
    "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus/filtered",
))
OUTPUT_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
TIMING_CSV  = Path("/home/amin/Research/Representation/Movie/data/movie_timing.csv")
SEG_SEC     = 5.0
BATCH_SIZE  = 8
SCRAMBLE_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_chunk_pairs(data_base: Path, seg_sec: float) -> list[tuple[Path, Path]]:
    pairs = []
    for vid_subdir in natsorted(data_base.glob("Video*")):
        m = re.search(r"Video(\d+)$", vid_subdir.name)
        if not m:
            continue
        idx = m.group(1)
        chunk_dir = vid_subdir / f"Video{idx}_chunks_{int(seg_sec)}s"
        audio_dir = data_base / f"Audio{idx}" / f"Audio{idx}_chunks_{int(seg_sec)}s"
        if not chunk_dir.exists() or not audio_dir.exists():
            continue
        for vid in natsorted(chunk_dir.glob("*_part_*.mp4")):
            part_num = re.search(r"_part_(\d+)", vid.stem).group(1)
            aud = audio_dir / f"Audio{idx}_part_{part_num}.wav"
            if aud.exists():
                pairs.append((vid, aud))
    return pairs


def scramble_audio(pairs: list[tuple[Path, Path]], seed: int) -> list[tuple[Path, Path]]:
    vids = [p[0] for p in pairs]
    auds = [p[1] for p in pairs]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(auds))
    n_fixed = int((perm == np.arange(len(auds))).sum())
    print(f"  [SCRAMBLE_AV] Permuted {len(auds)} audio segments "
          f"(seed={seed}, {n_fixed} incidental self-pairs)")
    auds_scrambled = [auds[i] for i in perm]
    return list(zip(vids, auds_scrambled))


@torch.no_grad()
def extract_embeddings(model, processor, pairs, batch_size, device) -> dict:
    video_embeds, audio_embeds, av_embeds = [], [], []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        vid_paths = [str(p[0]) for p in batch]
        aud_paths = [str(p[1]) for p in batch]
        inputs = processor(videos=vid_paths, audio=aud_paths, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs)
        video_embeds.append(outputs.video_embeds.cpu().numpy())
        audio_embeds.append(outputs.audio_embeds.cpu().numpy())
        av_embeds.append(outputs.audio_video_embeds.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(batch)}/{len(pairs)} chunks processed", end="\r")
    print()
    return {
        "v":  np.concatenate(video_embeds, axis=0),
        "a":  np.concatenate(audio_embeds, axis=0),
        "av": np.concatenate(av_embeds, axis=0),
    }


def main():
    p = argparse.ArgumentParser()
    pairing = p.add_mutually_exclusive_group()
    pairing.add_argument("--scramble-av", action="store_true", dest="scramble_av")
    pairing.add_argument("--held-out-run", type=int, choices=(1, 2, 3, 4))
    pairing.add_argument("--factorial-run", type=int, choices=(1, 2, 3, 4))
    p.add_argument("--seed", type=int, default=SCRAMBLE_SEED,
                   help="Pairing seed. "
                        "Seed 42 is the canonical control and keeps the legacy output name; "
                        "any other seed is tagged into the output model name.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--timing-csv", type=Path, default=TIMING_CSV)
    args = p.parse_args()

    if args.factorial_run is not None:
        model_out_name = f"{MODEL_NAME}_interaction_run{args.factorial_run}_seed{args.seed}"
    elif args.held_out_run is not None:
        model_out_name = (
            f"{MODEL_NAME}_avmismatch_run{args.held_out_run}_seed{args.seed}"
        )
    elif not args.scramble_av:
        model_out_name = MODEL_NAME
    elif args.seed == SCRAMBLE_SEED:
        model_out_name = f"{MODEL_NAME}_avscramble"
    else:
        model_out_name = f"{MODEL_NAME}_avscramble_seed{args.seed}"
    out_dir = OUTPUT_BASE / model_out_name / f"bin{int(SEG_SEC)}s_skip{int(SEG_SEC)}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    controlled = args.held_out_run is not None or args.factorial_run is not None
    modalities = ["av"] if controlled else ["v", "a", "av"]
    manifest_path = out_dir / "pairing_manifest.csv"
    complete = all((out_dir / f"{model_out_name}_{m}.npy").exists() for m in modalities)
    if args.held_out_run is not None or args.factorial_run is not None:
        complete = complete and manifest_path.exists()
    if not args.force and complete:
        print(f"All outputs already exist at {out_dir} -- skipping (use --force to overwrite).")
        return

    print(f"Device: {DEVICE}")
    print("Loading pe-av-small-16-frame ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
    model.eval()

    pairs = find_chunk_pairs(DATA_BASE, SEG_SEC)
    print(f"{len(pairs)} intact (video, audio) pairs found.")
    if args.factorial_run is not None:
        videos = [pair[0] for pair in pairs]
        audios = [pair[1] for pair in pairs]
        references, manifest = fold_reference_pairing(
            videos, pd.read_csv(args.timing_csv), args.factorial_run, args.seed,
        )
        first_cross = list(zip(videos, [audios[index] for index in references]))
        second_cross = list(zip(
            [videos[index] for index in references], audios,
        ))
        first = extract_embeddings(model, processor, first_cross, BATCH_SIZE, DEVICE)["av"]
        second = extract_embeddings(model, processor, second_cross, BATCH_SIZE, DEVICE)["av"]
        intact_path = (
            OUTPUT_BASE / MODEL_NAME / f"bin{int(SEG_SEC)}s_skip{int(SEG_SEC)}s"
            / f"{MODEL_NAME}_av.npy"
        )
        intact = np.load(intact_path)
        embeds = {"av": factorial_contrast(intact, references, first, second)}
        manifest["representation_space"] = "post_layer_norm_linear_head"
        manifest.to_csv(manifest_path, index=False)
    elif args.held_out_run is not None:
        videos = [pair[0] for pair in pairs]
        audios = [pair[1] for pair in pairs]
        paired_audio, manifest = fold_confined_pairing(
            videos, audios, pd.read_csv(args.timing_csv), args.held_out_run, args.seed,
        )
        pairs = list(zip(videos, paired_audio))
        manifest.to_csv(manifest_path, index=False)
    elif args.scramble_av:
        pairs = scramble_audio(pairs, args.seed)

    if args.factorial_run is None:
        print(f"Extracting from {len(pairs)} pairs (scramble_av={args.scramble_av}) ...")
        embeds = extract_embeddings(model, processor, pairs, BATCH_SIZE, DEVICE)

    for mod in modalities:
        arr = embeds[mod]
        out_path = out_dir / f"{model_out_name}_{mod}.npy"
        np.save(out_path, arr)
        print(f"  saved {out_path}  shape={arr.shape}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
