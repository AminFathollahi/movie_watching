"""
notebooks/feature_extraction/pe_av_extract_scramble.py
=========================================================
Move 3 (temporal-scramble binding control) — headless PE-AV extraction with
an A-V temporal scrambling option. Minimal, focused re-implementation of
pe_av_embeddings.ipynb's core extract_embeddings() path (facebook/pe-av-small-16-frame,
5s segments only, _a/_v/_av modalities only -- captions/transcripts/events are
out of scope for this control) so it can run headless with a --scramble-av flag.

Intact mode reproduces the notebook's existing pe-av-small-16-frame_{a,v,av}.npy
byte-for-byte (same pairs, same batch size, same model) -- run it once to verify
before trusting the scrambled output; if intact-mode files already exist on
disk this script will not overwrite them unless --force is passed.

Scrambled mode pairs each video segment with a RANDOMLY PERMUTED audio segment
(default seed 42, same convention as extract_cav_mae_sync.py's --scramble-av) and
saves to outputs/model_embeddings/pe-av-small-16-frame_avscramble/bin5s_skip5s/.

--seed lets this run be repeated with a DIFFERENT random AV pairing, saving to a
seed-tagged model name (pe-av-small-16-frame_avscramble_seed{N}) instead of the
canonical seed-42 output, so it doesn't clobber the intact-vs-scrambled control used
elsewhere (partial_rsa.py, temporal_scramble_binding.py). This is what
rsa/run_av_scramble_permutations.sh uses to build an empirical null of
re-inferred (not RDM-reindexed) AV pairings -- see that script's docstring for why
this can't be done cheaply by permuting an RDM post-hoc (PE-AV's fusion is nonlinear).

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/pe_av_extract_scramble.py --scramble-av [--seed 1]
"""

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from transformers import AutoModel, AutoProcessor

MODEL_ID    = "facebook/pe-av-small-16-frame"
MODEL_NAME  = "pe-av-small-16-frame"
DATA_BASE   = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
OUTPUT_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
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
    p.add_argument("--scramble-av", action="store_true", dest="scramble_av")
    p.add_argument("--seed", type=int, default=SCRAMBLE_SEED,
                   help="AV-pairing permutation seed (only used with --scramble-av). "
                        "Seed 42 is the canonical control and keeps the legacy output name; "
                        "any other seed is tagged into the output model name.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.scramble_av:
        model_out_name = MODEL_NAME
    elif args.seed == SCRAMBLE_SEED:
        model_out_name = f"{MODEL_NAME}_avscramble"
    else:
        model_out_name = f"{MODEL_NAME}_avscramble_seed{args.seed}"
    out_dir = OUTPUT_BASE / model_out_name / f"bin{int(SEG_SEC)}s_skip{int(SEG_SEC)}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["v", "a", "av"]
    if not args.force and all((out_dir / f"{model_out_name}_{m}.npy").exists() for m in modalities):
        print(f"All outputs already exist at {out_dir} -- skipping (use --force to overwrite).")
        return

    print(f"Device: {DEVICE}")
    print("Loading pe-av-small-16-frame ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
    model.eval()

    pairs = find_chunk_pairs(DATA_BASE, SEG_SEC)
    print(f"{len(pairs)} intact (video, audio) pairs found.")
    if args.scramble_av:
        pairs = scramble_audio(pairs, args.seed)

    print(f"Extracting from {len(pairs)} pairs (scramble_av={args.scramble_av}) ...")
    embeds = extract_embeddings(model, processor, pairs, BATCH_SIZE, DEVICE)

    for mod, arr in embeds.items():
        out_path = out_dir / f"{model_out_name}_{mod}.npy"
        np.save(out_path, arr)
        print(f"  saved {out_path}  shape={arr.shape}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
