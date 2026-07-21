"""
notebooks/feature_extraction/pe_av_extract_intact.py
=======================================================
Headless PE-AV (facebook/pe-av-small-16-frame) intact extraction with
configurable BIN_SEC/SKIP_SEC, for sliding-window configs (e.g. bin5s_skip1s)
that pe_av_embeddings.ipynb's notebook cells don't cover (they only sweep
skip==bin). Same extract_embeddings() core as pe_av_extract_scramble.py,
minus the scramble path; find_chunk_pairs is skip-aware, matching the
`{stem}_chunks_{dur}s_skip{skip}s` naming produced by segment_bin5s_skip1s.py.

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/pe_av_extract_intact.py
"""

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
BIN_SEC, SKIP_SEC = 5.0, 1.0
BATCH_SIZE  = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _chunk_suffix(bin_sec: float, skip_sec: float) -> str:
    dur_int, skip_int = int(bin_sec), int(skip_sec)
    return f"{dur_int}s" if skip_int == dur_int else f"{dur_int}s_skip{skip_int}s"


def find_chunk_pairs(data_base: Path, bin_sec: float, skip_sec: float) -> list[tuple[Path, Path]]:
    suffix = _chunk_suffix(bin_sec, skip_sec)
    pairs = []
    for vid_subdir in natsorted(data_base.glob("Video*")):
        m = re.search(r"Video(\d+)$", vid_subdir.name)
        if not m:
            continue
        idx = m.group(1)
        chunk_dir = vid_subdir / f"Video{idx}_chunks_{suffix}"
        audio_dir = data_base / f"Audio{idx}" / f"Audio{idx}_chunks_{suffix}"
        if not chunk_dir.exists() or not audio_dir.exists():
            continue
        for vid in natsorted(chunk_dir.glob("*_part_*.mp4")):
            part_num = re.search(r"_part_(\d+)", vid.stem).group(1)
            aud = audio_dir / f"Audio{idx}_part_{part_num}.wav"
            if aud.exists():
                pairs.append((vid, aud))
    return pairs


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
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    out_dir = OUTPUT_BASE / MODEL_NAME / f"bin{dur_int}s_skip{skip_int}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["v", "a", "av"]
    if all((out_dir / f"{MODEL_NAME}_{m}.npy").exists() for m in modalities):
        print(f"All outputs already exist at {out_dir} -- skipping.")
        return

    print(f"Device: {DEVICE}")
    print("Loading pe-av-small-16-frame ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
    model.eval()

    pairs = find_chunk_pairs(DATA_BASE, BIN_SEC, SKIP_SEC)
    print(f"{len(pairs)} intact (video, audio) pairs found at bin{dur_int}s_skip{skip_int}s.")
    assert len(pairs) > 0, f"No chunk pairs found for bin{dur_int}s_skip{skip_int}s under {DATA_BASE}"

    embeds = extract_embeddings(model, processor, pairs, BATCH_SIZE, DEVICE)

    for mod, arr in embeds.items():
        out_path = out_dir / f"{MODEL_NAME}_{mod}.npy"
        np.save(out_path, arr)
        print(f"  saved {out_path}  shape={arr.shape}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
