"""
notebooks/feature_extraction/pe_av_extract_intact.py
=======================================================
Headless PE-AV (facebook/pe-av-small-16-frame) intact extraction with
configurable BIN_SEC/SKIP_SEC, for sliding-window configs (e.g. bin5s_skip1s)
that pe_av_embeddings.ipynb's notebook cells don't cover (they only sweep
skip==bin). Same extract_embeddings() core as pe_av_extract_scramble.py,
minus the scramble path; find_chunk_pairs is skip-aware, matching the
`{stem}_chunks_{dur}s_skip{skip}s` naming produced by segment_bin5s_skip1s.py.

--unimodal own    A = audio_embeds, V = video_embeds, J = audio_video_embeds (one forward
                  pass per window); writes {model}_{a,v,av}.npy.
--unimodal dummy  A = audio_video_embeds of (real audio, blank video), V = audio_video_embeds
                  of (silent audio, real video); writes {model}_dummy_av_{a,v}.npy.

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/pe_av_extract_intact.py [--unimodal {own,dummy}]
"""

import argparse
import gc
import os
import re
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from transformers import AutoModel, AutoProcessor

MODEL_ID    = "facebook/pe-av-small-16-frame"
MODEL_NAME  = "pe-av-small-16-frame"
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS  # noqa: E402

DATA_BASE   = Path(os.environ.get("STIMULUS_DIR", str(DATA / "segmented_stimulus/filtered")))
OUTPUT_BASE = Path(os.environ.get("EMBEDDINGS_BASE", str(OUTPUTS / "model_embeddings")))
BIN_SEC = float(os.environ.get("BIN_SEC", "5.0"))
SKIP_SEC = float(os.environ.get("SKIP_SEC", "1.0"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "8"))
BLANK_PIXEL = -1.0  # processor output for all-black frames: (0 / 255 - 0.5) / 0.5
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
def extract_embeddings(model, processor, pairs, batch_size, device, unimodal) -> dict:
    out = {k: [] for k in (("v", "a", "av") if unimodal == "own" else ("a", "v"))}
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        vid_paths = [str(p[0]) for p in batch]
        aud_paths = [str(p[1]) for p in batch]
        inputs = processor(videos=vid_paths, audio=aud_paths, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if unimodal == "own":
            outputs = model(**inputs)
            out["v"].append(outputs.video_embeds.cpu().numpy())
            out["a"].append(outputs.audio_embeds.cpu().numpy())
            out["av"].append(outputs.audio_video_embeds.cpu().numpy())
        else:
            blank_video = {**inputs, "pixel_values_videos": torch.full_like(inputs["pixel_values_videos"], BLANK_PIXEL)}
            silent_audio = {**inputs, "input_values": torch.zeros_like(inputs["input_values"])}
            out["a"].append(model(**blank_video).audio_video_embeds.cpu().numpy())
            out["v"].append(model(**silent_audio).audio_video_embeds.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(batch)}/{len(pairs)} chunks processed", end="\r")
    print()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimodal", choices=["own", "dummy"], default="own")
    unimodal = parser.parse_args().unimodal
    name = MODEL_NAME if unimodal == "own" else f"{MODEL_NAME}_dummy_av"
    modalities = ["v", "a", "av"] if unimodal == "own" else ["a", "v"]

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    out_dir = OUTPUT_BASE / name / f"bin{dur_int}s_skip{skip_int}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    if all((out_dir / f"{name}_{m}.npy").exists() for m in modalities):
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

    embeds = extract_embeddings(model, processor, pairs, BATCH_SIZE, DEVICE, unimodal)

    for mod, arr in embeds.items():
        out_path = out_dir / f"{name}_{mod}.npy"
        np.save(out_path, arr)
        print(f"  saved {out_path}  shape={arr.shape}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
