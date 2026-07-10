"""
notebooks/feature_extraction/pe_av_extract_dummy_modality.py
================================================================
Extract PE-AV's native joint "audio_video_embeds" ("cls-av") using only ONE
real modality at a time, by feeding a fixed synthetic blank/silent
placeholder for the other modality.

Why this is necessary (not a workaround of convenience): PE-AV's forward()
(transformers.models.pe_audio_video.modeling_pe_audio_video.PeAudioVideoModel)
routes to a COMPLETELY DIFFERENT code path when either pixel_values_videos
or input_values is None -- it returns audio_plus_text_embeds or
video_plus_text_embeds (fused with TEXT, via different heads, different
embedding space) instead of audio_video_embeds. The only way to get
audio_video_embeds -- the model's genuine joint AV fusion head, "cls-av" --
is to pass BOTH input_values and pixel_values_videos through the real
audio_video_encoder. So to ask "what does PE-AV's own AV-fusion head do when
only audio is informative", we pass real audio + a fixed black/silent dummy
video; symmetrically real video + a fixed silent dummy audio for the other
condition. This keeps the SAME embedding space (2048-d? see actual shape
printed at runtime) across all three conditions (real av, dummy-video+real-a,
dummy-audio+real-v) so they are directly comparable as driving embeddings for
the Move-5 stimulus-clustering localizer -- unlike a/v (audio_embeds/
video_embeds), which live in different, unimodal-specific embedding spaces.

Dummy stimuli (generated once, reused for all 626 segments):
  data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4   (1024x720, 24fps, solid black, 5s)
  data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav (44100Hz stereo, all-zero, 5s)
Chosen over a real HCP inter-clip "REST" screen (searched for on disk, not
present as a standalone stimulus file in this project) per analyst decision:
a synthetic dummy is simpler and unambiguously content-free.

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/pe_av_extract_dummy_modality.py --dummy-modality a
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/pe_av_extract_dummy_modality.py --dummy-modality v

--dummy-modality a: dummy VIDEO + real AUDIO  -> saved as pe-av-small-16-frame_clsav_from_a
--dummy-modality v: dummy AUDIO + real VIDEO  -> saved as pe-av-small-16-frame_clsav_from_v
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
DUMMY_VIDEO = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4")
DUMMY_AUDIO = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav")
SEG_SEC     = 5.0
BATCH_SIZE  = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_chunk_pairs(data_base: Path, seg_sec: float) -> list[tuple[Path, Path]]:
    """Identical to pe_av_extract_scramble.py's function -- must produce the
    SAME 626-segment ordering as the existing pe-av-small-16-frame_{a,v,av}.npy
    files for row-alignment with every other model's embeddings."""
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
    p.add_argument("--dummy-modality", required=True, choices=["a", "v"], dest="dummy_modality",
                    help="'a' = dummy VIDEO + real AUDIO; 'v' = dummy AUDIO + real VIDEO")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test: only process first N segments")
    args = p.parse_args()

    assert DUMMY_VIDEO.exists(), f"missing {DUMMY_VIDEO}"
    assert DUMMY_AUDIO.exists(), f"missing {DUMMY_AUDIO}"

    model_out_name = f"{MODEL_NAME}_clsav_from_{args.dummy_modality}"
    out_dir = OUTPUT_BASE / model_out_name / f"bin{int(SEG_SEC)}s_skip{int(SEG_SEC)}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and (out_dir / f"{model_out_name}_av.npy").exists():
        print(f"Output already exists at {out_dir} -- skipping (use --force to overwrite).")
        return

    print(f"Device: {DEVICE}")
    print("Loading pe-av-small-16-frame ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
    model.eval()

    pairs = find_chunk_pairs(DATA_BASE, SEG_SEC)
    print(f"{len(pairs)} real (video, audio) pairs found.")

    if args.dummy_modality == "a":
        # dummy VIDEO + real AUDIO
        pairs = [(DUMMY_VIDEO, real_aud) for (_, real_aud) in pairs]
    else:
        # dummy AUDIO + real VIDEO
        pairs = [(real_vid, DUMMY_AUDIO) for (real_vid, _) in pairs]

    if args.limit:
        pairs = pairs[: args.limit]
        print(f"--limit set -- truncated to {len(pairs)} segments (smoke test).")

    print(f"Extracting cls-av from {len(pairs)} segments (dummy_modality={args.dummy_modality}) ...")
    embeds = extract_embeddings(model, processor, pairs, BATCH_SIZE, DEVICE)

    for mod, arr in embeds.items():
        out_path = out_dir / f"{model_out_name}_{mod}.npy"
        np.save(out_path, arr)
        print(f"  saved {out_path}  shape={arr.shape}")

    # Sanity check: the DUMMY modality's own embedding should be
    # near-constant across all segments (same blank/silent input every time).
    dummy_key = "v" if args.dummy_modality == "a" else "a"
    dummy_arr = embeds[dummy_key]
    row_std = dummy_arr.std(axis=0).mean()
    print(f"  Sanity: dummy-{dummy_key} embedding cross-segment mean std = {row_std:.6f} "
          f"(should be ~0 -- confirms the dummy input is genuinely constant/content-free)")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
