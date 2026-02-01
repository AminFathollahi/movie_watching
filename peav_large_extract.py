#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from transformers import PeAudioVideoModel, PeAudioVideoProcessor


def extract_part_number(filename: Path):
    match = re.search(r"part[_-]?(\d+)", filename.stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def collect_pairs(base_dir: Path, max_runs: int):
    all_videos = []
    all_audios = []

    for run_idx in range(1, max_runs + 1):
        vid_chunk_dir = base_dir / f"Video{run_idx}" / f"Video{run_idx}_chunks"
        aud_chunk_dir = base_dir / f"Audio{run_idx}" / f"Audio{run_idx}_chunks"

        if not vid_chunk_dir.exists() or not aud_chunk_dir.exists():
            print(f"Warning: Missing dir for run {run_idx}")
            continue

        vids = natsorted(vid_chunk_dir.glob("*.mp4"))
        auds = natsorted(aud_chunk_dir.glob("*.wav"))
        if len(vids) == 0 or len(auds) == 0:
            print(f"Warning: No chunks found for run {run_idx}")
            continue

        vid_parts = {}
        for v in vids:
            part_num = extract_part_number(v)
            if part_num is not None:
                vid_parts[part_num] = v

        aud_parts = {}
        for a in auds:
            part_num = extract_part_number(a)
            if part_num is not None:
                aud_parts[part_num] = a

        common_parts = sorted(set(vid_parts.keys()) & set(aud_parts.keys()))

        for part_num in common_parts:
            all_videos.append(vid_parts[part_num])
            all_audios.append(aud_parts[part_num])

        print(
            f"Run {run_idx}: {len(common_parts)} paired chunks "
            f"(video: {len(vids)}, audio: {len(auds)})"
        )

    paired = list(zip(all_videos, all_audios))
    print(f"\n{'=' * 60}")
    print(f"Total: {len(paired)} paired 2s segments")
    print(f"{'=' * 60}\n")

    if len(paired) == 0:
        raise RuntimeError("No paired segments found.")

    return paired


def main():
    parser = argparse.ArgumentParser(
        description="Extract PE-AV Large embeddings from paired audio/video chunks."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local directory with pe-av-large checkpoints.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Base data directory containing Video*/Audio* subfolders.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Output .pt file path for torch.save.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-runs", type=int, default=18)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=300,
        help="HF_HUB_DOWNLOAD_TIMEOUT seconds (unused when local-only).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force local-only loading for model/processor.",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

    device = torch.device(args.device)
    print(f"device is {device}")

    if args.local_files_only:
        model = PeAudioVideoModel.from_pretrained(
            args.model_dir, local_files_only=True
        ).to(device)
        processor = PeAudioVideoProcessor.from_pretrained(
            args.model_dir, local_files_only=True
        )
    else:
        model = PeAudioVideoModel.from_pretrained(args.model_dir).to(device)
        processor = PeAudioVideoProcessor.from_pretrained(args.model_dir)

    print("Model & Processor Ready")

    paired = collect_pairs(args.data_dir, args.max_runs)

    video_out, audio_out, av_out = [], [], []
    clip_names = []
    run_ids = []

    with torch.inference_mode():
        for i in range(0, len(paired), args.batch_size):
            batch_pairs = paired[i : i + args.batch_size]
            v_batch = [str(v) for v, _ in batch_pairs]
            a_batch = [str(a) for _, a in batch_pairs]

            inputs = processor(videos=v_batch, audio=a_batch, return_tensors="pt").to(
                device
            )

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            video_out.append(outputs.video_embeds.float().cpu())
            audio_out.append(outputs.audio_embeds.float().cpu())
            av_out.append(outputs.audio_video_embeds.float().cpu())

            for v, _ in batch_pairs:
                clip_names.append(v.stem)
                run_ids.append(v.parent.parent.name)

            if (i // args.batch_size) % 50 == 0 or i == 0:
                print(f"Processed {i + len(batch_pairs)}/{len(paired)}")

    video_embeds = torch.cat(video_out, dim=0)
    audio_embeds = torch.cat(audio_out, dim=0)
    av_embeds = torch.cat(av_out, dim=0)

    print("\nDone!")
    print(f"Video embeddings: {video_embeds.shape}")
    print(f"Audio embeddings: {audio_embeds.shape}")
    print(f"AV embeddings:    {av_embeds.shape}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "video": video_embeds,
            "audio": audio_embeds,
            "av": av_embeds,
            "clip_names": clip_names,
            "run_ids": run_ids,
        },
        args.output_path,
    )
    print(f"saved to {args.output_path}")

    npz_path = args.output_path.with_suffix(".npz")
    np.savez(
        npz_path,
        video=video_embeds.numpy(),
        audio=audio_embeds.numpy(),
        av=av_embeds.numpy(),
        clip_names=np.array(clip_names),
        run_ids=np.array(run_ids),
    )
    print(f"also saved to {npz_path}")


if __name__ == "__main__":
    main()
