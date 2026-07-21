"""
notebooks/feature_extraction/segment_bin5s_skip1s.py
======================================================
One-off sliding-window segmentation at bin=5s, skip=1s (80% overlap) for all
18 videos. Produces both chunk layouts needed downstream:

  - separate video-only + audio-only chunks (for pe_av_extract_intact.py,
    which pairs them via find_chunk_pairs-style matching):
      Video{N}_chunks_5s_skip1s/Video{N}_part_NNNN.mp4
      Audio{N}_chunks_5s_skip1s/Audio{N}_part_NNNN.wav
  - muxed audio+video chunks (for nemotron/omni3b/topoomni, which read a
    single AV file), reusing the exact ffmpeg recipe from segmentation.ipynb's
    segment_av() cell:
      Video{N}_av_chunks_5s_skip1s/Video{N}_part_NNNN.mp4

Idempotent (skips chunks that already exist). ffmpeg-only, no GPU needed.

Run with:
    python notebooks/feature_extraction/segment_bin5s_skip1s.py
"""

import csv
import math
import subprocess
from pathlib import Path

from natsort import natsorted

DATA_BASE = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
BIN_SEC, SKIP_SEC = 5.0, 1.0


def _probe_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def _num_windows(duration: float) -> int:
    if duration < BIN_SEC:
        return 0
    return int(math.floor((duration - BIN_SEC) / SKIP_SEC)) + 1


def segment_pair(idx: int) -> None:
    video_path = DATA_BASE / f"Video{idx}" / f"Video{idx}.mp4"
    audio_path = DATA_BASE / f"Audio{idx}" / f"Audio{idx}.m4a"
    duration = _probe_duration(video_path)
    n = _num_windows(duration)
    pad_len = max(4, len(str(n)))
    print(f"--- Video{idx}/Audio{idx}: {duration:.2f}s -> {n} windows (bin{BIN_SEC}s skip{SKIP_SEC}s) ---")

    vid_out_dir = video_path.parent / f"Video{idx}_chunks_5s_skip1s"
    aud_out_dir = audio_path.parent / f"Audio{idx}_chunks_5s_skip1s"
    av_out_dir = video_path.parent / f"Video{idx}_av_chunks_5s_skip1s"
    for d in (vid_out_dir, aud_out_dir, av_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    timing_rows = []
    for i in range(n):
        start_t = i * SKIP_SEC
        end_t = start_t + BIN_SEC
        timing_rows.append({"part": i + 1, "start_sec": start_t, "end_sec": end_t})
        part = f"{i + 1:0{pad_len}d}"

        vid_chunk = vid_out_dir / f"Video{idx}_part_{part}.mp4"
        if not vid_chunk.exists():
            cmd = ["ffmpeg", "-y", "-ss", str(start_t), "-t", str(BIN_SEC), "-i", str(video_path),
                   "-c:v", "libx264", "-preset", "ultrafast", "-an", str(vid_chunk)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  video ffmpeg error part {part}: {r.stderr[-300:]}")

        aud_chunk = aud_out_dir / f"Audio{idx}_part_{part}.wav"
        if not aud_chunk.exists():
            cmd = ["ffmpeg", "-y", "-ss", str(start_t), "-t", str(BIN_SEC), "-i", str(audio_path),
                   "-c:a", "pcm_s16le", "-vn", str(aud_chunk)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  audio ffmpeg error part {part}: {r.stderr[-300:]}")

        av_chunk = av_out_dir / f"Video{idx}_part_{part}.mp4"
        if not av_chunk.exists():
            cmd = ["ffmpeg", "-y",
                   "-ss", str(start_t), "-t", str(BIN_SEC), "-i", str(video_path),
                   "-ss", str(start_t), "-t", str(BIN_SEC), "-i", str(audio_path),
                   "-c:v", "libx264", "-preset", "ultrafast",
                   "-c:a", "aac", "-ar", "44100", "-ac", "2",
                   "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(av_chunk)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  av ffmpeg error part {part}: {r.stderr[-300:]}")

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", end="\r")
    print()

    for out_dir in (vid_out_dir, aud_out_dir, av_out_dir):
        with open(out_dir / "timing.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["part", "start_sec", "end_sec"])
            writer.writeheader()
            writer.writerows(timing_rows)


def main():
    video_dirs = natsorted(DATA_BASE.glob("Video*"))
    indices = []
    for d in video_dirs:
        try:
            indices.append(int(d.name.replace("Video", "")))
        except ValueError:
            continue
    print(f"Found {len(indices)} videos: {indices}")
    for idx in indices:
        segment_pair(idx)
    print("Done.")


if __name__ == "__main__":
    main()
