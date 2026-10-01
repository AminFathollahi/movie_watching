"""
notebooks/feature_extraction/segment_official.py
==================================================
Cut the 18 movie clips and their 1 s, 2 s and 5 s windows straight from the four
full run movies at the official HCP 7T clip times (data/HCP_7T_Movie_Clip_Timing.csv).
Only the 20 s rest blocks are removed, plus the end credits of video1.

Writes the same layout the extraction scripts already read:
  {OUT}/Video{N}/Video{N}_chunks_{B}s/Video{N}_part_{NNN}.mp4      muted video window
  {OUT}/Audio{N}/Audio{N}_chunks_{B}s/Audio{N}_part_{NNN}.wav      audio window
  {OUT}/Video{N}/Video{N}_av_chunks_{B}s/Video{N}_part_{NNN}.mp4   muxed window (B = 1, 2, 5)
and data/movie_timing.csv (global onset, official duration, run).

Window i of a clip starts at clip_start + i * B; the last partial window is dropped.

Run with:
    python notebooks/feature_extraction/segment_official.py
"""

import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from official_timing import add_global_time, load_official_timing  # noqa: E402

DATA = Path("/home/amin/Research/Representation/Movie/data")
STIMULUS = Path("/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus")
FULL, OUT = STIMULUS / "full", STIMULUS / "filtered"
MOVIES = {1: "7T_MOVIE1_CC1_v2", 2: "7T_MOVIE2_HO1_v2", 3: "7T_MOVIE3_CC2_v2", 4: "7T_MOVIE4_HO2_v2"}
BINS = (1, 2, 5)
AV_BINS = (1, 2, 5)
FRAME = 1 / 24
CREDITS_START = {"video1": 5579 * FRAME}  # first credit frame of clip 1.1 (film ends at frame 5557)


def clip_table() -> pd.DataFrame:
    official = load_official_timing(str(DATA / "HCP_7T_Movie_Clip_Timing.csv"))
    official = add_global_time(official, str(DATA / "preprocessed/average_sub/raw/group_average_raw_run_trs.npy"))
    clips = official[~official["is_rest"]].reset_index(drop=True)
    clips["video_id"] = [f"video{i}" for i in range(1, len(clips) + 1)]
    for column in ("start_local", "duration", "global_start"):
        clips[column] = (clips[column] * 24).round() / 24
    for video_id, credits_start in CREDITS_START.items():
        row = clips["video_id"] == video_id
        clips.loc[row, "duration"] = credits_start - clips.loc[row, "start_local"]
    return clips


def n_windows(duration: float, width: int) -> int:
    return int(math.floor(duration / width + 1e-9))


def commands(clips: pd.DataFrame) -> list[list[str]]:
    jobs = []
    for number, clip in enumerate(clips.itertuples(), start=1):
        movie = FULL / f"{MOVIES[clip.run]}.mp4"
        audio = FULL / f"{MOVIES[clip.run]}_audio.wav"
        for width in BINS:
            for i in range(n_windows(clip.duration, width)):
                start = f"{clip.start_local + i * width:.6f}"
                part = f"{i + 1:03d}"
                video_dir = OUT / f"Video{number}" / f"Video{number}_chunks_{width}s"
                audio_dir = OUT / f"Audio{number}" / f"Audio{number}_chunks_{width}s"
                video_dir.mkdir(parents=True, exist_ok=True)
                audio_dir.mkdir(parents=True, exist_ok=True)
                base = ["ffmpeg", "-v", "error", "-y", "-ss", start, "-t", str(width)]
                jobs.append(base + ["-i", str(movie), "-c:v", "libx264", "-preset", "ultrafast", "-an",
                                    str(video_dir / f"Video{number}_part_{part}.mp4")])
                jobs.append(base + ["-i", str(audio), "-c:a", "pcm_s16le", "-vn",
                                    str(audio_dir / f"Audio{number}_part_{part}.wav")])
                if width in AV_BINS:
                    av_dir = OUT / f"Video{number}" / f"Video{number}_av_chunks_{width}s"
                    av_dir.mkdir(parents=True, exist_ok=True)
                    jobs.append(base + ["-i", str(movie), "-c:v", "libx264", "-preset", "ultrafast",
                                        "-c:a", "aac", "-ar", "44100", "-ac", "2",
                                        str(av_dir / f"Video{number}_part_{part}.mp4")])
    return jobs


def run(job: list[str]) -> None:
    if Path(job[-1]).exists():
        return
    subprocess.run(job, check=True)


def main() -> None:
    clips = clip_table()
    timing = pd.DataFrame({
        "video_id": clips["video_id"],
        "onset_sec": clips["global_start"].round(6),
        "end_sec": (clips["global_start"] + clips["duration"]).round(6),
        "duration_sec": clips["duration"].round(6),
        "run_id": clips["run"],
    })
    timing.to_csv(DATA / "movie_timing.csv", index=False)
    print(timing.to_string(index=False))
    jobs = commands(clips)
    print(f"{len(jobs)} ffmpeg jobs")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, _ in enumerate(pool.map(run, jobs), start=1):
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    counts = {w: sum(n_windows(d, w) for d in clips["duration"]) for w in BINS}
    print("windows per bin:", counts)


if __name__ == "__main__":
    main()
