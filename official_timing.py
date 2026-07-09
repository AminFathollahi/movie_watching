"""
official_timing.py
===================
Parser for data/HCP_7T_Movie_Clip_Timing.csv — the official HCP 7T movie
stimulus timing sheet (one section per run, `\\r`-terminated rows, block
label "0" = 20s inter-clip REST block, "N.M" = clip M of run N).

Times in the file are RUN-LOCAL seconds. This module converts them to
GLOBAL cumulative seconds (matching the convention used throughout this
repo, e.g. data/movie_timing.csv and rsa/shared/rsa_utils.py) using
run-length offsets (default: data/preprocessed/average_sub/raw/
group_average_raw_run_trs.npy, i.e. TR=1s so seconds == TR index).

Relationship to data/movie_timing.csv
--------------------------------------
movie_timing.csv is the FILTERED set of 18 clip windows actually used for
RSA/encoding — it already excludes both the inter-clip rest blocks *and*
some end-of-clip content (e.g. video1: official clip 1.1 runs 20-264.04s
but movie_timing.csv's video1 stops at 232s — the tail ~32s is excluded,
likely end credits). For the other clips the two sources agree to within
~2s. Neither source alone gives "clean rest" vs "meaningful AV content"
without the other:
    rest             = official_timing blocks with label "0"
    filtered stimulus = movie_timing.csv rows (18 clips)
    (the small gaps between the two, e.g. end-credit tails, are neither
     and should be excluded from both windows)

Usage
-----
    from official_timing import load_official_timing, load_run_offsets
    df = load_official_timing("data/HCP_7T_Movie_Clip_Timing.csv")
    offsets = load_run_offsets("data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy")
    df["global_start"] = df["start_local"] + df["run"].map(offsets)
    df["global_end"]   = df["end_local"]   + df["run"].map(offsets)
"""

from pathlib import Path

import numpy as np
import pandas as pd

_MOVIE_RUN = {"7T_MOVIE1": 1, "7T_MOVIE2": 2, "7T_MOVIE3": 3, "7T_MOVIE4": 4}


def load_official_timing(csv_path: str) -> pd.DataFrame:
    """Parse the official HCP 7T movie clip timing sheet.

    Returns a DataFrame with columns:
        run          : int (1-4)
        block_label  : str ("0" for rest, "N.M" for clip M of run N)
        is_rest      : bool
        start_local  : float (run-local seconds)
        end_local    : float (run-local seconds)
        duration     : float (seconds)
    """
    raw = Path(csv_path).read_text()
    rows_txt = raw.splitlines()  # read_text() already normalizes \r -> \n

    rows = []
    current_run = None
    for line in rows_txt:
        line = line.strip()
        if not line:
            continue
        if line.startswith("7T_MOVIE"):
            key = line.split("_v2")[0].split(" ")[0]
            prefix = "_".join(key.split("_")[:2])  # "7T_MOVIE1"
            current_run = _MOVIE_RUN.get(prefix)
            continue
        if line.startswith("movie.block") or line.startswith("block"):
            continue  # header row

        parts = line.split(",")
        if len(parts) < 4:
            continue
        block_label = parts[0].strip()
        try:
            start_local = float(parts[1])
            end_local = float(parts[2])
            duration = float(parts[3])
        except ValueError:
            continue
        if current_run is None:
            continue
        rows.append({
            "run": current_run,
            "block_label": block_label,
            "is_rest": block_label == "0",
            "start_local": start_local,
            "end_local": end_local,
            "duration": duration,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["run", "start_local"]).reset_index(drop=True)


def load_run_offsets(run_trs_npy) -> dict:
    """Return {run_id (1-4): cumulative_offset_seconds} assuming TR=1s.

    run_trs_npy may be a path to a .npy file or an array/sequence of
    per-run TR counts directly (e.g. as returned by preprocess_subject()).
    """
    run_trs = np.load(run_trs_npy) if isinstance(run_trs_npy, (str, Path)) else np.asarray(run_trs_npy)
    offsets = {}
    cum = 0
    for i, n in enumerate(run_trs, start=1):
        offsets[i] = float(cum)
        cum += int(n)
    return offsets


def add_global_time(df: pd.DataFrame, run_trs_npy: str) -> pd.DataFrame:
    """Add global_start / global_end columns (cumulative seconds, TR=1s)."""
    offsets = load_run_offsets(run_trs_npy)
    out = df.copy()
    out["global_start"] = out["start_local"] + out["run"].map(offsets)
    out["global_end"] = out["end_local"] + out["run"].map(offsets)
    return out


if __name__ == "__main__":
    import sys
    _BASE = Path("/home/amin/Research/Representation/Movie")
    df = load_official_timing(str(_BASE / "data/HCP_7T_Movie_Clip_Timing.csv"))
    df = add_global_time(df, str(_BASE / "data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy"))
    print(df.to_string())
    print(f"\nTotal rest blocks: {int(df['is_rest'].sum())}  "
          f"total rest seconds: {df.loc[df['is_rest'], 'duration'].sum():.1f}")
    print(f"Total clip blocks: {int((~df['is_rest']).sum())}  "
          f"total clip seconds: {df.loc[~df['is_rest'], 'duration'].sum():.1f}")
