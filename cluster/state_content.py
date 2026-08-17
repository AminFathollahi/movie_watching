"""
cluster/state_content.py
===========================
Characterizes what each HMM temporal state actually IS in stimulus terms:
which movie video_id(s) and time ranges dominate it. No caption/transcript
text exists on disk for this stimulus set (only embeddings), so "meaning"
here is video identity + timing, not semantic labels — go back to that
video/timestamp range to see the actual content.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_cluster import GROUP_AVG_TRS, get_segment_metadata  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TIMING_CSV = "/home/amin/Research/Representation/Movie/data/movie_timing.csv"


def _merge_contiguous(sorted_starts_ends, gap_tol=0.01):
    """[(start,end), ...] sorted -> merged list of (start,end) contiguous runs."""
    if not sorted_starts_ends:
        return []
    merged = [list(sorted_starts_ends[0])]
    for start, end in sorted_starts_ends[1:]:
        if start <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def summarize_states(state_labels: np.ndarray, meta: pd.DataFrame) -> dict:
    """For each HMM state, which video(s)/time-range(s) it's assigned to."""
    assert len(state_labels) == len(meta), (
        f"state_labels ({len(state_labels)}) and segment metadata ({len(meta)}) length mismatch")
    meta = meta.copy()
    meta["state"] = state_labels

    summary = {}
    for k in sorted(np.unique(state_labels)):
        sub = meta[meta["state"] == k]
        n_seg = len(sub)
        total_duration_sec = n_seg * 5.0  # bin_sec, non-overlapping since skip_sec==bin_sec
        video_counts = sub["video_id"].value_counts().to_dict()
        dominant_video = max(video_counts, key=video_counts.get)

        ranges_by_video = {}
        for vid, vsub in sub.groupby("video_id"):
            pairs = sorted(zip(vsub["stim_start_sec"], vsub["stim_end_sec"]))
            ranges_by_video[vid] = _merge_contiguous(pairs)

        summary[int(k)] = {
            "n_segments": n_seg,
            "total_duration_sec": total_duration_sec,
            "dominant_video": dominant_video,
            "dominant_video_frac": video_counts[dominant_video] / n_seg,
            "video_segment_counts": video_counts,
            "time_ranges_by_video": ranges_by_video,
        }
        ranges_str = "; ".join(f"{v}:{['%.0f-%.0fs' % r for r in rs]}"
                               for v, rs in ranges_by_video.items())
        log.info(f"  state {k}: n={n_seg}  dominant={dominant_video} "
                 f"({video_counts[dominant_video]}/{n_seg})  {ranges_str}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", default="/home/amin/Research/Representation/Movie/outputs/cluster")
    p.add_argument("--model", required=True)
    p.add_argument("--modality", default="av")
    p.add_argument("--config", required=True)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    args = p.parse_args()

    run_dir = Path(args.output_dir) / "group_average" / f"{args.model}_{args.modality}" / args.config
    state_labels = np.load(str(run_dir / "temporal_state_labels.npy"))
    run_trs = np.load(GROUP_AVG_TRS)
    timing_df = pd.read_csv(TIMING_CSV)
    meta = get_segment_metadata(timing_df, run_trs, args.bin_sec, args.tr,
                                delay_sec=args.delay_sec, skip_sec=args.skip_sec)
    log.info(f"state_labels={state_labels.shape}  segment_metadata={meta.shape}")

    summary = summarize_states(state_labels, meta)
    out_path = run_dir / "state_content_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Saved: {out_path}")


def demo():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({
        "seg_idx": range(6),
        "run_id": [1, 1, 1, 1, 1, 1],
        "video_id": ["video1", "video1", "video1", "video2", "video2", "video2"],
        "window_i": [0, 1, 2, 0, 1, 2],
        "stim_start_sec": [0, 5, 10, 20, 25, 35],
        "stim_end_sec": [5, 10, 15, 25, 30, 40],
    })
    # state 0 = video1's contiguous 0-15s block; state 1 = video2's two chunks
    state_labels = np.array([0, 0, 0, 1, 1, 1])
    summary = summarize_states(state_labels, meta)
    assert summary[0]["dominant_video"] == "video1"
    assert summary[0]["time_ranges_by_video"]["video1"] == [(0.0, 15.0)]
    assert summary[1]["time_ranges_by_video"]["video2"] == [(20.0, 30.0), (35.0, 40.0)]
    print(f"[demo] summarize_states OK: {summary[1]['time_ranges_by_video']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        main()
