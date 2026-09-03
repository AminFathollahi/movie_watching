"""
cluster/io_cluster.py
======================
Thin loaders for the cluster pipeline. Re-exports the existing binning /
alignment machinery (rsa/shared/rsa_utils.py) and CIFTI I/O (cifti_io.py)
verbatim — no re-implementation. The only new code here is the group-average
loader (a two-array read, not the compute-and-cache logic subcortical uses,
since the cortical group average is already built by rsa/analysis.sh) and a
minimal .dlabel writer (no writer exists anywhere in the repo).
"""

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsa"))

from rsa.shared.rsa_utils import (  # noqa: E402
    preprocess_fmri, process_model_embeddings, get_run_bin_counts, align_and_assert_bins,
)
from cifti_io import (  # noqa: E402
    load_cifti_data, get_bm_axis, get_cortex_vertex_indices, save_cifti_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROUP_AVG_CIFTI = "/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
GROUP_AVG_TRS = "/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy"


def load_group_average(cifti_path: str = GROUP_AVG_CIFTI, trs_path: str = GROUP_AVG_TRS):
    """Load the cached group-average cortical timeseries.

    Returns
    -------
    X       : (V, T) float32
    run_trs : (n_runs,) int
    bm_axis : nibabel BrainModelAxis
    """
    X = load_cifti_data(cifti_path)
    run_trs = np.load(trs_path)
    bm_axis = get_bm_axis(cifti_path)
    log.info(f"Group average loaded: X={X.shape}  run_trs={run_trs.tolist()}")
    return X, run_trs, bm_axis


def get_segment_metadata(timing_df, run_trs: np.ndarray, bin_sec: float, tr: float,
                         delay_sec: float = 0.0, skip_sec: float = None):
    """Per-segment (video_id, stimulus time window) for every row preprocess_fmri
    produces. Mirrors preprocess_fmri's windowing loop exactly (same n_wins,
    start_tr, and early-break-at-run-boundary logic) so row i here is segment i
    in fmri_binned / process_model_embeddings output / HMM state_labels.

    No existing function returns this — preprocess_fmri discards it after
    slicing, and get_run_bin_counts only returns per-run totals.

    Returns
    -------
    pandas.DataFrame with columns: seg_idx, run_id, video_id, window_i,
    stim_start_sec, stim_end_sec (stimulus-clock time, NOT fMRI/delay time)
    """
    import pandas as pd

    if skip_sec is None:
        skip_sec = bin_sec
    bin_trs = max(1, int(np.round(bin_sec / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
    run_col = ('run' if 'run' in timing_df.columns
               else ('run_id' if 'run_id' in timing_df.columns else None))
    groups = ([(None, timing_df)] if run_col is None
             else list(timing_df.groupby(run_col, sort=False)))

    rows = []
    seg_idx = 0
    for run_idx, (run_id, run_df) in enumerate(groups):
        run_tr_count = run_trs[run_idx]
        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr
        for _, row in run_df.iterrows():
            dur = row["duration_sec"]
            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)
            if n_wins == 0:
                continue
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            if start_tr >= run_tr_count or start_tr < 0:
                continue
            for i in range(n_wins):
                w_start = start_tr + i * skip_trs
                w_end = w_start + bin_trs
                if w_start >= run_tr_count or w_end > run_tr_count:
                    break
                rows.append({
                    "seg_idx": seg_idx, "run_id": run_id, "video_id": row.get("video_id", "?"),
                    "window_i": i,
                    "stim_start_sec": float(row["onset_sec"] + i * skip_sec),
                    "stim_end_sec": float(row["onset_sec"] + i * skip_sec + bin_sec),
                })
                seg_idx += 1
    return pd.DataFrame(rows)


def _get_distinct_colormap(n_colors: int):
    """Return a list of n_colors distinct RGBA tuples.
    
    Uses a combination of matplotlib's qualitative colormaps (tab20, tab20b, tab20c)
    for up to 60 colors, then falls back to a golden-angle HSV sequence for more.
    """
    import colorsys
    import matplotlib
    
    colors = []
    qualitative_cmaps = ["tab20", "tab20b", "tab20c"]
    
    for cmap_name in qualitative_cmaps:
        cmap = matplotlib.colormaps[cmap_name]
        for i in range(cmap.N):
            if len(colors) >= n_colors:
                return colors
            r, g, b, _ = cmap(i / cmap.N)
            colors.append((r, g, b, 1.0))
    
    for i in range(n_colors - len(colors)):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((r, g, b, 1.0))
    
    return colors


def write_dlabel(labels_1d: np.ndarray, template_path: str, out_path: str,
                 network_rgba: dict = None, label_names: dict = None,
                 map_name: str = "cluster_labels") -> np.ndarray:
    """Write a 1-D integer label map as a CIFTI .dlabel.nii.

    Noise / unassigned vertices (label == -1) are remapped to key 0
    ("unassigned", fully transparent). Every other unique label value is
    remapped to consecutive keys 1..G in sorted order and given a distinct
    color from a combined qualitative colormap (tab20+tab20b+tab20c for up to
    60 networks, then golden-angle HSV sequence).

    Parameters
    ----------
    labels_1d      : (V,) int — raw cluster labels, noise/unassigned = -1
    template_path   : CIFTI whose BrainModelAxis is reused (defines V and vertex order)
    out_path        : destination .dlabel.nii path
    network_rgba    : optional {remapped_key: (r,g,b,a)} override, all in [0,1]
    label_names     : optional {remapped_key: name} override (default "network_{key}")
    map_name        : name of the single LabelAxis row

    Returns
    -------
    remapped : (V,) int32 — the label vector actually written (0 = unassigned)
    """
    import colorsys
    import matplotlib

    labels_1d = np.asarray(labels_1d)
    if labels_1d.ndim != 1:
        raise ValueError(f"labels_1d must be one-dimensional, got shape {labels_1d.shape}")
    if not np.isfinite(labels_1d).all():
        raise ValueError("labels_1d contains NaN or infinite values")
    if not np.equal(labels_1d, np.floor(labels_1d)).all():
        raise ValueError("labels_1d must contain integer-valued cluster labels")
    labels_1d = labels_1d.astype(int)
    uniq = sorted(int(u) for u in np.unique(labels_1d) if u != -1)
    key_map = {-1: 0}
    key_map.update({orig: i + 1 for i, orig in enumerate(uniq)})
    remapped = np.array([key_map[v] for v in labels_1d], dtype=np.int32)

    max_key = max(key_map.values())
    distinct_colors = _get_distinct_colormap(max_key)
    
    label_table = {0: ((label_names or {}).get(0, "unassigned"), (0.0, 0.0, 0.0, 0.0))}
    for orig, key in key_map.items():
        if key == 0:
            continue
        if network_rgba is not None and key in network_rgba:
            rgba = network_rgba[key]
        else:
            rgba = distinct_colors[key - 1]
        name = (label_names or {}).get(key, f"network_{key}")
        label_table[key] = (name, rgba)

    template = nib.load(str(template_path))
    bm_axis = template.header.get_axis(1)
    if len(bm_axis) != remapped.size:
        raise ValueError(
            f"Label vector has {remapped.size} entries but template BrainModelAxis "
            f"has {len(bm_axis)}"
        )
    label_axis = nib.cifti2.LabelAxis([map_name], [label_table])
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    arr = remapped.reshape(1, -1)
    img = nib.Cifti2Image(arr, header=header, nifti_header=template.nifti_header)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))

    log.info(f"write_dlabel: V={remapped.shape[0]}  keys={sorted(label_table.keys())}  "
             f"key_map(-1->0, orig->key)={key_map}")
    return remapped


def write_channel_labels_csv(labels_1d: np.ndarray, channel_ids: list[str], out_path: str,
                             label_names: dict = None, map_name: str = "cluster_labels") -> np.ndarray:
    """Write a 1-D integer label vector as a per-channel CSV.

    Channel analogue of :func:`write_dlabel`: same noise/unassigned (-1 -> 0)
    and consecutive 1..G remapping, but channels have no CIFTI BrainModelAxis
    to attach labels to, so this writes a plain table instead.

    Parameters
    ----------
    labels_1d    : (C,) int — raw cluster labels, noise/unassigned = -1
    channel_ids  : (C,) str — one identifier per channel, same order as labels_1d
    out_path     : destination .csv path
    label_names  : optional {remapped_key: name} override (default "cluster_{key}")
    map_name     : recorded in a constant column for provenance when concatenating files

    Returns
    -------
    remapped : (C,) int32 — the label vector actually written (0 = unassigned)
    """
    import pandas as pd

    labels_1d = np.asarray(labels_1d)
    if labels_1d.ndim != 1:
        raise ValueError(f"labels_1d must be one-dimensional, got shape {labels_1d.shape}")
    if len(channel_ids) != labels_1d.shape[0]:
        raise ValueError(
            f"channel_ids has {len(channel_ids)} entries but labels_1d has {labels_1d.shape[0]}"
        )
    if not np.isfinite(labels_1d).all():
        raise ValueError("labels_1d contains NaN or infinite values")
    if not np.equal(labels_1d, np.floor(labels_1d)).all():
        raise ValueError("labels_1d must contain integer-valued cluster labels")
    labels_1d = labels_1d.astype(int)
    uniq = sorted(int(u) for u in np.unique(labels_1d) if u != -1)
    key_map = {-1: 0}
    key_map.update({orig: i + 1 for i, orig in enumerate(uniq)})
    remapped = np.array([key_map[v] for v in labels_1d], dtype=np.int32)
    names = {0: (label_names or {}).get(0, "unassigned")}
    names.update({key: (label_names or {}).get(key, f"cluster_{key}") for key in range(1, max(key_map.values()) + 1)})

    df = pd.DataFrame({
        "channel_index": np.arange(labels_1d.size),
        "channel_id": channel_ids,
        "raw_label": labels_1d,
        "cluster_key": remapped,
        "cluster_name": [names[key] for key in remapped],
        "map_name": map_name,
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    log.info(f"write_channel_labels_csv: C={remapped.shape[0]}  keys={sorted(names.keys())}  "
             f"key_map(-1->0, orig->key)={key_map}")
    return remapped


def demo():
    X, run_trs, bm_axis = load_group_average()
    assert X.shape[0] == 108441, f"expected V=108441, got {X.shape[0]}"
    assert X.shape[1] == 3655, f"expected T=3655, got {X.shape[1]}"
    assert int(run_trs.sum()) == 3655, f"run_trs sum {run_trs.sum()} != 3655"
    print(f"[demo] load_group_average OK: X={X.shape} run_trs={run_trs.tolist()}")

    rng = np.random.default_rng(0)
    fake_labels = rng.integers(-1, 4, size=X.shape[0])  # -1..3
    out_path = "/tmp/claude-1000/-home-amin-Research-Representation-Movie-movie-watching/9e3f6255-2452-440c-8a49-0deca8947cc3/scratchpad/demo_labels.dlabel.nii"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    remapped = write_dlabel(fake_labels, GROUP_AVG_CIFTI, out_path)

    reloaded = nib.load(out_path)
    assert reloaded.shape[1] == X.shape[0], "reloaded dlabel V mismatch"
    label_axis_reloaded = reloaded.header.get_axis(0)
    n_keys = len(label_axis_reloaded.label[0])
    n_expected = len(np.unique(fake_labels))  # includes -1 remapped to 0
    assert n_keys == n_expected, f"label table has {n_keys} entries, expected {n_expected}"
    print(f"[demo] write_dlabel round-trip OK: V={reloaded.shape[1]}  n_label_keys={n_keys}")

    import pandas as pd
    timing_df = pd.read_csv("/home/amin/Research/Representation/Movie/data/movie_timing.csv")
    meta = get_segment_metadata(timing_df, run_trs, bin_sec=5.0, tr=1.0, delay_sec=5.0, skip_sec=5.0)
    assert len(meta) == 626, f"expected 626 segments, got {len(meta)}"
    assert list(meta["seg_idx"]) == list(range(626)), "seg_idx must be contiguous 0..625"
    assert (meta["stim_end_sec"] - meta["stim_start_sec"] == 5.0).all(), "every window must be 5s"
    print(f"[demo] get_segment_metadata OK: {len(meta)} segments, "
         f"videos={sorted(meta['video_id'].unique(), key=lambda s: int(s.replace('video','')))}")


if __name__ == "__main__":
    demo()
