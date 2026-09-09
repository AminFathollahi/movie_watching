from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


REPEATED_CLIPS = ("video5", "video9", "video14", "video18")


def factorial_contrast(
    intact: np.ndarray,
    reference_indices: np.ndarray,
    target_video_reference_audio: np.ndarray,
    reference_video_target_audio: np.ndarray,
) -> np.ndarray:
    first_cross = np.asarray(target_video_reference_audio)
    second_cross = np.asarray(reference_video_target_audio)
    references = np.asarray(reference_indices, dtype=int)
    target_intact = np.asarray(intact)[: len(references)]
    if target_intact.shape != first_cross.shape or first_cross.shape != second_cross.shape:
        raise ValueError("Intact and crossed embeddings must have matching shapes")
    return target_intact + np.asarray(intact)[references] - first_cross - second_cross


def video_id(path: Path) -> str:
    match = re.search(r"video(\d+)", str(path), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot determine video ID from {path}")
    return f"video{int(match.group(1))}"


def _constrained_permutation(
    indices: np.ndarray, clip_ids: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    clips, counts = np.unique(clip_ids[indices], return_counts=True)
    if clips.size < 2:
        raise ValueError("Each pairing pool must contain at least two clips")
    largest = int(counts.max())
    if largest > len(indices) - largest:
        raise ValueError("No cross-clip pairing exists for this pool")

    ordered = np.concatenate([
        rng.permutation(indices[clip_ids[indices] == clip])
        for clip in rng.permutation(clips)
    ])
    sources = np.roll(ordered, largest)
    permutation = np.empty(len(indices), dtype=int)
    positions = {int(index): position for position, index in enumerate(indices)}
    for target, source in zip(ordered, sources):
        permutation[positions[int(target)]] = source
    if np.any(clip_ids[permutation] == clip_ids[indices]):
        raise RuntimeError("Could not construct a cross-clip pairing")
    return permutation


def fold_confined_pairing(
    video_paths: list[Path],
    audio_paths: list[Path],
    timing: pd.DataFrame,
    held_out_run: int,
    seed: int,
    excluded_clips: tuple[str, ...] = REPEATED_CLIPS,
) -> tuple[list[Path], pd.DataFrame]:
    if len(video_paths) != len(audio_paths) or not video_paths:
        raise ValueError("Video and audio paths must be nonempty and matched")

    timing_runs = {
        str(row.video_id).lower(): int(row.run_id)
        for row in timing.itertuples(index=False)
    }
    clip_ids = np.asarray([video_id(path) for path in video_paths])
    missing = sorted(set(clip_ids) - set(timing_runs))
    if missing:
        raise KeyError(f"Missing timing rows for {missing}")
    run_ids = np.asarray([timing_runs[clip] for clip in clip_ids])
    excluded = np.isin(clip_ids, np.asarray(excluded_clips))
    split = np.where(excluded, "excluded", np.where(run_ids == held_out_run, "test", "train"))

    rng = np.random.default_rng(seed)
    source_indices = np.empty(len(video_paths), dtype=int)
    for split_name in ("train", "test", "excluded"):
        indices = np.flatnonzero(split == split_name)
        source_indices[indices] = _constrained_permutation(indices, clip_ids, rng)

    paired_audio = [Path(audio_paths[index]) for index in source_indices]
    manifest = pd.DataFrame({
        "target_index": np.arange(len(video_paths)),
        "target_video_id": clip_ids,
        "target_run_id": run_ids,
        "split": split,
        "target_video_path": [str(path) for path in video_paths],
        "intact_audio_path": [str(path) for path in audio_paths],
        "audio_source_index": source_indices,
        "audio_source_video_id": clip_ids[source_indices],
        "audio_source_run_id": run_ids[source_indices],
        "paired_audio_path": [str(path) for path in paired_audio],
        "held_out_run": held_out_run,
        "seed": seed,
    })
    if np.any(manifest["target_video_id"] == manifest["audio_source_video_id"]):
        raise RuntimeError("Pairing contains a within-clip audio source")
    if np.any(manifest["split"].to_numpy() != split[source_indices]):
        raise RuntimeError("Pairing crosses train, test, or excluded pools")
    return paired_audio, manifest


def fold_reference_pairing(
    video_paths: list[Path],
    timing: pd.DataFrame,
    held_out_run: int,
    seed: int,
    excluded_clips: tuple[str, ...] = REPEATED_CLIPS,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not video_paths:
        raise ValueError("Video paths must be nonempty")

    timing_runs = {
        str(row.video_id).lower(): int(row.run_id)
        for row in timing.itertuples(index=False)
    }
    clip_ids = np.asarray([video_id(path) for path in video_paths])
    missing = sorted(set(clip_ids) - set(timing_runs))
    if missing:
        raise KeyError(f"Missing timing rows for {missing}")
    run_ids = np.asarray([timing_runs[clip] for clip in clip_ids])
    excluded = np.isin(clip_ids, np.asarray(excluded_clips))
    split = np.where(excluded, "excluded", np.where(run_ids == held_out_run, "test", "train"))

    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(split == "train")
    reference_indices = np.empty(len(video_paths), dtype=int)
    reference_indices[train_indices] = _constrained_permutation(
        train_indices, clip_ids, rng,
    )
    for index in np.flatnonzero(split != "train"):
        candidates = train_indices[clip_ids[train_indices] != clip_ids[index]]
        if not len(candidates):
            raise ValueError("No cross-clip training reference exists")
        reference_indices[index] = int(rng.choice(candidates))

    manifest = pd.DataFrame({
        "target_index": np.arange(len(video_paths)),
        "target_video_id": clip_ids,
        "target_run_id": run_ids,
        "split": split,
        "target_video_path": [str(path) for path in video_paths],
        "reference_index": reference_indices,
        "reference_video_id": clip_ids[reference_indices],
        "reference_run_id": run_ids[reference_indices],
        "reference_video_path": [str(video_paths[index]) for index in reference_indices],
        "held_out_run": held_out_run,
        "seed": seed,
    })
    if np.any(manifest["target_video_id"] == manifest["reference_video_id"]):
        raise RuntimeError("Reference pairing contains a within-clip match")
    if np.any(split[reference_indices] != "train"):
        raise RuntimeError("Reference pairing uses a non-training reference")
    return reference_indices, manifest
