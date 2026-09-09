import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks.feature_extraction.av_pairing import (
    factorial_contrast,
    fold_confined_pairing,
    fold_reference_pairing,
)


def test_factorial_contrast_cancels_additive_components():
    audio = np.arange(12, dtype=float).reshape(6, 2)
    video = np.arange(12, 24, dtype=float).reshape(6, 2)
    references = np.array([1, 0, 3, 2, 5, 4])
    intact = audio + video
    first_cross = video + audio[references]
    second_cross = video[references] + audio

    np.testing.assert_allclose(
        factorial_contrast(intact, references, first_cross, second_cross), 0.0,
    )


def test_fold_pairing_is_deterministic_and_confined():
    timing = pd.DataFrame({
        "video_id": [f"video{i}" for i in range(1, 9)],
        "run_id": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    videos = [Path(f"Video{i}/Video{i}_part_{part:03d}.mp4")
              for i in range(1, 9) for part in range(3)]
    audios = [Path(f"Audio{i}/Audio{i}_part_{part:03d}.wav")
              for i in range(1, 9) for part in range(3)]

    first_audio, first = fold_confined_pairing(
        videos, audios, timing, held_out_run=2, seed=7,
        excluded_clips=("video1", "video8"),
    )
    second_audio, second = fold_confined_pairing(
        videos, audios, timing, held_out_run=2, seed=7,
        excluded_clips=("video1", "video8"),
    )

    assert first_audio == second_audio
    pd.testing.assert_frame_equal(first, second)
    assert np.all(first.target_video_id != first.audio_source_video_id)
    source_splits = first.set_index("target_index").loc[
        first.audio_source_index, "split"
    ].to_numpy()
    assert np.array_equal(first.split.to_numpy(), source_splits)
    assert set(first.loc[first.target_run_id == 2, "split"]) == {"test"}


def test_fold_pairing_rejects_a_single_clip_pool():
    timing = pd.DataFrame({"video_id": ["video1", "video2"], "run_id": [1, 2]})
    videos = [Path("Video1/Video1_part_001.mp4"), Path("Video2/Video2_part_001.mp4")]
    audios = [Path("Audio1/Audio1_part_001.wav"), Path("Audio2/Audio2_part_001.wav")]

    try:
        fold_confined_pairing(
            videos, audios, timing, held_out_run=1, seed=0, excluded_clips=(),
        )
    except ValueError as error:
        assert "at least two clips" in str(error)
    else:
        raise AssertionError("Expected a single-clip test pool to be rejected")


def test_fold_reference_pairing_is_deterministic_and_uses_training_references():
    timing = pd.DataFrame({
        "video_id": [f"video{i}" for i in range(1, 9)],
        "run_id": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    videos = [Path(f"Video{i}/Video{i}_part_{part:03d}.mp4")
              for i in range(1, 9) for part in range(3)]

    first_indices, first = fold_reference_pairing(
        videos, timing, held_out_run=2, seed=7,
        excluded_clips=("video1", "video8"),
    )
    second_indices, second = fold_reference_pairing(
        videos, timing, held_out_run=2, seed=7,
        excluded_clips=("video1", "video8"),
    )

    assert np.array_equal(first_indices, second_indices)
    pd.testing.assert_frame_equal(first, second)
    assert np.all(first.target_video_id != first.reference_video_id)
    reference_splits = first.set_index("target_index").loc[
        first.reference_index, "split"
    ].to_numpy()
    assert np.all(reference_splits == "train")

    train = first[first.split == "train"]
    assert train.reference_index.is_unique
    assert set(train.reference_index) == set(train.target_index)
