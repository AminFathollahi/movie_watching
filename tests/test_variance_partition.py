import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import variance_partition
from encoding.shared import splits
from encoding.shared.fold_evaluator import ALL_SUBSETS
from encoding.variance_partition import clip_r2, partition_maps, stem


def _args(root, **kwargs):
    base = dict(
        model="joint", audio_model="wav", video_model="vid", joint_model_template=None, split="fixed",
        bin_sec=5.0, skip_sec=5.0, embeddings_dir=str(root), timing_csv=str(root / "timing.csv"),
        exclude_video_ids="video2,video4", hrf=False, preprocessed_dir=str(root), subject="s",
        fmri_suffix="raw", delay_sec=5.0, tr=1.0, subsets=ALL_SUBSETS,
    )
    return SimpleNamespace(**{**base, **kwargs})


def _write_inputs(root, monkeypatch, models):
    timing = pd.DataFrame({
        "video_id": ["video1", "video2", "video3", "video4"], "run_id": [1, 1, 2, 2],
        "onset_sec": [0, 10, 20, 30], "duration_sec": [10] * 4,
    })
    timing.to_csv(root / "timing.csv", index=False)
    rng = np.random.default_rng(0)
    for name, band in models:
        folder = root / name / "bin5s_skip5s"
        folder.mkdir(parents=True)
        np.save(folder / f"{name}_{band}.npy", rng.normal(size=(8, 3)))
    monkeypatch.setattr(
        splits, "build_fmri_arrays",
        lambda *a, **k: (rng.normal(size=(4, 5)), rng.normal(size=(4, 5)), np.array([0, 2])),
    )


def _argv(root, *extra):
    return [
        "--preprocessed-dir", str(root), "--timing-csv", str(root / "timing.csv"),
        "--embeddings-dir", str(root), "--output-dir", str(root / "out"),
        "--template-cifti", "template", "--split", "fixed", "--feature-scaling", "center",
        "--exclude-video-ids", "video2,video4", "--backend", "numpy", "--n-iter", "2",
        "--n-alphas", "3", *extra,
    ]


def test_single_feature_screening_fits_one_model_in_its_own_directory(tmp_path, monkeypatch):
    _write_inputs(tmp_path, monkeypatch, [("wav", "a")])
    saved = {}
    monkeypatch.setattr(
        variance_partition, "save_cifti_maps",
        lambda values, template, path: saved.update({Path(path).name: list(values)}),
    )
    args = variance_partition.parse_args(_argv(tmp_path, "--subsets", "a", "--audio-model", "wav", "--tag", "screen"))
    out = variance_partition.run(args)
    assert out.parts[-2] == "wav"
    assert saved == {
        "encoding_r2_fixed_center_screen_models.dscalar.nii": ["r2_a"],
        "encoding_pearson_r_fixed_center_screen_models.dscalar.nii": ["r_a"],
    }
    fold = np.load(out / "encoding_r2_fixed_center_screen_per_fold.npz")
    assert {"a", "alphas_a", "deltas_a", "clip_r2_a"} <= set(fold.files) and "avj" not in fold.files
    provenance = json.loads((out / "encoding_r2_fixed_center_screen_provenance.json").read_text())
    assert provenance["subsets"] == ["a"] and provenance["audio_model"] == "wav"
    assert provenance["video_model"] is None and provenance["partition"] is None


def test_screening_requires_a_model():
    with pytest.raises(SystemExit):
        variance_partition.parse_args(["--subsets", "a", "--preprocessed-dir", "x"])


def test_partition_has_ten_maps_including_j_minus_av():
    rng = np.random.default_rng(1)
    r2 = {key: rng.normal(size=5).astype(np.float32) for key in ALL_SUBSETS}
    maps = partition_maps(r2)
    assert len(maps) == 10
    np.testing.assert_allclose(maps["j_minus_av"], r2["j"] - r2["av"])


def test_a_v_and_j_come_from_their_own_models(tmp_path, monkeypatch):
    timing = pd.DataFrame({
        "video_id": ["video1", "video2", "video3", "video4"], "run_id": [1, 1, 2, 2],
        "onset_sec": [0, 10, 20, 30], "duration_sec": [10] * 4,
    })
    timing.to_csv(tmp_path / "timing.csv", index=False)
    for name, band, fill in (("wav", "a", 1.0), ("vid", "v", 2.0), ("joint", "av", 3.0)):
        folder = tmp_path / name / "bin5s_skip5s"
        folder.mkdir(parents=True)
        np.save(folder / f"{name}_{band}.npy", np.full((8, 3), fill))
    monkeypatch.setattr(
        splits, "build_fmri_arrays",
        lambda *a, **k: (np.zeros((4, 5)), np.zeros((4, 5)), np.array([0, 2])),
    )
    data = splits.load_inputs(_args(tmp_path))
    assert (data["a"] == 1).all() and (data["v"] == 2).all() and (data["joint"]["default"] == 3).all()
    assert (data["audio_model"], data["video_model"]) == ("wav", "vid")


def test_tag_is_inserted_after_the_scaling():
    args = SimpleNamespace(split="fixed", feature_scaling="center", tag="unimodal")
    assert stem(args, "r2") == "encoding_r2_fixed_center_unimodal"
    assert stem(args, "pearson_r") == "encoding_pearson_r_fixed_center_unimodal"
    args.tag = None
    assert stem(args, "r2") == "encoding_r2_fixed_center"


def test_clip_r2_scores_each_clip_on_its_own_rows():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(6, 2))
    prediction = y.copy()
    prediction[3:] += 1.0
    out = clip_r2(y, {"a": prediction}, np.array(["c2"] * 3 + ["c1"] * 3))
    assert list(out["clips"]) == ["c2", "c1"]
    assert out["clip_r2_a"].shape == (2, 2)
    np.testing.assert_allclose(out["clip_r2_a"][0], 1.0, atol=1e-6)
    assert (out["clip_r2_a"][1] < 1.0).all()
