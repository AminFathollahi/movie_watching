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
        fmri_suffix="raw", delay_sec=5.0, tr=1.0, subsets=ALL_SUBSETS, response_scaling="run", feature_scaling="none",
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
        "--template-cifti", "template", "--split", "fixed", "--feature-scaling", "demean",
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
        "encoding_r2_fixed_demean_screen_models.dscalar.nii": ["r2_a"],
        "encoding_pearson_r_fixed_demean_screen_models.dscalar.nii": ["r_a"],
    }
    fold = np.load(out / "encoding_r2_fixed_demean_screen_per_fold.npz")
    assert {"a", "alphas_a", "deltas_a", "clip_r2_a"} <= set(fold.files) and "avj" not in fold.files
    provenance = json.loads((out / "encoding_r2_fixed_demean_screen_provenance.json").read_text())
    assert provenance["subsets"] == ["a"] and provenance["audio_model"] == "wav"
    assert provenance["video_model"] is None and provenance["partition"] is None


def test_later_fit_adds_its_subsets_and_completes_the_partition(tmp_path, monkeypatch):
    import nibabel as nib

    from encoding.shared.encoding_utils import load_cifti_maps

    _write_inputs(tmp_path, monkeypatch, [("wav", "a"), ("vid", "v"), ("joint", "av")])
    axes = (
        nib.cifti2.ScalarAxis(["x"]),
        nib.cifti2.BrainModelAxis.from_mask(np.ones(5, bool), name="CORTEX_LEFT"),
    )
    template = str(tmp_path / "template.dscalar.nii")
    nib.save(nib.Cifti2Image(np.zeros((1, 5)), header=nib.cifti2.Cifti2Header.from_axes(axes)), template)

    def fit(*extra):
        argv = _argv(
            tmp_path, "--model", "joint", "--audio-model", "wav", "--video-model", "vid",
            "--output-name", "m", "--tag", "t", *extra,
        )
        argv[argv.index("--template-cifti") + 1] = template
        return variance_partition.run(variance_partition.parse_args(argv))

    out = fit("--subsets", "a", "av")
    models = str(out / "encoding_r2_fixed_demean_t_models.dscalar.nii")
    partition = out / "encoding_r2_fixed_demean_t_partition.dscalar.nii"
    first = load_cifti_maps(models)
    assert list(first) == ["r2_a", "r2_av"] and not partition.exists()

    fit("--subsets", "v", "j", "aj", "vj", "avj")
    merged = load_cifti_maps(models)
    assert list(merged) == [f"r2_{key}" for key in ALL_SUBSETS]
    np.testing.assert_array_equal(merged["r2_a"], first["r2_a"])
    np.testing.assert_allclose(
        load_cifti_maps(str(partition))["unique_j"], merged["r2_avj"] - merged["r2_av"], atol=1e-6,
    )
    fold = np.load(out / "encoding_r2_fixed_demean_t_per_fold.npz")
    assert {"a", "avj", "alphas_a", "deltas_avj", "clip_r2_a", "clip_r2_avj"} <= set(fold.files)
    provenance = json.loads((out / "encoding_r2_fixed_demean_t_provenance.json").read_text())
    assert provenance["subsets"] == list(ALL_SUBSETS) and provenance["audio_model"] == "wav"

    with pytest.raises(ValueError, match="n_iter"):
        fit("--subsets", "j", "--n-iter", "3")
    fit("--n-iter", "3")


def test_screening_requires_a_model():
    with pytest.raises(SystemExit):
        variance_partition.parse_args(["--subsets", "a", "--preprocessed-dir", "x"])


def test_partition_has_eleven_maps_including_j_minus_av_and_shared_av():
    rng = np.random.default_rng(1)
    r2 = {key: rng.normal(size=5).astype(np.float32) for key in ALL_SUBSETS}
    maps = partition_maps(r2)
    assert len(maps) == 11
    np.testing.assert_allclose(maps["j_minus_av"], r2["j"] - r2["av"])
    np.testing.assert_allclose(maps["shared_av"], maps["shared_av_only"] + maps["shared_avj"], atol=1e-6)


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
    args = SimpleNamespace(split="fixed", feature_scaling="demean", tag="unimodal")
    assert stem(args, "r2") == "encoding_r2_fixed_demean_unimodal"
    assert stem(args, "pearson_r") == "encoding_pearson_r_fixed_demean_unimodal"
    args.tag = None
    assert stem(args, "r2") == "encoding_r2_fixed_demean"


def test_unscaled_fit_leaves_features_and_responses_untouched(tmp_path, monkeypatch):
    from encoding.shared.encoding_utils import _bin_and_split_fmri

    rng = np.random.default_rng(0)

    timing = pd.DataFrame({"video_id": ["video1", "video2"], "run_id": [1, 2], "onset_sec": [0, 10], "duration_sec": [10, 10]})
    fmri = rng.normal(1000, 5, (4, 20)).astype(np.float32)
    y_train, y_test, _ = _bin_and_split_fmri(fmri, timing, ["video2"], 5.0, 1.0, np.array([10, 10]), zscore=False)
    np.testing.assert_allclose(y_train[0], fmri[:, :5].mean(1), rtol=1e-6)
    np.testing.assert_allclose(y_test[1], fmri[:, 15:].mean(1), rtol=1e-6)
    args = SimpleNamespace(split="loco", feature_scaling="none", response_scaling="none", tag="t")
    assert stem(args, "r2") == "encoding_r2_loco_none_rawresponse_t"


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


def test_own_gain_is_added_to_the_partition_of_other_tags(tmp_path):
    import nibabel as nib

    from encoding.shared.encoding_utils import load_cifti_maps, save_cifti_maps
    from encoding.variance_partition import refresh_own_gains

    axes = (nib.cifti2.ScalarAxis(["x"]), nib.cifti2.BrainModelAxis.from_mask(np.ones(4, bool), name="CORTEX_LEFT"))
    template = str(tmp_path / "template.dscalar.nii")
    nib.save(nib.Cifti2Image(np.zeros((1, 4)), header=nib.cifti2.Cifti2Header.from_axes(axes)), template)
    prefix = "encoding_r2_loro_demean"
    own, other = np.arange(4.0), np.full(4, 0.5)
    fits = {
        "unimodal_own": ({"r2_a": own, "r2_v": own, "r2_av": own, "r2_avj": own + 1}, "m", "m"),
        "text-asr": ({"r2_a": other, "r2_v": other, "r2_av": other, "r2_avj": other}, "whisper", "pe-core"),
        "dummy_av": ({"r2_a": own, "r2_v": other, "r2_av": other, "r2_avj": other}, "m_dummy_av", "m_dummy_av"),
    }
    for tag, (models, audio, video) in fits.items():
        save_cifti_maps(models, template, str(tmp_path / f"{prefix}_{tag}_models.dscalar.nii"))
        save_cifti_maps({"unique_j": models["r2_av"]}, template, str(tmp_path / f"{prefix}_{tag}_partition.dscalar.nii"))
        (tmp_path / f"{prefix}_{tag}_provenance.json").write_text(json.dumps(
            {"tag": tag, "audio_model": audio, "video_model": video, "joint_models": {"default": "m"}}))
    reduced = f"{prefix}_pca32_text-asr"
    save_cifti_maps(fits["text-asr"][0], template, str(tmp_path / f"{reduced}_models.dscalar.nii"))
    save_cifti_maps({"unique_j": other}, template, str(tmp_path / f"{reduced}_partition.dscalar.nii"))
    (tmp_path / f"{reduced}_provenance.json").write_text(json.dumps(
        {"tag": "text-asr", "audio_model": "whisper", "video_model": "pe-core", "joint_models": {"default": "m"}}))
    assert refresh_own_gains(tmp_path, prefix) == ["dummy_av", "text-asr"]
    assert "own_avj_minus_av" not in load_cifti_maps(str(tmp_path / f"{reduced}_partition.dscalar.nii"))
    maps = load_cifti_maps(str(tmp_path / f"{prefix}_text-asr_partition.dscalar.nii"))
    np.testing.assert_allclose(maps["own_avj_minus_av"], own + 1 - other)
    np.testing.assert_allclose(maps["own_shared_av_minus_shared_av"], own - other)
    np.testing.assert_allclose(maps["unique_j"], other)
    summary = load_cifti_maps(str(tmp_path / f"{prefix}_own_vs_controls.dscalar.nii"))
    np.testing.assert_allclose(summary["mean_own_shared_av_minus_shared_av"], own - other)
    np.testing.assert_allclose(summary["n_controls_own_shared_av_greater"], (own > other).astype(float))
    assert "own_avj_minus_av" not in load_cifti_maps(str(tmp_path / f"{prefix}_unimodal_own_partition.dscalar.nii"))
