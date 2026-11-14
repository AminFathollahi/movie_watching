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
from encoding.variance_partition import clip_r2, stem


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
    np.save(root / "s_raw_run_trs.npy", np.array([20, 20]))
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


def test_later_fit_adds_its_subsets_and_completes_the_models(tmp_path, monkeypatch):
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
    first = load_cifti_maps(models)
    assert list(first) == ["r2_a", "r2_av"]

    fit("--subsets", "v", "j", "aj", "vj", "avj")
    merged = load_cifti_maps(models)
    assert list(merged) == [f"r2_{key}" for key in ALL_SUBSETS]
    np.testing.assert_array_equal(merged["r2_a"], first["r2_a"])
    assert not list(out.glob("*_partition.dscalar.nii"))
    fold = np.load(out / "encoding_r2_fixed_demean_t_per_fold.npz")
    assert {"a", "avj", "alphas_a", "deltas_avj", "clip_r2_a", "clip_r2_avj"} <= set(fold.files)
    provenance = json.loads((out / "encoding_r2_fixed_demean_t_provenance.json").read_text())
    assert provenance["subsets"] == list(ALL_SUBSETS) and provenance["audio_model"] == "wav"

    with pytest.raises(ValueError, match="n_iter"):
        fit("--subsets", "j", "--n-iter", "3")
    fit("--n-iter", "3")


def test_saved_predictions_reproduce_the_saved_r2(tmp_path, monkeypatch):
    _write_inputs(tmp_path, monkeypatch, [("wav", "a"), ("vid", "v"), ("joint", "av")])
    saved = {}
    monkeypatch.setattr(
        variance_partition, "save_cifti_maps",
        lambda values, template, path: saved.update({Path(path).name: values}),
    )
    args = variance_partition.parse_args(_argv(
        tmp_path, "--model", "joint", "--audio-model", "wav", "--video-model", "vid", "--tag", "t", "--save-predictions", "--subject", "s",
    ))
    out = variance_partition.run(args)
    stem_r2 = "encoding_r2_fixed_demean_t"
    observed = np.load(out / f"{stem_r2}_observed.npy")
    rows = pd.read_csv(out / f"{stem_r2}_rows.csv")
    assert list(rows.columns) == ["run", "clip", "onset_sec", "fold"] and len(rows) == len(observed)
    assert rows["onset_sec"].tolist() == [10, 15, 10, 15] and set(rows["clip"]) == {"video2", "video4"}
    assert not (out / f"{stem_r2}_predictions_aj.npy").exists() and not (out / f"{stem_r2}_predictions_vj.npy").exists()
    for key in ("a", "v", "av", "j", "avj"):
        predicted = np.load(out / f"{stem_r2}_predictions_{key}.npy")
        r2 = 1 - ((observed - predicted) ** 2).sum(0) / ((observed - observed.mean(0)) ** 2).sum(0)
        np.testing.assert_allclose(r2, saved[f"{stem_r2}_models.dscalar.nii"][f"r2_{key}"], atol=1e-5)


def test_screening_requires_a_model():
    with pytest.raises(SystemExit):
        variance_partition.parse_args(["--subsets", "a", "--preprocessed-dir", "x"])


def test_partition_has_ten_joint_dependent_maps_and_four_av_maps():
    from encoding.collect_partitions import av_maps, partition_maps

    rng = np.random.default_rng(1)
    r2 = {key: rng.normal(size=5).astype(np.float32) for key in ALL_SUBSETS}
    maps, av = partition_maps(r2), av_maps(r2)
    assert len(maps) == 10 and len(av) == 4
    ga, gv = av["gain_a_over_v"], av["gain_v_over_a"]
    np.testing.assert_allclose(ga, r2["av"] - r2["v"], atol=1e-6)
    np.testing.assert_allclose(gv, r2["av"] - r2["a"], atol=1e-6)
    np.testing.assert_allclose(ga + gv + av["shared_av"], r2["av"], atol=1e-6)
    product = av["gain_a_over_v_times_gain_v_over_a"]
    np.testing.assert_allclose(product, np.maximum(ga, 0) * np.maximum(gv, 0))
    assert (product >= 0).all() and (product[(ga < 0) & (gv < 0)] == 0).all()
    np.testing.assert_allclose(maps["j_minus_av"], r2["j"] - r2["av"])
    np.testing.assert_allclose(av["shared_av"], maps["shared_av_only"] + maps["shared_avj"], atol=1e-6)


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


def test_loro_keeps_the_first_showing_of_the_repeated_clip(tmp_path, monkeypatch):
    clips = ["video1", "video5", "video6", "video9", "video11", "video14", "video15", "video18"]
    pd.DataFrame({
        "video_id": clips, "run_id": np.repeat([1, 2, 3, 4], 2),
        "onset_sec": np.arange(8) * 10, "duration_sec": [10] * 8,
    }).to_csv(tmp_path / "timing.csv", index=False)
    folder = tmp_path / "joint" / "bin5s_skip5s"
    folder.mkdir(parents=True)
    np.save(folder / "joint_av.npy", np.zeros((16, 3)))
    np.save(tmp_path / "s_raw_run_trs.npy", np.array([20] * 4))
    monkeypatch.setattr(
        splits, "build_fmri_arrays",
        lambda *a, **k: (np.zeros((10, 5)), np.zeros((6, 5)), np.array([0, 4, 6, 8])),
    )
    data = splits.load_inputs(_args(tmp_path, split="loro", exclude_video_ids=None, subsets=("j",), audio_model=None, video_model=None))
    kept = set(data["metadata"]["video_id"][data["keep"]])
    assert data["excluded"] == {"video9", "video14", "video18"} and "video5" in kept
    (_, _, test), *_ = splits.make_folds(data, "loro")
    assert set(data["metadata"]["video_id"][test]) == {"video1", "video5"}
    assert splits.DEFAULT_EXCLUDED["fixed"] == splits.REPEATED_VALIDATION_CLIPS


def test_tag_is_inserted_after_the_scaling():
    args = SimpleNamespace(split="fixed", feature_scaling="demean", tag="unimodal", model="m", output_name=None, subsets=ALL_SUBSETS, audio_model=None, video_model=None)
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
    args = SimpleNamespace(split="loco", feature_scaling="none", response_scaling="none", tag="t", model="m", output_name=None, subsets=ALL_SUBSETS, audio_model=None, video_model=None)
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


def test_collect_writes_one_file_per_quantity_with_the_pairs_as_maps(tmp_path):
    import nibabel as nib

    from encoding.collect_partitions import collect_folder
    from encoding.shared.encoding_utils import load_cifti_maps, save_cifti_maps

    axes = (nib.cifti2.ScalarAxis(["x"]), nib.cifti2.BrainModelAxis.from_mask(np.ones(4, bool), name="CORTEX_LEFT"))
    template = str(tmp_path / "template.dscalar.nii")
    nib.save(nib.Cifti2Image(np.zeros((1, 4)), header=nib.cifti2.Cifti2Header.from_axes(axes)), template)
    rng = np.random.default_rng(0)
    base = {tag: {key: rng.normal(size=4).astype(np.float32) for key in ALL_SUBSETS} for tag in ("peavs16fa_peavs16fv", "clap_viclip", "peavs16fdummya_peavs16fdummyv", "nemotronl18mpcata_nemotronl18mpcatv")}
    audio = {"peavs16fa_peavs16fv": "pe-av-small-16-frame", "clap_viclip": "clap-larger", "peavs16fdummya_peavs16fdummyv": "pe-av-small-16-frame_dummy_av"}
    video = {"peavs16fa_peavs16fv": "pe-av-small-16-frame", "clap_viclip": "viclip-l14", "peavs16fdummya_peavs16fdummyv": "pe-av-small-16-frame_dummy_av"}
    own = {"peav": "peavs16fa_peavs16fv", "nemotron": "nemotronl18mpcata_nemotronl18mpcatv"}
    models = {"peav": "pe-av-small-16-frame", "nemotron": "nemotron_layer18_mpcat"}
    r2 = {}
    for joint in ("peav", "nemotron"):
        r2[joint] = {tag: dict(values) for tag, values in base.items() if tag != own["nemotron" if joint == "peav" else "peav"]}
        for tag, values in r2[joint].items():
            values["j"] = values["j"] + (1 if joint == "nemotron" else 0)
            prefix = tmp_path / f"encoding_r2_loro_demean_{joint}_{tag}"
            save_cifti_maps({f"r2_{key}": values[key] for key in ALL_SUBSETS}, template, f"{prefix}_models.dscalar.nii")
            Path(f"{prefix}_provenance.json").write_text(json.dumps({
                "tag": tag, "audio_model": audio.get(tag, models[joint]), "video_model": video.get(tag, models[joint]),
                "joint_models": {"default": models[joint]}}))
    report = collect_folder(tmp_path, ["demean"], colour_maps=False)
    free = load_cifti_maps(str(tmp_path / "shared_av_demean.dscalar.nii"))
    assert list(free) == ["clap_viclip", "peavs16fa_peavs16fv", "nemotronl18mpcata_nemotronl18mpcatv", "peavs16fdummya_peavs16fdummyv"]
    np.testing.assert_allclose(free["clap_viclip"], base["clap_viclip"]["a"] + base["clap_viclip"]["v"] - base["clap_viclip"]["av"], atol=1e-6)
    assert report["demean"]["joint_free_max_abs_difference_between_fits"]["clap_viclip"] == {"a": 0.0, "v": 0.0, "av": 0.0}
    unique = load_cifti_maps(str(tmp_path / "unique_j_nemotron_demean.dscalar.nii"))
    assert list(unique) == ["clap_viclip", "nemotronl18mpcata_nemotronl18mpcatv", "peavs16fdummya_peavs16fdummyv"]
    np.testing.assert_allclose(unique["clap_viclip"], r2["nemotron"]["clap_viclip"]["avj"] - r2["nemotron"]["clap_viclip"]["av"], atol=1e-6)
    gain = load_cifti_maps(str(tmp_path / "own_avj_minus_av_peav_demean.dscalar.nii"))
    assert list(gain) == ["clap_viclip", "peavs16fdummya_peavs16fdummyv"]
    own_values, other = r2["peav"]["peavs16fa_peavs16fv"], r2["peav"]["clap_viclip"]
    np.testing.assert_allclose(gain["clap_viclip"], own_values["avj"] - other["av"], atol=1e-6)
    shared = load_cifti_maps(str(tmp_path / "own_shared_av_minus_shared_av_peav_demean.dscalar.nii"))
    np.testing.assert_allclose(shared["clap_viclip"], (own_values["a"] + own_values["v"] - own_values["av"]) - (other["a"] + other["v"] - other["av"]), atol=1e-6)
    summary = load_cifti_maps(str(tmp_path / "own_vs_controls_peav_demean.dscalar.nii"))
    np.testing.assert_allclose(summary["mean_own_shared_av_minus_shared_av"], shared["clap_viclip"], atol=1e-6)
    assert not (tmp_path / "unique_j_demean.dscalar.nii").exists()


def test_tags_name_the_audio_and_video_models():
    from encoding.shared.tags import ENCODER_PAIRS, own_tag, pair_tag

    assert own_tag("pe-av-small-16-frame") == "peavs16fa_peavs16fv"
    assert own_tag("nemotron_layer27_mpcat") == "nemotronl27mpcata_nemotronl27mpcatv"
    assert pair_tag("clap-larger", "viclip-l14") == "clap_viclip"
    assert pair_tag("whisper-large-v3-l30", "qwen2.5-vl-3b-vit-l32") == "whisperl30_qwen25vl3bvitl32"
    assert [pair_tag(*models) for models in ENCODER_PAIRS] == ["nemotronaencl25_nemotronvencl32", "nemotronaencl32_nemotronvencl32", "whisperl32_qwen25vl3bvitl32"]
    assert pair_tag("languagebind-audio", "languagebind-video") == "languagebinda_languagebindv"
    assert pair_tag("dasheng-1.2b", "pe-av-small-16-frame") == "dasheng12b_peavs16fv"


def test_av_bivariate_limits_and_label_index():
    from cf_modeling.bivariate_cifti import quantize_bivariate
    from encoding.av_bivariate import TYPES, clean_limit, symmetric_texture

    assert set(TYPES) == {"r2", "gain"} and TYPES["r2"]["limit"] == 0.20 and TYPES["gain"]["limit"] == 0.10
    assert clean_limit([np.linspace(0, 1, 101)]) == 1 and clean_limit([np.linspace(0, 0.27, 1001)]) == 0.25
    texture = symmetric_texture()
    swapped = texture[::-1, ::-1].transpose(1, 0, 2)
    luminance = lambda t: t[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    assert texture.shape[0] == texture.shape[1] and np.allclose(luminance(texture), luminance(swapped), atol=0.02) and texture[0, -1, :3].min() > 0.9
    texture = np.ones((8, 8, 4), dtype=np.float32)
    keys, _ = quantize_bivariate(np.array([0.5 * 0.2]), np.array([0.1875 * 0.2 + 0.0001]), texture, bins=32, vmin=0.0, vmax=0.2)
    assert keys[0] == 1 + 6 * 32 + 16


def test_joint_model_is_named_in_pair_fits_only():
    def namespace(model, output_name=None):
        return SimpleNamespace(split="loro", feature_scaling="demean", tag="clap_viclip", model=model, output_name=output_name,
                               subsets=ALL_SUBSETS, audio_model="clap-larger", video_model="viclip-l14")

    assert stem(namespace("pe-av-small-16-frame"), "r2") == "encoding_r2_loro_demean_peav_clap_viclip"
    assert stem(namespace("nemotron_layer18_mpcat"), "pearson_r") == "encoding_pearson_r_loro_demean_nemotron_clap_viclip"
    assert stem(namespace("nemotron_layer27_mpcat"), "r2") == "encoding_r2_loro_demean_clap_viclip"
    assert stem(namespace("pe-av-small-16-frame", "pe-av-small-16-frame_resid"), "r2") == "encoding_r2_loro_demean_clap_viclip"
    screen = namespace("nemotron_layer18_mpcat")
    screen.subsets = ("a",)
    assert stem(screen, "r2") == "encoding_r2_loro_demean_clap_viclip"


def test_fit_done_requires_the_requested_settings(tmp_path, monkeypatch):
    import nibabel as nib

    _write_inputs(tmp_path, monkeypatch, [("wav", "a"), ("vid", "v"), ("joint", "av")])
    axes = (nib.cifti2.ScalarAxis(["x"]), nib.cifti2.BrainModelAxis.from_mask(np.ones(5, bool), name="CORTEX_LEFT"))
    template = str(tmp_path / "template.dscalar.nii")
    nib.save(nib.Cifti2Image(np.zeros((1, 5)), header=nib.cifti2.Cifti2Header.from_axes(axes)), template)

    def parse(*extra):
        argv = _argv(tmp_path, "--model", "joint", "--audio-model", "wav", "--video-model", "vid", "--tag", "t", "--subject", "s", *extra)
        argv[argv.index("--template-cifti") + 1] = template
        return variance_partition.parse_args(argv)

    assert not variance_partition.fit_done(parse())
    out = variance_partition.run(parse())
    provenance = out / "encoding_r2_fixed_demean_t_provenance.json"
    assert variance_partition.fit_done(parse()) and variance_partition.fit_done(parse("--subsets", "a", "av"))
    assert not variance_partition.fit_done(parse("--n-iter", "3"))
    assert not variance_partition.fit_done(parse("--model-random-state", "1"))
    assert not variance_partition.fit_done(parse("--audio-model", "other"))
    assert not variance_partition.fit_done(parse("--feature-scaling", "zscore"))
    stored = json.loads(provenance.read_text())
    provenance.write_text(json.dumps({**stored, "delay_sec": 6.0}))
    assert not variance_partition.fit_done(parse())
    older = {key: value for key, value in stored.items() if key not in ("bin_sec", "skip_sec", "delay_sec")}
    provenance.write_text(json.dumps(older))
    assert variance_partition.fit_done(parse())
    provenance.write_text(json.dumps(stored)[:40])
    assert not variance_partition.fit_done(parse())
    provenance.unlink()
    assert not variance_partition.fit_done(parse())


def test_collect_layers_orders_the_maps_by_hidden_state(tmp_path, monkeypatch):
    import nibabel as nib

    from encoding import collect_layers, job_list
    from encoding.shared.encoding_utils import load_cifti_maps, save_cifti_maps
    from encoding.shared.layout import fit_folder

    monkeypatch.setattr(collect_layers, "group_dir", lambda space: tmp_path)
    monkeypatch.setattr(collect_layers, "EMBEDDINGS", tmp_path / "embeddings")
    for name, info in [(f"m-l{k:02d}", {"model": "m", "layer": k}) for k in (10, 0, 2, 1)] + [("m-merger", {"model": "m", "layer": 33}), ("viclip-l14", {"model": "viclip-l14"})]:
        (tmp_path / "embeddings" / name / "bin5s_skip5s").mkdir(parents=True)
        (tmp_path / "embeddings" / name / "bin5s_skip5s" / f"{name}_layers.json").write_text(json.dumps(info))
    (tmp_path / "viclip-l14").mkdir()
    axes = (nib.cifti2.ScalarAxis(["x"]), nib.cifti2.BrainModelAxis.from_mask(np.ones(4, bool), name="CORTEX_LEFT"))
    template = str(tmp_path / "template.dscalar.nii")
    nib.save(nib.Cifti2Image(np.zeros((1, 4)), header=nib.cifti2.Cifti2Header.from_axes(axes)), template)
    settings = {key: 1 for key in collect_layers.SAME_ACROSS_LAYERS}
    for k in (10, 0, 2, 1, 11):
        folder = fit_folder(tmp_path, f"m-l{k:02d}" if k < 11 else "m-merger", 5)
        folder.mkdir(parents=True)
        for metric, prefix in (("pearson_r", "r_v"), ("r2", "r2_v")):
            save_cifti_maps({prefix: np.full(4, k + (0.5 if metric == "r2" else 0))}, template, str(folder / f"encoding_{metric}_loro_zscore_screen_models.dscalar.nii"))
        (folder / "encoding_r2_loro_zscore_screen_provenance.json").write_text(json.dumps(settings))
    report = collect_layers.collect("cortex", 5, "m", ["zscore", "demean"])
    assert list(report) == ["zscore"] and collect_layers.layer_models(tmp_path) == ["m"]
    out = fit_folder(tmp_path, "layers", 5)
    maps = load_cifti_maps(str(out / "encoding_pearson_r_loro_zscore_screen_m_layers.dscalar.nii"))
    assert list(maps) == ["layer_01", "layer_02", "layer_03", "layer_11", "layer_12"]
    assert [float(v[0]) for v in maps.values()] == [0, 1, 2, 10, 11]
    assert report["zscore"]["maps"]["layer_11"]["hidden_state"] == 10 and report["zscore"]["maps"]["layer_12"]["hidden_state"] == 11
    args = SimpleNamespace(spaces=["cortex"], bins=[5], scalings=["zscore"])
    monkeypatch.setattr(job_list, "group_dir", lambda space: tmp_path)
    assert [line.split("\t")[3] for line in job_list.collect_rows(args, "python") if "layers" in line.split("\t")[0]] == ["done"]
