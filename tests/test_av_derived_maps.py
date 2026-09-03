from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_combined_map_names, save_cifti_multimap
from rsa.av_derived_maps import (
    FAMILY_ORDER,
    append_derived_maps,
    compute_derived_maps,
    dependency_pairs,
    expected_map_names,
)
from rsa.shared.model_registry import av_derived_baselines


def _template(path: Path, n_vertices: int = 4) -> None:
    bm_axis = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="CortexLeft"
    )
    series_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=1)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    nib.save(nib.Cifti2Image(np.zeros((1, n_vertices), dtype=np.float32), header), path)


def test_baseline_mapping_uses_own_encoders_for_omni_families():
    pe = av_derived_baselines("pe-av-small-16-frame")

    assert pe["own"] == (
        ("pe-av-small-16-frame", "a"),
        ("pe-av-small-16-frame", "v"),
    )
    targets = {
        "omni3b_layer1": "omni3b_encoder_penultimate",
        "topoomni_layer27_sheet_mp": "topoomni_encoder_penultimate",
        "nemotron_layer36_lt": "nemotron_encoder_penultimate",
    }
    for target, encoder in targets.items():
        assert av_derived_baselines(target)["own"] == (
            (encoder, "a"),
            (encoder, "v"),
        )
    assert av_derived_baselines("topoomni_layer27_mp_avscramble") is None


def test_compute_derived_maps_has_expected_comparisons():
    av = np.array([0.5, 0.4, -0.1, np.nan], dtype=np.float32)
    baselines = {
        family: (
            np.array([0.2, 0.5, -0.2, 0.0], dtype=np.float32),
            np.array([0.3, 0.1, -0.3, 0.0], dtype=np.float32),
        )
        for family in FAMILY_ORDER
    }

    maps = compute_derived_maps(av, baselines)

    np.testing.assert_array_equal(
        maps["av_conjunction_own"], np.array([1, 0, 0, 0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        maps["av_superadditivity_own"][:3],
        np.array([0.0, -0.2, 0.4], dtype=np.float32),
        atol=1e-7,
    )
    assert np.isnan(maps["av_superadditivity_own"][3])
    np.testing.assert_allclose(
        maps["av_max_uni_own"][:3],
        np.array([0.2, -0.1, 0.1], dtype=np.float32),
        atol=1e-7,
    )
    assert np.isnan(maps["av_max_uni_own"][3])


def test_append_derived_maps_resumes_per_map_without_replacing_existing(tmp_path):
    target = "pe-av-small-16-frame"
    config = "k100_delay5s_bin5s_skip5s_spearman"
    rho_filename = "rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"
    rsa_root = tmp_path / "group_average"
    template = tmp_path / "template.dtseries.nii"
    combined = rsa_root / f"{target}_av" / "rsa_59k_raw_maps.dscalar.nii"
    _template(template)

    values = {
        (target, "av"): np.array([0.6, 0.2, -0.1, 0.9], dtype=np.float32),
    }
    for index, pair in enumerate(dependency_pairs(target), start=1):
        values[pair] = np.full(4, 0.05 * index, dtype=np.float32)
    for (model, modality), data in values.items():
        path = rsa_root / f"{model}_{modality}" / config / rho_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)

    combined.parent.mkdir(parents=True, exist_ok=True)
    preserved_conjunction = np.full(4, 7, dtype=np.float32)
    save_cifti_multimap(
        np.stack([np.arange(4, dtype=np.float32), preserved_conjunction]),
        ["existing", "av_conjunction_own"],
        str(template),
        str(combined),
    )

    added = append_derived_maps(
        target_model=target,
        rsa_root=rsa_root,
        config=config,
        rho_filename=rho_filename,
        combined_output=combined,
        template_cifti=template,
    )

    assert added == expected_map_names()[1:]
    assert get_combined_map_names(combined) == ["existing", *expected_map_names()]
    data = nib.load(combined).get_fdata(dtype=np.float32)
    np.testing.assert_array_equal(data[0], np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(data[1], preserved_conjunction)

    # A completed CIFTI is a true resume: no source is re-merged/replaced.
    assert append_derived_maps(
        target_model=target,
        rsa_root=rsa_root,
        config=config,
        rho_filename=rho_filename,
        combined_output=combined,
        template_cifti=template,
    ) == []
    np.testing.assert_array_equal(
        nib.load(combined).get_fdata(dtype=np.float32)[1], preserved_conjunction
    )
