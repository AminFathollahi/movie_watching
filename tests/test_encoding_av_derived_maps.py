from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_combined_map_names, load_named_map, save_cifti_map
from encoding.av_derived_maps import (
    CONJUNCTION_MASK_NAME,
    CONJUNCTION_NAME,
    SUPERADDITIVITY_NAME,
    STIM_MAP_NAMES,
    append_av_derived_maps,
    compute_av_derived_maps,
    is_supported_target,
)


def _template(path: Path, n_vertices: int = 5) -> None:
    bm_axis = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="CortexLeft"
    )
    series_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=1)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    nib.save(nib.Cifti2Image(np.zeros((1, n_vertices), dtype=np.float32), header), path)


def test_compute_av_derived_maps_matches_requested_formulas():
    av = np.array([0.6, 0.4, -0.1, 0.8, np.nan], dtype=np.float32)
    audio = np.array([0.2, 0.5, -0.3, 0.1, 0.0], dtype=np.float32)
    video = np.array([0.3, 0.1, -0.2, 0.9, 0.0], dtype=np.float32)

    superadditivity, conjunction, mask = compute_av_derived_maps(av, audio, video)

    np.testing.assert_allclose(
        superadditivity[:4], [0.1, -0.2, 0.4, -0.2], atol=1e-7
    )
    assert np.isnan(superadditivity[4])
    np.testing.assert_array_equal(mask, [1, 0, 0, 0, 0])
    np.testing.assert_allclose(conjunction, [0.6, 0, 0, 0, 0])


def test_append_av_derived_maps_preserves_base_and_writes_mask(tmp_path):
    config_dir = tmp_path / "pe-av-small-16-frame" / "config"
    config_dir.mkdir(parents=True)
    template = tmp_path / "template.dtseries.nii"
    _template(template)

    values = {
        "a": np.array([0.2, 0.5, -0.3, 0.1, 0.0], dtype=np.float32),
        "v": np.array([0.3, 0.1, -0.2, 0.9, 0.0], dtype=np.float32),
        "av": np.array([0.6, 0.4, -0.1, 0.8, 0.2], dtype=np.float32),
    }
    modality_names = {"a": "audio", "v": "visual", "av": "audiovisual"}
    for modality, data in values.items():
        map_name = f"encoding_pearson_r_{modality_names[modality]}"
        save_cifti_map(
            data,
            str(template),
            str(config_dir / f"{map_name}.dscalar.nii"),
            map_name,
        )

    stimulus_path = tmp_path / "stimulus.dscalar.nii"
    stimulus = np.array([0.5, -0.2, 0.0, np.nan, 2.0], dtype=np.float32)
    save_cifti_map(
        stimulus, str(template), str(stimulus_path), "stimulus_regressor"
    )

    assert append_av_derived_maps(config_dir, stimulus_map=stimulus_path) == [
        SUPERADDITIVITY_NAME,
        CONJUNCTION_NAME,
        *STIM_MAP_NAMES,
    ]
    av_path = config_dir / "encoding_pearson_r_audiovisual.dscalar.nii"
    assert get_combined_map_names(av_path) == [
        "encoding_pearson_r_audiovisual",
        SUPERADDITIVITY_NAME,
        CONJUNCTION_NAME,
        *STIM_MAP_NAMES,
    ]
    np.testing.assert_array_equal(
        load_named_map(av_path, "encoding_pearson_r_audiovisual"), values["av"]
    )
    # Positive stimulus values become a binary mask: surviving values are
    # identical to the corresponding source maps, not stimulus-weighted.
    np.testing.assert_allclose(
        load_named_map(av_path, "stim_r"), [0.6, 0.0, 0.0, 0.0, 0.2]
    )
    np.testing.assert_allclose(
        load_named_map(av_path, "stim_superadditivity"),
        [0.1, 0.0, 0.0, 0.0, 0.2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        load_named_map(av_path, "stim_conjunction"), [0.6, 0.0, 0.0, 0.0, 0.2]
    )

    mask_path = config_dir / "encoding_pearson_r_audiovisual_conjunction.mask.nii"
    assert mask_path.is_file()
    np.testing.assert_array_equal(
        load_named_map(mask_path, CONJUNCTION_MASK_NAME),
        np.array([1, 0, 0, 0, 1], dtype=np.float32),
    )

    assert append_av_derived_maps(config_dir, stimulus_map=stimulus_path) == []


def test_supported_targets_require_registered_a_v_and_av_modalities():
    assert is_supported_target("pe-av-small-16-frame")
    assert is_supported_target("cav-mae-sync")
    assert is_supported_target("topoomni_layer27_sheet_mp")
    assert not is_supported_target("topoomni_layer27_sheet_lt")
    assert not is_supported_target("imagebind")
