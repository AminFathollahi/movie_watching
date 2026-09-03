from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_combined_map_names, save_cifti_multimap
from rsa.residualized_maps import (
    MAP_ORDER,
    append_residualized_maps,
    output_path,
    source_paths,
)
from rsa.searchlight import _precompute_model_rdm
from rsa.shared.model_registry import LEGACY_BARE_AV_MODELS, RESIDUALIZED_AV_MODELS
from notebooks.feature_extraction.compute_projection_residual_embeddings import (
    _place_own_streams_in_av_space,
)


def _template(path: Path, n_vertices: int = 4) -> None:
    bm_axis = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="CortexLeft"
    )
    series_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=1)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    nib.save(nib.Cifti2Image(np.zeros((1, n_vertices), dtype=np.float32), header), path)


def test_append_residualized_maps_consolidates_and_resumes_per_scalar(tmp_path):
    model = "pe-av-small-16-frame"
    config = "k100_delay5s_bin5s_skip5s_spearman"
    normal_filename = (
        "rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii"
    )
    rsa_root = tmp_path / "group_average"
    template = tmp_path / "template.dtseries.nii"
    _template(template)

    sources = source_paths(rsa_root, model, config, normal_filename)
    values = {
        "partial_correlation": np.full(4, 0.1, dtype=np.float32),
        "linear_resid": np.full(4, 0.2, dtype=np.float32),
        "projection_resid": np.full(4, 0.3, dtype=np.float32),
    }
    for name, source in sources.items():
        source.parent.mkdir(parents=True, exist_ok=True)
        scalar_name = "partial_source" if name == "partial_correlation" else "searchlight_spearman_rho"
        save_cifti_multimap(values[name][None, :], [scalar_name], str(template), str(source))

    destination = output_path(rsa_root, model, normal_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preserved = np.full(4, 7, dtype=np.float32)
    save_cifti_multimap(
        preserved[None, :],
        ["partial_correlation"],
        str(template),
        str(destination),
    )

    added = append_residualized_maps(
        model=model,
        rsa_root=rsa_root,
        config=config,
        normal_maps_filename=normal_filename,
        template_cifti=template,
    )

    assert added == ["linear_resid", "projection_resid"]
    assert get_combined_map_names(destination) == list(MAP_ORDER)
    data = nib.load(destination).get_fdata(dtype=np.float32)
    np.testing.assert_array_equal(data[0], preserved)
    np.testing.assert_array_equal(data[1], values["linear_resid"])
    np.testing.assert_array_equal(data[2], values["projection_resid"])

    assert append_residualized_maps(
        model=model,
        rsa_root=rsa_root,
        config=config,
        normal_maps_filename=normal_filename,
        template_cifti=template,
    ) == []
    np.testing.assert_array_equal(
        nib.load(destination).get_fdata(dtype=np.float32)[0], preserved
    )


def test_cav_streams_are_placed_in_separate_av_blocks():
    audio = np.array([[1.0, 2.0]])
    video = np.array([[3.0, 4.0]])
    av = np.concatenate([audio, video], axis=1)

    audio_block, video_block = _place_own_streams_in_av_space(
        "cav-mae-sync", av, audio, video
    )

    np.testing.assert_array_equal(audio_block, [[1.0, 2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(video_block, [[0.0, 0.0, 3.0, 4.0]])


def test_constant_model_geometry_has_zero_normalized_rdm():
    embeddings = np.full((4, 3), np.nan, dtype=np.float32)
    tril = np.tril_indices(4, k=-1)

    _, normalized = _precompute_model_rdm(embeddings, 4, tril, "spearman")

    np.testing.assert_array_equal(normalized, np.zeros(6, dtype=np.float32))


def test_residualized_roster_covers_legacy_bare_models():
    assert len(RESIDUALIZED_AV_MODELS) == 42
    assert set(LEGACY_BARE_AV_MODELS) <= set(RESIDUALIZED_AV_MODELS)
