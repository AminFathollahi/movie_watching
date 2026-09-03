from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_combined_map_names, save_cifti_multimap
from rsa.max_uni import migrate_legacy_artifacts


def _template(path: Path, n_vertices: int = 4) -> None:
    bm_axis = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="CortexLeft"
    )
    series_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=1)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    nib.save(
        nib.Cifti2Image(np.zeros((1, n_vertices), dtype=np.float32), header),
        path,
    )


def test_migrate_legacy_artifacts_renames_files_maps_and_summary_keys(tmp_path):
    legacy_stem = "delta_rho_target_vs_a+v_config"
    current_stem = "max_uni_target_vs_a+v_config"
    n_subs = 3
    template = tmp_path / "template.dtseries.nii"
    _template(template)

    old_cifti = tmp_path / f"{legacy_stem}_{n_subs}subs.dscalar.nii"
    save_cifti_multimap(
        np.stack([
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32) + 1,
        ]),
        ["delta_rho", "rho_max_base"],
        str(template),
        str(old_cifti),
    )
    old_summary = tmp_path / f"{legacy_stem}_{n_subs}subs_summary.json"
    old_summary.write_text(json.dumps({
        "delta_rho_range": [-0.1, 0.2],
        "n_sig_c2f_delta": 10,
        "rho_max_base_range": [0.0, 0.3],
    }))

    migrated = migrate_legacy_artifacts(
        tmp_path, legacy_stem, current_stem, n_subs
    )

    new_cifti = tmp_path / f"{current_stem}_{n_subs}subs.dscalar.nii"
    new_summary = tmp_path / f"{current_stem}_{n_subs}subs_summary.json"
    assert migrated == [new_cifti, new_summary]
    assert not old_cifti.exists()
    assert not old_summary.exists()
    assert get_combined_map_names(new_cifti) == ["max_uni", "rho_max_uni"]
    np.testing.assert_array_equal(
        nib.load(new_cifti).get_fdata(dtype=np.float32)[0],
        np.arange(4, dtype=np.float32),
    )
    summary = json.loads(new_summary.read_text())
    assert summary == {
        "max_uni_range": [-0.1, 0.2],
        "n_sig_c2f_max_uni": 10,
        "rho_max_uni_range": [0.0, 0.3],
    }
