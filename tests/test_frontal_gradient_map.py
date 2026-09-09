"""Tests for the winner-take-all seed dominance map in frontal_gradient_map.py."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "cf_modeling" / "frontal_gradient_map.py"
SPEC = importlib.util.spec_from_file_location("frontal_gradient_map", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

N_GRAY = 6


def _template(path: Path) -> None:
    brain = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(N_GRAY, dtype=bool), name="CortexLeft")
    scalar = nib.cifti2.ScalarAxis(["dummy"])
    header = nib.cifti2.Cifti2Header.from_axes((scalar, brain))
    nib.save(nib.Cifti2Image(np.zeros((1, N_GRAY), dtype=np.float32), header=header), path)


def test_whole_cortex_floor_and_exclusion(tmp_path):
    template_path = tmp_path / "template.dscalar.nii"
    _template(template_path)
    template_image = nib.load(template_path)

    cca_a = np.array([0.5, -0.5, 0.0, 0.3, -0.2, 0.4], dtype=np.float32)
    cca_p = np.array([0.2, 0.1, -0.1, 0.3, -0.3, 0.6], dtype=np.float32)
    a_path = tmp_path / "cca_a.npy"
    p_path = tmp_path / "cca_p.npy"
    np.save(a_path, cca_a)
    np.save(p_path, cca_p)

    exclude_mask = np.zeros(N_GRAY, dtype=np.float32)
    exclude_mask[0] = 1.0
    exclude_path = tmp_path / "cca_a_mask.dscalar.nii"
    MODULE._save_dscalar(exclude_mask[None, :], ["cca_a"], template_image, exclude_path)

    out_dir = tmp_path / "out"
    MODULE.main([
        "--seed", f"cca_a={a_path}",
        "--seed", f"cca_p={p_path}",
        "--template-cifti", str(template_path),
        "--exclude-roi", f"cca_a={exclude_path}",
        "--output-dir", str(out_dir),
    ])

    image = nib.load(out_dir / "frontal_gradient_winner_take_all.dscalar.nii")
    label_map, value_map = np.asarray(image.dataobj)

    # Vertex 0 excluded (cca_a's own ROI) despite the largest raw r overall.
    assert np.isnan(label_map[0])
    # Vertices 2 and 4 have max r <= 0 -> unlabelled, not a false winner.
    assert np.isnan(label_map[2]) and np.isnan(label_map[4])
    # Vertex 3 is a tie (0.3 == 0.3); argmax keeps the first seed, cca_a.
    assert label_map[3] == 1 and np.isclose(value_map[3], 0.3, atol=1e-6)
    # Vertices 1 and 5: cca_p strictly dominates.
    assert label_map[1] == 2 and np.isclose(value_map[1], cca_p[1], atol=1e-6)
    assert label_map[5] == 2 and np.isclose(value_map[5], cca_p[5], atol=1e-6)

    import pandas as pd
    summary = pd.read_csv(out_dir / "frontal_gradient_summary.csv").set_index("seed")
    assert summary.loc["unlabelled", "n_vertices"] == 2
    assert summary.loc["cca_a", "n_vertices"] == 1
    assert summary.loc["cca_p", "n_vertices"] == 2
    # Candidate cortex excludes vertex 0: 5 vertices, fractions sum to 1.
    assert np.isclose(summary["fraction_of_candidate_cortex"].sum(), 1.0)


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        test_whole_cortex_floor_and_exclusion(tmp_dir)
        print("ok")
    finally:
        shutil.rmtree(tmp_dir)
