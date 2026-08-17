import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.prepare_wb_view_cortical import discover_group_results


def test_discover_excludes_per_subject_and_cache_dirs(tmp_path: Path):
    root = tmp_path
    hits = []
    for folder in ("group_average", "groupstats", "noise_ceiling"):
        path = root / "raw" / folder / "modelA" / "result.dscalar.nii"
        path.parent.mkdir(parents=True)
        path.touch()
        hits.append(path)

    for skip_dir in ("subject_data", "_geodesic_cache", "rdm_diagonal", "spin_tests"):
        path = root / "raw" / skip_dir / "result.dscalar.nii"
        path.parent.mkdir(parents=True)
        path.touch()

    assert discover_group_results(root) == sorted(hits)
