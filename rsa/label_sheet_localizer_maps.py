"""
rsa/label_sheet_localizer_maps.py
=====================================
Merges every per-cluster + combined searchlight result produced by
rsa/run_sheet_localizer_brain_maps.sh into ONE combined dscalar.nii, with
descriptive map names (cluster{N}__t{t}__n{size}__rho / ..._fdr_mask /
allclusters__rho / ...) so all of them can be opened as a single file in
wb_view.

Reads the cluster list from the same summary JSON (see
rsa/localizer_naming.py's summary_json_name()) used by
topoomni_sheet_localizer.py and the shell runner, so this always merges
exactly the clusters that were actually run.

Everything for one analysis lives together: raw per-cluster/all-clusters
searchlight runs under group_average/<analysis>/raw/, the merged deliverable
at group_average/<analysis>/<analysis>_all_clusters_maps.dscalar.nii, and the
cluster summary at group_average/<analysis>/summary.json (already copied
there by the shell runner).

Run with:
    conda run -n movie python rsa/label_sheet_localizer_maps.py <kind> <driver> <sheet> [suffix]
    kind: speech (only localizer kind produced by topoomni_sheet_localizer.py;
    av_integration was retired in favor of topoomni_av_separability_localizer.py)
    suffix: optional, e.g. "_fdr" for the FDR unit-selection variant.
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifti_io import merge_into_combined  # noqa: E402
from rsa.localizer_naming import base_name as loc_base_name, summary_json_name  # noqa: E402

EMBEDDINGS_DIR = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
GROUP_DIR = Path("/home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average")

# Must match run_sheet_localizer_brain_maps.sh's BIN_SEC/SKIP_SEC env vars
# (default 5.0) so the merge looks for the same filenames the searchlight
# run actually wrote.
_BIN_SEC = float(os.environ.get("BIN_SEC", "5.0"))
_SKIP_SEC = float(os.environ.get("SKIP_SEC", "5.0"))
CONFIG = f"k100_delay5s_bin{_BIN_SEC:g}s_skip{_SKIP_SEC:g}s_spearman"

SRC_MAPS = {
    "searchlight_spearman_rho": "rho",
    "searchlight_spearman_rho_sigmap_uncorr": "p_uncorr_mask",
    "searchlight_spearman_rho_sigmap_fdr": "fdr_mask",
}


def main():
    kind, driver, sheet = sys.argv[1], sys.argv[2], sys.argv[3]
    suffix = sys.argv[4] if len(sys.argv) > 4 else ""
    base_name = loc_base_name(kind, driver, sheet, suffix)

    summary_json = EMBEDDINGS_DIR / summary_json_name(kind, driver, sheet, suffix)
    summary = json.load(open(summary_json))
    b = summary[f"{kind}_localizer"]

    analysis_dir = GROUP_DIR / base_name
    raw_dir = analysis_dir / "raw"

    targets = [(f"{base_name}_c{r['cluster_idx']}",
                f"cluster{r['cluster_idx']}", r) for r in b["significant_clusters"]]
    targets.append((f"{base_name}_all", "allclusters", None))

    out_path = analysis_dir / f"{base_name}_all_clusters_maps.dscalar.nii"

    for folder, short_name, r in targets:
        path = raw_dir / f"{folder}_av" / f"rsa_59k_raw_{CONFIG}_maps.dscalar.nii"
        if not path.exists():
            print(f"[MISSING] {folder}: {path}")
            continue
        img = nib.load(str(path))
        names = list(img.header.get_axis(0).name)
        data = img.get_fdata(dtype=np.float32)

        for src_name, map_suffix in SRC_MAPS.items():
            if src_name not in names:
                print(f"  [skip] {src_name} not found in {folder}")
                continue
            idx = names.index(src_name)
            arr = data[idx]
            extra = f"__t{r['t_value']:.2f}__n{r['size']}" if r is not None else ""
            new_name = f"{short_name}{extra}__{map_suffix}"
            merge_into_combined(arr, new_name, out_path, str(path))
            print(f"  [{folder}] added map '{new_name}' -> {out_path.name}")

    print(f"Done. Combined file: {out_path}")
    print(f"n_terminal_clusters={b['n_terminal_clusters']}  "
          f"n_significant={len(b['significant_clusters'])}  "
          f"(see {analysis_dir / 'summary.json'} for the full breakdown)")


if __name__ == "__main__":
    main()
