"""
Validate our CF model pipeline using Hedger's subject 999999 data.

Concatenates Hedger's 4 per-run sg_psc CIFTIs (cortex only),
runs our 01_extract_geometry.py and 02_fit_cf_model.py, and compares R² maps.

Usage:
    conda run -n movie python cf_modeling/validate_hedger_data.py
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cf_naming import cf_model_map_stems
from paths import ROOT as MOVIE_ROOT

REPO_ROOT     = MOVIE_ROOT / "movie_watching"
CF_DIR        = REPO_ROOT / "cf_modeling"
HEDGER_SUB    = MOVIE_ROOT / "data/hedger2026/hcp_movie/hcp_movie/subjects/999999"
OUTPUT_BASE   = MOVIE_ROOT / "outputs/cf_modeling"
VALIDATE_DIR  = OUTPUT_BASE / "validation_hedger999999"
VALIDATE_DIR.mkdir(parents=True, exist_ok=True)

VICSOMPY_REPO = MOVIE_ROOT / "Vicarious_somatotopy"
GLASSER_DLABEL = str(
    MOVIE_ROOT / "data/HCP_S1200_GroupAvg_v1" /
    "Q1-Q6_RelatedParcellation210"
    ".CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".59k_fs_LR.dlabel.nii"
)
PYCORTEX_STORE = str(MOVIE_ROOT / "data/hedger2026")
MASKS_DIR      = str(OUTPUT_BASE / "masks")
N_LBOE         = 200

# Hedger's run filenames in order (MOVIE1=run1, …, MOVIE4=run4)
RUN_FILES = [
    HEDGER_SUB / "tfMRI_MOVIE1_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii",
    HEDGER_SUB / "tfMRI_MOVIE2_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii",
    HEDGER_SUB / "tfMRI_MOVIE3_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii",
    HEDGER_SUB / "tfMRI_MOVIE4_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii",
]

# =============================================================================
# Step 1: Concatenate Hedger's per-run data → single cortex-only CIFTI
# =============================================================================
out_cifti  = VALIDATE_DIR / "hedger_999999_sg_psc_cortex_59k.dtseries.nii"
out_run_trs = VALIDATE_DIR / "hedger_999999_sg_psc_run_trs.npy"

if not out_cifti.exists():
    print("Concatenating Hedger's per-run CIFTIs ...")
    chunks = []
    run_trs = []
    bm_axis_cortex = None

    for f in RUN_FILES:
        img = nib.load(str(f))
        d   = img.get_fdata(dtype=np.float32)  # (T, 170494)
        # Cortex = first 108441 columns (matches our 59k CIFTI structure)
        cortex = d[:, :108441]
        chunks.append(cortex)
        run_trs.append(cortex.shape[0])
        if bm_axis_cortex is None:
            # Build cortex-only BM axis from Hedger's CIFTI
            full_bm = img.header.get_axis(1)
            cortex_structs = []
            for name, sl, model in full_bm.iter_structures():
                if "CORTEX_LEFT" in name or "CORTEX_RIGHT" in name:
                    cortex_structs.append((name, sl, model))
            bm_axis_cortex = nib.cifti2.BrainModelAxis.from_mask(
                np.ones(108441, dtype=bool), name="CIFTI_STRUCTURE_CORTEX"
            )
            # Use the existing axis directly (already cortex-indexed)
            # Build proper axis from slice info
            bm_axis_cortex = full_bm[0:108441]
        print(f"  {f.name}: {cortex.shape[0]} TRs")

    all_data = np.concatenate(chunks, axis=0)  # (T_total, 108441)
    np.save(str(out_run_trs), np.array(run_trs))
    print(f"Total TRs: {all_data.shape[0]}  run_trs: {run_trs}")

    # Save as dtseries CIFTI using the template from our pipeline
    template = nib.load(str(
        MOVIE_ROOT / "data/preprocessed/average_sub/gsr/group_average_gsr_cortex_59k.dtseries.nii"
    ))
    tr_ax = nib.cifti2.SeriesAxis(start=0.0, step=1.0, size=all_data.shape[0])
    bm_ax = template.header.get_axis(1)
    new_hdr = nib.Cifti2Header.from_axes((tr_ax, bm_ax))
    new_img = nib.Cifti2Image(all_data, new_hdr)
    new_img.to_filename(str(out_cifti))
    print(f"Saved: {out_cifti}")
else:
    run_trs = np.load(str(out_run_trs)).tolist()
    print(f"Concatenated CIFTI already exists. run_trs: {run_trs}")

# =============================================================================
# Step 2: Build the geometry, then fit the CF model on Hedger's data (3b x V1)
# =============================================================================
ROI_A, ROI_B = f"3b_lboe{N_LBOE}", f"V1_lboe{N_LBOE}"
PAIR = f"{ROI_A}_{ROI_B}"
PYTHON = ["conda", "run", "--no-capture-output", "-n", "movie", "python"]
COMMON = [
    "--mode",          "group_average",
    "--roi-a",         ROI_A,
    "--roi-b",         ROI_B,
    "--output-base",   str(VALIDATE_DIR),
    "--vicsompy-repo", str(VICSOMPY_REPO),
]
STEPS = {
    "01_extract_geometry.py": [
        "--n-lboe",         str(N_LBOE),
        "--glasser-dlabel", GLASSER_DLABEL,
        "--pycortex-store", PYCORTEX_STORE,
        "--masks-dir",      MASKS_DIR,
    ],
    "02_fit_cf_model.py": [
        "--preprocessed-dir", str(VALIDATE_DIR),
        "--fmri-suffix",      "hedger_999999_sg_psc",
        "--template-cifti",   str(out_cifti),
        "--backend",          "torch_cuda",
    ],
}
for script, extra in STEPS.items():
    cmd = PYTHON + [str(CF_DIR / script)] + COMMON + extra
    print(f"\nRunning {script} on Hedger's data ...")
    print("Command:\n  " + " \\\n  ".join(cmd[5:]))
    if subprocess.run(cmd, text=True).returncode != 0:
        print(f"{script} failed; check output above")
        sys.exit(1)

# =============================================================================
# Step 3: Compare R² maps
# =============================================================================
our_prep = OUTPUT_BASE / "group_average" / PAIR / "prep"
hed_prep = VALIDATE_DIR / "group_average" / PAIR / "prep"
stems = cf_model_map_stems(ROI_A, ROI_B)

print("\n=== R² comparison: our group average vs Hedger's subject 999999 ===")
print(f"{'Map':<50}  {'Our mean':>9}  {'Our f>0':>8}  {'Hed mean':>9}  {'Hed f>0':>8}")
print("-" * 92)
for key in ["split_r2_a", "split_r2_b", "null_corrected_split_r2_a", "null_corrected_split_r2_b", "full_r2"]:
    our_f = our_prep / f"{stems[key]}.npy"
    hed_f = hed_prep / f"{stems[key]}.npy"
    if our_f.exists() and hed_f.exists():
        our_a = np.load(str(our_f))
        hed_a = np.load(str(hed_f))
        print(f"{stems[key]:<50}  {our_a.mean():>9.4f}  {np.mean(our_a>0):>8.1%}"
              f"  {hed_a.mean():>9.4f}  {np.mean(hed_a>0):>8.1%}")
    else:
        print(f"{stems[key]:<50}  (missing)")

print("\nDone. Hedger validation results in:", hed_prep)
