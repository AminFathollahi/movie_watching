"""
Validate our CF model pipeline using Hedger's subject 999999 data.

Concatenates Hedger's 4 per-run sg_psc CIFTIs (cortex only),
runs our 02_fit_cf_model.py, and compares R² maps.

Usage:
    conda run -n movie python cf_modeling/validate_hedger_data.py
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np
import nibabel as nib

MOVIE_ROOT    = Path("/home/amin/Research/Representation/Movie")
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
# Step 2: Run CF model on Hedger's data (3b x V1)
# =============================================================================
validate_output = VALIDATE_DIR / "group_average" / "3b_V1"
validate_output.mkdir(parents=True, exist_ok=True)

cmd = [
    "conda", "run", "--no-capture-output", "-n", "movie",
    "python", str(CF_DIR / "02_fit_cf_model.py"),
    "--mode",             "group_average",
    "--roi-a",            "3b",
    "--roi-b",            "V1",
    "--preprocessed-dir", str(VALIDATE_DIR),
    "--fmri-suffix",      "hedger_999999_sg_psc",
    "--template-cifti",   str(out_cifti),
    "--output-base",      str(VALIDATE_DIR),
    "--glasser-dlabel",   GLASSER_DLABEL,
    "--pycortex-store",   PYCORTEX_STORE,
    "--masks-dir",        MASKS_DIR,
    "--vicsompy-repo",    str(VICSOMPY_REPO),
    "--backend",          "torch_cuda",
    "--n-lboe",           "200",
]
print("\nRunning CF model on Hedger's data ...")
print("Command:\n  " + " \\\n  ".join(cmd[5:]))
result = subprocess.run(cmd, capture_output=False, text=True)
if result.returncode != 0:
    print("CF model failed — check output above")
    sys.exit(1)

# =============================================================================
# Step 3: Compare R² maps
# =============================================================================
import glob
our_prep  = OUTPUT_BASE / "group_average" / "3b_V1" / "prep"
hed_prep  = VALIDATE_DIR / "group_average" / "3b_V1" / "prep"

print("\n=== R² comparison: our corrupted data vs Hedger's data ===")
print(f"{'Map':<20}  {'Our mean':>9}  {'Our f>0':>8}  {'Hed mean':>9}  {'Hed f>0':>8}")
print("-" * 62)
for key in ["R2_3b", "R2_V1", "R2_3b_nc", "R2_V1_nc", "R2_full"]:
    our_f = our_prep / f"{key}.npy"
    hed_f = hed_prep / f"{key}.npy"
    if our_f.exists() and hed_f.exists():
        our_a = np.load(str(our_f))
        hed_a = np.load(str(hed_f))
        print(f"{key:<20}  {our_a.mean():>9.4f}  {np.mean(our_a>0):>8.1%}"
              f"  {hed_a.mean():>9.4f}  {np.mean(hed_a>0):>8.1%}")
    else:
        print(f"{key:<20}  (missing)")

print("\nDone. Hedger validation results in:", hed_prep)
