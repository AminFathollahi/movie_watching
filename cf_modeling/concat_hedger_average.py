"""
Concatenate Hedger's 4 per-run preprocessed CIFTIs for subject 999999
into two output files:

  1. cortex-only CIFTI (108441 grayords) — used as CIFTI template downstream
  2. full-brain CIFTI (170494 grayords) — used for model fitting to match
     Hedger's exact pipeline (they fit on all grayords including subcortex)

Input:  4 × (T_run, 170494) CIFTIs — cortex = first 108441 grayords
Output:
  group_average_hedger_sg_psc_cortex_59k.dtseries.nii   (3655, 108441)
  group_average_hedger_sg_psc_fullbrain.dtseries.nii    (3655, 170494)
  group_average_hedger_sg_psc_run_trs.npy               [921, 918, 915, 901]
"""
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

HEDGER_DIR = Path("/home/amin/Research/Representation/Movie/data/hedger2026"
                  "/hcp_movie/hcp_movie/subjects/999999")
OUT_DIR    = Path("/home/amin/Research/Representation/Movie/data/preprocessed"
                  "/average_sub/hedger_sg_psc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [HEDGER_DIR / f"tfMRI_MOVIE{r}_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean"
                      f".dtseries_sg_psc.nii"
        for r in range(1, 5)]
N_CORTEX = 108441   # L(54216) + R(54225)

# ------------------------------------------------------------------
# Load full runs
# ------------------------------------------------------------------
arrays, run_trs = [], []
bm_axis_full = None
for p in RUNS:
    img = nib.load(str(p))
    if bm_axis_full is None:
        bm_axis_full = img.header.get_axis(1)
    d = img.get_fdata(dtype="float32")          # (T_run, 170494)
    arrays.append(d)
    run_trs.append(d.shape[0])
    print(f"  {p.name}: {d.shape}  mean={d.mean():.4f}")

data_full = np.concatenate(arrays, axis=0)     # (T_total, 170494)
run_trs   = np.array(run_trs, dtype=np.int32)
T = data_full.shape[0]
print(f"\nFull-brain concatenated: {data_full.shape}   run_trs={run_trs}   sum={run_trs.sum()}")

# ------------------------------------------------------------------
# Build SeriesAxis (TR = 1 s)
# ------------------------------------------------------------------
series_ax = nib.cifti2.SeriesAxis(start=0.0, step=1.0, size=T)

# ------------------------------------------------------------------
# 1. Full-brain CIFTI  (170494 grayords — matches Hedger's fit)
# ------------------------------------------------------------------
header_full  = nib.cifti2.Cifti2Header.from_axes((series_ax, bm_axis_full))
img_full_out = nib.Cifti2Image(data_full, header=header_full)
cifti_full   = OUT_DIR / "group_average_hedger_sg_psc_fullbrain.dtseries.nii"
nib.save(img_full_out, str(cifti_full))
print(f"Saved full-brain CIFTI : {cifti_full}")

# ------------------------------------------------------------------
# 2. Cortex-only CIFTI  (108441 grayords — used as output template)
# ------------------------------------------------------------------
data_cortex = data_full[:, :N_CORTEX]          # (T, 108441)
bm_cortex   = bm_axis_full[np.arange(N_CORTEX)]
header_ctx  = nib.cifti2.Cifti2Header.from_axes((series_ax, bm_cortex))
img_ctx_out = nib.Cifti2Image(data_cortex, header=header_ctx)
cifti_ctx   = OUT_DIR / "group_average_hedger_sg_psc_cortex_59k.dtseries.nii"
nib.save(img_ctx_out, str(cifti_ctx))
print(f"Saved cortex-only CIFTI: {cifti_ctx}")

# ------------------------------------------------------------------
# 3. run_trs (named to match _run_disk() lookup convention)
# ------------------------------------------------------------------
trs_out = OUT_DIR / "group_average_hedger_sg_psc_run_trs.npy"
np.save(str(trs_out), run_trs)
print(f"Saved run_trs          : {trs_out}")

# ------------------------------------------------------------------
# Quick verify
# ------------------------------------------------------------------
chk_full = nib.load(str(cifti_full))
chk_ctx  = nib.load(str(cifti_ctx))
print(f"\nVerify full-brain shape  : {chk_full.shape}  (expected ({T}, 170494))")
print(f"Verify cortex-only shape : {chk_ctx.shape}   (expected ({T}, {N_CORTEX}))")
print(f"run_trs: {run_trs}")
