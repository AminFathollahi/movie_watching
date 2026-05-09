"""
05_audiovisual_integration_maps.py
===================================
Derives the Figure 3a display maps from the null-corrected split R² produced
by Script 04. No model fitting — pure map arithmetic.

Figure 3a in Hedger et al. (2025) shows, per vertex:
    X-axis = visual R²     (blue)
    Y-axis = somato R²     (red — here: audio R²)
    Both high = purple     → audiovisual integration zone
    Neither   = transparent

Since a CIFTI dscalar encodes one value per vertex, we produce several
complementary maps that together replicate this 2D visualization in Workbench:

1. R2_{AUDIO_ROI}_nc.dscalar.nii  — already from Script 04, load with red/Reds colormap
2. R2_{VIDEO_ROI}_nc.dscalar.nii  — already from Script 04, load with blue/Blues colormap
   (dual-overlay of 1+2 in Workbench = Figure 3a)

3. integration_score.dscalar.nii
   Harmonic mean of the positive parts of both null-corrected maps.
   High only where BOTH audio AND video specificity are positive.
   Formula: 2*a*b/(a+b)  for a,b > 0;  0 elsewhere.
   This is the single-map summary of the 2D colormap.

4. modality_balance.dscalar.nii
   R2_{AUDIO_ROI}_nc − R2_{VIDEO_ROI}_nc
   Diverging colormap in Workbench shows which modality dominates per vertex.
   Positive = audio-dominant, negative = video-dominant, near-zero = balanced.

5. integration_mask.npy / .dscalar.nii
   Binary mask: vertices in the top MASK_PERCENTILE of integration_score.
   Used by downstream scripts (06, 07, 08) as the audiovisual integration ROI.

Inputs:
    OUTPUT_DIR/R2_{AUDIO_ROI}_nc.npy   (59412,)  from Script 04
    OUTPUT_DIR/R2_{VIDEO_ROI}_nc.npy   (59412,)  from Script 04

Outputs:
    OUTPUT_DIR/integration_score.{npy,dscalar.nii}
    OUTPUT_DIR/modality_balance.{npy,dscalar.nii}
    OUTPUT_DIR/integration_mask.{npy,dscalar.nii}

Run
---
    conda activate vicsompy_av
    python 05_audiovisual_integration_maps.py
"""

# =============================================================================
# CONFIG
# =============================================================================
import os
DATA_BASE  = "/home/amin/Research/Representation/Movie/data/Setareh"
TEMPLATE_CIFTI = (f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
                  "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii")

# ── ROI selection ─────────────────────────────────────────────────────────────
AUDIO_ROI = os.getenv("AUDIO_ROI")
VIDEO_ROI = os.getenv("VIDEO_ROI")

OUTPUT_DIR = f"/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/{AUDIO_ROI}_{VIDEO_ROI}"
PREP_DIR   = f"{OUTPUT_DIR}/prep"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"

# Top percentile of integration_score used to define the integration mask
MASK_PERCENTILE = 90   # top 10% of vertices


# =============================================================================
# IMPORTS
# =============================================================================

import os
import logging

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(CIFTI_DIR, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def save_map(arr, name, template):
    """Save (59412,) array as .npy and .dscalar.nii."""
    arr_f32 = arr.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr_f32)
    cifti = nib.Cifti2Image(arr_f32.reshape(1, -1),
                            header=template.header,
                            nifti_header=template.nifti_header)
    nib.save(cifti, os.path.join(CIFTI_DIR, f"{name}.dscalar.nii"))
    log.info(f"  Saved {name}: min={arr.min():.4f}  max={arr.max():.4f}  "
             f"mean={arr.mean():.4f}  frac>0={np.mean(arr>0):.1%}")


def square_(a, b):
    """Square root of product of positive parts: high only where both a>0 and b>0.

    For the integration score, we want a metric that is high only when BOTH
    audio and video null-corrected R² are positive. The square root of the product
    penalises imbalance more than the arithmetic mean: if either value is near zero, the
    score collapses to near zero even if the other is large.
    """
    a_pos = np.clip(a, 0, None)
    b_pos = np.clip(b, 0, None)
    score = np.sqrt(a_pos * b_pos)
    return score.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 05 — Audiovisual integration maps (Figure 3a data)")
    log.info("=" * 60)

    # --- Load null-corrected maps from Script 04 ---
    log.info("\nLoading null-corrected R² maps …")
    R2_audio_nc = np.load(os.path.join(OUTPUT_DIR, f"R2_{AUDIO_ROI}_nc.npy"))
    R2_video_nc = np.load(os.path.join(OUTPUT_DIR, f"R2_{VIDEO_ROI}_nc.npy"))
    log.info(f"  R2_{AUDIO_ROI}_nc: mean={R2_audio_nc.mean():.4f}  frac>0={np.mean(R2_audio_nc>0):.1%}")
    log.info(f"  R2_{VIDEO_ROI}_nc: mean={R2_video_nc.mean():.4f}  frac>0={np.mean(R2_video_nc>0):.1%}")

    template = nib.load(TEMPLATE_CIFTI)

    # --- Integration score: square root of product of positive null-corrected R² ---
    # This single map encodes the 2D colormap information: it is high only in
    # vertices where both audio AND video topographic signals are present.
    # Corresponds to the "purple" zone in Figure 3a.
    integration_score = square_(R2_audio_nc, R2_video_nc)
    save_map(integration_score, "integration_score", template)

    # --- Modality balance: signed difference ---
    # Positive = audio-dominant, negative = video-dominant.
    # Load in Workbench with a diverging colormap (e.g. red-white-blue).
    modality_balance = (R2_audio_nc - R2_video_nc).astype(np.float32)
    save_map(modality_balance, "modality_balance", template)

    # --- Integration mask: binary ROI for downstream analyses (06, 07, 08) ---
    # Vertices in the top MASK_PERCENTILE of the integration_score.
    # Only computed over vertices with positive integration score (truly bimodal).

    threshold = np.percentile(integration_score, MASK_PERCENTILE)
    integration_mask = (integration_score >= threshold).astype(np.float32)
    log.info(f"\n  Integration mask: threshold={threshold:.5f}  "
             f"n_verts={int(integration_mask.sum())}  "
             f"({100*(1-MASK_PERCENTILE/100):.0f}% of positive-score vertices)")
    save_map(integration_mask, f"top_{MASK_PERCENTILE}_percentile_integration", template)

    # --- Workbench viewing instructions ---
    log.info("\nWorkbench dual-overlay for Figure 3a replication:")
    log.info(f"  Overlay 1: R2_{VIDEO_ROI}_nc.dscalar.nii  → colormap Blues  [0, 95th pct]")
    log.info(f"  Overlay 2: R2_{AUDIO_ROI}_nc.dscalar.nii  → colormap Reds   [0, 95th pct]")
    p95_audio = np.percentile(R2_audio_nc[R2_audio_nc > 0], 95) if (R2_audio_nc > 0).any() else 0
    p95_video = np.percentile(R2_video_nc[R2_video_nc > 0], 95) if (R2_video_nc > 0).any() else 0
    log.info(f"  Suggested range: {AUDIO_ROI} [0, {p95_audio:.4f}]  {VIDEO_ROI} [0, {p95_video:.4f}]")

    log.info("\nScript 05 complete.")
    log.info("  Next: python 06_rsa_overlap.py")


if __name__ == "__main__":
    main()
