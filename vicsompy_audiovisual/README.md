# Audiovisual Cortical Mapping — Connective Field Pipeline

Replication and extension of Hedger et al. (2025) "Vicarious body maps bridge vision and touch in the human brain" (Nature), **substituting auditory and visual topographic areas for somatosensory and visual cortex**.

## Scientific Question
Where in the brain do auditory and visual topographic signals converge? And does this audiovisual integration zone match where the PE-AV joint embedding explains brain representational geometry?

## What We Do

### The Nature Paper Approach (Scripts 01–05)
Hedger et al. fitted connective-field (CF) models from V1 and S1 to every cortex vertex, obtaining a per-vertex R² from each source. They then:
1. Subtracted a null-model R² (mean V1/S1 activity) to isolate **topographic** specificity
2. Plotted these two values as a 2D colormap: blue = visual, red = somatosensory, purple = both

We do the same with configurable audio/video source ROIs (currently **TA2 × MST**):
- `R2_{video_roi}_nc` = how well the video ROI's topographic pattern predicts this vertex (beyond mean)
- `R2_{audio_roi}_nc` = how well the audio ROI's topographic pattern predicts this vertex (beyond mean)
- Purple vertices = audiovisual integration zone

### Our Extension (Script 06)
We compare the integration zone to an independent method: searchlight RSA of PE-AV's joint embedding against brain RDMs. Convergence of two completely different methods on the same region is the core scientific claim.

## ROI Configuration
Each script has a `ROI_DEFS` dict at the top that controls which brain areas are used:

```python
ROI_DEFS = {
    "ta2": {"L": 287, "R": 107},   # audio source ROI — Glasser label codes
    "mst": {"L": 182, "R": 2},     # video source ROI
}
AUDIO_ROI = "ta2"
VIDEO_ROI  = "mst"
```

The dict **key** (e.g. `"ta2"`) becomes the name used in all output files. The **value** holds the Glasser parcellation codes for left (L) and right (R) hemispheres. To run for a different ROI pair, change the dict and keys in all 6 scripts — no other changes needed.

Known codes: A1 (L=204, R=24), V1 (L=181, R=1), TA2 (L=287, R=107), MST (L=182, R=2).

## Run Order
```bash
conda activate vicsompy_av

# Foundation
python 01_extract_geometry.py       # TA2/MST Laplace-Beltrami eigenfunctions → sub_ta2.pkl, sub_mst.pkl
python 02_prep_hcp_timeseries.py    # train/test design matrix + targets
python 03_fit_banded_ridge.py       # banded ridge → R2_ta2, R2_mst (1-2 hrs)

# Null correction & integration maps
python 04_null_corrected_maps.py    # null model → R2_ta2_nc, R2_mst_nc
python 05_audiovisual_integration_maps.py  # integration_score, modality_balance
python 06_rsa_overlap.py            # spatial overlap with PE-AV RSA
```

## Outputs
All outputs go to `/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/`.

ROI-specific files use the ROI key name; derived maps use generic names:

| File | Description |
|---|---|
| `subsurfaces/sub_ta2.pkl` | TA2 Subsurface object + LBOEs |
| `subsurfaces/sub_mst.pkl` | MST Subsurface object + LBOEs |
| `R2_ta2.npy/.dscalar.nii` | TA2-band split R² (audio) |
| `R2_mst.npy/.dscalar.nii` | MST-band split R² (video) |
| `R2_ta2_nc.dscalar.nii` | Null-corrected TA2 R² — load in Workbench with red colormap |
| `R2_mst_nc.dscalar.nii` | Null-corrected MST R² — load in Workbench with blue colormap |
| `integration_score.dscalar.nii` | √(ta2_nc × mst_nc) — audiovisual integration map |
| `modality_balance.dscalar.nii` | R2_ta2_nc − R2_mst_nc — diverging colormap shows dominance |
| `integration_mask.dscalar.nii` | Binary: top 10% of integration_score vertices |
| `overlap_score_{config}.dscalar.nii` | CF integration × PE-AV RSA joint — convergent validity map |

## Viewing in HCP Workbench (Figure 3a Replication)
1. Open Workbench → load any surface
2. Load `R2_mst_nc.dscalar.nii` as Overlay 1 → set colormap to **Blues**, range [0, 0.05]
3. Load `R2_ta2_nc.dscalar.nii` as Overlay 2 → set colormap to **Reds**, range [0, 0.05]
4. Regions that are both blue AND red (purple) = audiovisual integration zone

For a single-map view, load `integration_score.dscalar.nii` with a sequential colormap.

## Planned Extensions
- **Script 07** (CF weights): What part of TA2/MST do integration-zone vertices connect to?  
  Requires refitting Script 03 with weight saving (see CLAUDE.md).
- **Script 08** (RSA geometry): Does the integration zone's representational geometry  
  look more like audio, video, or joint PE-AV information?

## Key Conceptual Note
The `Shared_R2` map from Script 03 (= R2_audio + R2_video − R2_full) is typically negative with himalaya's banded ridge — this is an optimiser property, not a neuroscience finding. The correct way to identify audiovisual integration is the 2D spatial co-occurrence approach from Hedger et al., which is what Scripts 04–05 implement.
