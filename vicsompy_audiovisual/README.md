# Audiovisual Cortical Mapping — A1/V1 Connective Field Pipeline

Replication and extension of Hedger et al. (2025) "Vicarious body maps bridge vision and touch in the human brain" (Nature), **substituting auditory cortex (A1) for somatosensory cortex (S1)**.

## Scientific Question
Where in the brain do auditory and visual topographic signals converge? And does this audiovisual integration zone match where the PE-AV joint embedding explains brain representational geometry?

## What We Do

### The Nature Paper Approach (Scripts 01–05)
Hedger et al. fitted connective-field (CF) models from V1 and S1 to every cortex vertex, obtaining a per-vertex R² from each source. They then:
1. Subtracted a null-model R² (mean V1/S1 activity) to isolate **topographic** specificity
2. Plotted these two values as a 2D colormap: blue = visual, red = somatosensory, purple = both

We do the same with **A1 replacing S1**:
- `R2_video_nc` = how well V1's topographic pattern predicts this vertex (beyond mean V1)
- `R2_audio_nc` = how well A1's topographic pattern predicts this vertex (beyond mean A1)
- Purple vertices = audiovisual integration zone

### Our Extension (Script 06)
We compare the integration zone to an independent method: searchlight RSA of PE-AV's joint embedding against brain RDMs. Convergence of two completely different methods on the same region is the core scientific claim.

## Run Order
```bash
conda activate vicsompy_av

# Foundation (already done — outputs in outputs/vicsompy_audiovisual/)
python 01_extract_geometry.py       # A1/V1 Laplace-Beltrami eigenfunctions
python 02_prep_hcp_timeseries.py    # train/test design matrix + targets
python 03_fit_banded_ridge.py       # banded ridge → R2_audio, R2_video (1-2 hrs)

# New analyses
python 04_null_corrected_maps.py    # null model → R2_audio_nc, R2_video_nc
python 05_audiovisual_integration_maps.py  # integration_score, modality_balance
python 06_rsa_overlap.py            # spatial overlap with PE-AV RSA
```

## Outputs
All outputs go to `/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/`.

| File | Description |
|---|---|
| `R2_audio_nc.dscalar.nii` | Null-corrected audio R² — load in Workbench with red colormap |
| `R2_video_nc.dscalar.nii` | Null-corrected video R² — load in Workbench with blue colormap |
| `integration_score.dscalar.nii` | Harmonic mean of both — audiovisual integration map |
| `modality_balance.dscalar.nii` | R2_audio_nc − R2_video_nc — diverging colormap shows dominance |
| `integration_mask.dscalar.nii` | Binary: top 10% of integration_score vertices |
| `overlap_score_{config}.dscalar.nii` | CF integration × PE-AV RSA joint — convergent validity map |

## Viewing in HCP Workbench (Figure 3a Replication)
1. Open Workbench → load any surface
2. Load `R2_video_nc.dscalar.nii` as Overlay 1 → set colormap to **Blues**, range [0, 0.05]
3. Load `R2_audio_nc.dscalar.nii` as Overlay 2 → set colormap to **Reds**, range [0, 0.05]
4. Regions that are both blue AND red (purple) = audiovisual integration zone

For a single-map view, load `integration_score.dscalar.nii` with a sequential colormap.

## Planned Extensions
- **Script 07** (CF weights): What part of A1/V1 do integration-zone vertices connect to?  
  Requires refitting Script 03 with weight saving (see CLAUDE.md).
- **Script 08** (RSA geometry): Does the integration zone's representational geometry  
  look more like audio, video, or joint PE-AV information?

## Key Conceptual Note
The `Shared_R2` map from Script 03 (= R2_audio + R2_video − R2_full) is typically negative with himalaya's banded ridge — this is an optimiser property, not a neuroscience finding. The correct way to identify audiovisual integration is the 2D spatial co-occurrence approach from Hedger et al., which is what Scripts 04–05 implement.
