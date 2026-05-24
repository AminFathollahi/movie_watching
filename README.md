# Movie-Watching fMRI — Representation Analysis

HCP 7T movie-watching fMRI analyses examining how cortical regions represent audiovisual content. Three analysis pillars, each self-contained with its own `run_analysis.sh`.

## Repository Structure

```
movie_watching/
├── preprocess_individual.py    # Continuous per-run signal cleaning → full-run CIFTI
├── make_average.sh             # Build group-average CIFTI from per-subject CIFTIs
├── rsa/                        # Representational Similarity Analysis
├── encoding/                   # Ridge encoding models (himalaya)
├── cf_modeling/                # Audiovisual cortical field modeling (Hedger et al. 2025)
└── notebooks/
    ├── visualization/          # RSA significance maps, surface plots, pycortex flatmaps
    └── feature_extraction/     # Model embedding extraction
```

## Analysis Pillars

| Folder | Method | Script | Environment |
|---|---|---|---|
| `rsa/` | Searchlight + Glasser parcel RSA | `run_analysis.sh` | `analysis` |
| `encoding/` | Ridge encoding (himalaya RidgeCV) | `run_analysis.sh` | `analysis` |
| `cf_modeling/` | Banded ridge cortical field modeling (Hedger 2025) | `run_analysis.sh` | `cfmod` |

## Quick Start

### 1. Preprocess fMRI (optional — streaming mode skips this step)

`preprocess_individual.py` applies continuous per-run signal cleaning (SG high-pass → PSC → GSR) and concatenates all 4 runs into a single full-run CIFTI. Timing filtering and hemodynamic delay are applied at analysis time, not here.

```bash
conda activate analysis

# Save per-subject continuous CIFTIs to disk (disk mode for RSA/encoding)
python preprocess_individual.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --out-dir /path/to/data/preprocessed \
    --subjects-list /path/to/data/subjects.txt \
    --sg-filter --psc --gsr \
    --save-individual

# Also save group-average CIFTI (used as template for CIFTI output headers)
python preprocess_individual.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --out-dir /path/to/data/average_sub/sg_psc_gsr \
    --subjects-list /path/to/data/subjects.txt \
    --sg-filter --psc --gsr \
    --save-average
```

Output per subject: `{sub}_sg_psc_gsr_cortex_59k.dtseries.nii` + `{sub}_sg_psc_gsr_run_trs.npy`

> **Streaming mode** (default for RSA per-subject): Set `STREAM=true` in `rsa/run_analysis.sh`.
> Raw CIFTIs are preprocessed on-the-fly inside the analysis scripts — no preprocessed CIFTI is needed.

### 2. Extract model embeddings

```bash
conda activate avtransformer
# Edit MODEL_CONFIGS in the notebook, then run all cells:
jupyter notebook notebooks/feature_extraction/pe_av_embeddings.ipynb
```

Output: `{EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{v,a,av}.npy`

### 3. Run RSA

```bash
# From repo root:
bash rsa/run_analysis.sh avg                      # group-average (disk mode)
bash rsa/run_analysis.sh persubject               # all subjects, streaming (default)
bash rsa/run_analysis.sh persubject all 4 100610  # 4 parallel jobs, resume from 100610
bash rsa/run_analysis.sh groupstats              # re-aggregate per-subject maps
```

### 4. Run encoding models

```bash
bash encoding/run_analysis.sh avg          # group-average
bash encoding/run_analysis.sh persubject   # all subjects
```

### 5. Run CF modeling

```bash
conda activate cfmod
bash cf_modeling/run_analysis.sh all           # group-average + per-subject
bash cf_modeling/run_analysis.sh persubject    # per-subject only
```

## Data Conventions

- **fMRI space**: HCP 7T, 4 runs, TR = 1 s, 59k grayordinate surface (cortex only, 59412 vertices)
- **Preprocessing**: SG high-pass → PSC → GSR applied per run on the full continuous run; outputs a concatenated dtseries. No timing filtering is done here.
- **Timing**: `data/movie_timing.csv` is the authoritative timing file (18 videos, 4 runs, global `onset_sec`). All analysis scripts load it and apply hemodynamic delay at analysis time.
- **Hemodynamic delay**: Applied at analysis time via `--delay-sec 5.0` (boxcar shift) or `--hrf` (SPM HRF convolution). Not baked into preprocessed CIFTIs.
- **Embeddings**: `{EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{v,a,av}.npy`
- **Outputs**: `{OUTPUT_DIR}/{subject_or_group_average}/{model}/{config_label}/`
- **Sphere vs grayordinate space**: The 59k_fs_LR sphere has 59292 vertices per hemisphere (118584 bilateral); only 59412 are grayordinates (medial wall excluded). CF modeling and analysis scripts translate subsurface indices via a sphere→grayordinate LUT.

## Environments

```bash
conda env create -f rsa/environment.yml          # analysis  (RSA + encoding)
conda env create -f cf_modeling/environment.yml  # cfmod     (CF modeling)
conda env create -f notebooks/feature_extraction/environment.yml  # avtransformer
```

