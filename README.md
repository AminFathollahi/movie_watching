# Movie-Watching fMRI — Representation Analysis

HCP 7T movie-watching fMRI analyses examining how cortical regions represent audiovisual content. Three analysis pillars, each self-contained with its own `run_analysis.sh` and `environment.yml`.

## Repository Structure

```
movie_watching/
├── preprocess_individual.py    # GSR + z-score per subject → full-run CIFTI
├── rsa/                        # Representational Similarity Analysis
├── encoding/                   # Ridge encoding models (himalaya)
├── cf_modeling/                # Audiovisual cortical field modeling
└── notebooks/
    ├── visualization/          # Supplementary figures
    └── feature_extraction/     # Model embedding extraction
```

## Analysis Pillars

| Folder | Method | Script | Environment |
|---|---|---|---|
| `rsa/` | Searchlight + Glasser parcel RSA | `run_analysis.sh` | `analysis` |
| `encoding/` | Ridge encoding (himalaya RidgeCV) | `run_analysis.sh` | `analysis` |
| `cf_modeling/` | Banded ridge cortical field modeling | `run_analysis.sh` | `cfmod` |

## Quick Start

### 1. Preprocess fMRI

```bash
conda activate analysis
python preprocess_individual.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --out-dir /path/to/outputs/preprocessed \
    --subjects-list /path/to/subjects.txt
```

Outputs per subject: `{sub}_gsr_zscore_cortex_59k.dtseries.nii` + `{sub}_gsr_zscore_run_trs.npy`  
Group average: `group_average_gsr_zscore_cortex_59k.dtseries.nii`

### 2. Extract model embeddings

```bash
conda activate avtransformer
# Edit MODEL_CONFIGS in the notebook, then run all cells:
jupyter notebook notebooks/feature_extraction/pe_av_embeddings.ipynb
```

### 3. Run RSA

```bash
cd rsa && bash run_analysis.sh avg           # group-average
cd rsa && bash run_analysis.sh persubject    # all subjects (GNU parallel)
```

### 4. Run encoding models

```bash
cd encoding && bash run_analysis.sh avg      # group-average
cd encoding && bash run_analysis.sh persubject
```

## Data Conventions

- **fMRI**: HCP 7T, 4 runs, TR = 1 s, 59k grayordinate surface (cortex only)
- **Preprocessing**: GSR + z-score per run; full runs saved (no timing filtering)
- **Timing extraction**: done at analysis time with `--delay-sec 5.0` (boxcar) or `--hrf`
- **Embeddings**: `{EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{v,a,av}.npy`
- **Outputs**: `{OUTPUT_DIR}/{subject_or_group_average}/{model}/{config_label}/`

## Environments

```bash
conda env create -f rsa/environment.yml          # analysis
conda env create -f cf_modeling/environment.yml  # cfmod
conda env create -f notebooks/feature_extraction/environment.yml  # avtransformer
```
