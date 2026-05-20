# Encoding — Ridge Encoding Models

Fits ridge regression models (himalaya RidgeCV, SVD solver) predicting cortical fMRI responses from model embeddings. Alpha is selected per vertex via leave-one-run-out cross-validation.

## Structure

```
encoding/
├── run_analysis.sh          # master runner (all config lives here)
├── environment.yml          # conda env: analysis
├── run_encoding.py          # per-model/modality encoding script
└── shared/
    └── encoding_utils.py    # fMRI + embedding array builders, CV helpers, model runner
```

## Usage

```bash
conda activate analysis
bash run_analysis.sh                       # group-average, all models
bash run_analysis.sh persubject            # per-subject (GNU parallel)
bash run_analysis.sh persubject 4          # 4 parallel jobs
bash run_analysis.sh persubject 8 100610   # resume from subject 100610
```

## Configuration

All parameters are set in `run_analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `BIN_SEC` | 2.0 | Temporal bin size (seconds) |
| `DELAY_SEC` | 5.0 | Hemodynamic delay for fMRI window shift |
| `HRF` | false | Convolve embeddings with SPM HRF instead |
| `NORMALIZE` | true | Z-score embeddings per run |
| `ALPHA_MIN` | -2 | Log10 of minimum ridge alpha |
| `ALPHA_MAX` | 9 | Log10 of maximum ridge alpha |
| `N_ALPHAS` | 23 | Number of alpha values |
| `CHUNK_SIZE` | 2000 | Vertices per batch |
| `TEST_VIDEO_IDS` | video5,9,14,18 | Held-out test videos |
| `MODELS` | (array) | Model registry — add new models here |

## Train / Test Split

- **Test**: `video5`, `video9`, `video14`, `video18` (last video of each run)
- **Train**: remaining 14 videos; z-scored per run; LORO-CV for alpha selection
- **Metric**: Pearson r on test set per vertex

## Outputs

```
{OUTPUT_DIR}/{subject}/{model}/{config_label}/
    encoding_r_{v,a,av}.dscalar.nii    # Pearson r map per modality
```

`config_label` encodes parameters, e.g. `delay5s_norm_bin2s`.

## Delay / HRF

When `HRF=false` (default): fMRI extraction window is shifted by `DELAY_SEC`; embeddings unchanged.  
When `HRF=true`: embeddings are convolved with SPM HRF; fMRI window is not shifted.
