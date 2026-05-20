# RSA — Representational Similarity Analysis

Vertex-wise searchlight RSA and Glasser parcel RSA comparing cortical fMRI representations to model embeddings.

## Structure

```
rsa/
├── run_analysis.sh          # master runner (all config lives here)
├── environment.yml          # conda env: analysis
├── shared/
│   ├── rsa_utils.py         # fMRI extraction, embedding processing, RDM utilities
│   └── cifti_io.py          # CIFTI load/save helpers
├── searchlight/
│   └── run_searchlight.py   # geodesic neighbourhood RSA, outputs .dscalar.nii
└── glasser/
    └── run_glasser.py       # Glasser parcel RSA, outputs .dscalar.nii + ranked_report.csv
```

## Usage

```bash
conda activate analysis
bash run_analysis.sh                            # group-average, all methods
bash run_analysis.sh avg searchlight            # group-average, searchlight only
bash run_analysis.sh avg glasser                # group-average, Glasser only
bash run_analysis.sh persubject                 # per-subject, all methods (GNU parallel)
bash run_analysis.sh persubject all 4           # 4 parallel jobs
bash run_analysis.sh persubject all 8 100610    # resume from subject 100610
```

## Configuration

All parameters are set in `run_analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `BIN_SEC` | 2.0 | Temporal bin size (seconds) |
| `DELAY_SEC` | 5.0 | Hemodynamic delay for boxcar shift |
| `HRF` | false | Convolve embeddings with SPM HRF instead |
| `NORMALIZE` | true | Z-score embeddings per run |
| `METHOD` | spearman | RDM correlation method |
| `K` | 100 | Searchlight neighbourhood size |
| `MODELS` | (array) | Model registry — add new models here |

## Model Registry

```bash
MODELS=(
    "pe-av-small-16-frame:v,a,av"
    "pe-av-base:v,a,av"
    "audiomae:a"
    ...
)
```

Any model whose embedding file is absent is silently skipped. Add a new model by appending a line; analyses run automatically when embeddings are present.

## Outputs

```
{OUTPUT_DIR}/{subject}/{model}/{config_label}/
    left_rsa_spearman.dscalar.nii
    right_rsa_spearman.dscalar.nii
    left_rsa_pval.dscalar.nii
    right_rsa_pval.dscalar.nii
    stats_summary.txt
    ranked_report.csv          # Glasser only
```

`config_label` encodes analysis parameters, e.g. `k100_norm_delay5s_bin2s_spearman`.

## Delay / HRF

When `HRF=false` (default): fMRI extraction window is shifted by `DELAY_SEC`.  
When `HRF=true`: embeddings are convolved with SPM HRF; fMRI window is not shifted.
