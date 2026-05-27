# Encoding — Ridge Encoding Models

Fits ridge regression models (himalaya RidgeCV, SVD solver) predicting cortical fMRI responses from model embeddings. Alpha is selected independently per vertex via leave-one-run-out cross-validation on the training set.

## GPU Acceleration

The encoding model uses himalaya `RidgeCV` with `torch_cuda` backend by default.
On an RTX 5070Ti (or any CUDA-capable GPU), the banded ridge fitting is offloaded
to GPU; the large Y matrix (59412 vertices × T_train) stays in CPU RAM via
`Y_in_cpu=True` to avoid GPU OOM.

Pearson r evaluation uses a fully vectorized numpy computation (no per-vertex loop),
replacing the previous O(n_vertices) scipy.stats.pearsonr calls.

Control via CLI: `--backend torch_cuda | torch | numpy`.
Control via run_analysis.sh: `BACKEND="torch_cuda"`.

## Structure

```
encoding/
├── run_analysis.sh          # master runner — all config lives here
├── environment.yml          # conda env: analysis
├── run_encoding.py          # per-model/modality encoding script
└── shared/
    └── encoding_utils.py    # fMRI + embedding array builders, CV helpers, model runner
```

## Usage

```bash
conda activate analysis
cd movie_watching   # run from repo root

bash encoding/run_analysis.sh                       # group-average, all models
bash encoding/run_analysis.sh persubject            # per-subject (GNU parallel)
bash encoding/run_analysis.sh persubject 4          # 4 parallel jobs
bash encoding/run_analysis.sh persubject 8 100610   # resume from subject 100610
```

## Configuration

All parameters are set in `run_analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `BIN_SEC` | 2.0 | Temporal bin size (seconds) |
| `DELAY_SEC` | 5.0 | Hemodynamic delay applied at analysis time |
| `HRF` | `false` | Convolve embeddings with SPM HRF instead of boxcar delay |
| `NORMALIZE` | `false` | Per-run z-score normalization of embeddings |
| `ALPHA_MIN` | -2 | Log10 of minimum ridge alpha |
| `ALPHA_MAX` | 9 | Log10 of maximum ridge alpha |
| `N_ALPHAS` | 23 | Number of alpha values on log scale |
| `CHUNK_SIZE` | 2000 | Vertices per batch for Pearson r computation |
| `TEST_VIDEO_IDS` | `video5,video9,video14,video18` | Held-out test videos |
| `MODELS` | (array) | Model registry |

## Train / Test Split

- **Test**: `video5`, `video9`, `video14`, `video18` (last video of each run, 82 TRs each)
- **Train**: remaining 14 videos
- **Z-scoring**: per run on training time bins; same mean/std applied to test bins from the same run to prevent data leakage
- **Alpha selection**: leave-one-run-out CV on training data (himalaya `PredefinedSplit`)
- **Metric**: Pearson r on held-out test set per vertex

## Input Modes

### Disk mode (default, `--preprocessed-dir`)

Reads pre-saved full-run CIFTIs produced by `preprocess_individual.py`:

```bash
python preprocess_individual.py \
    --raw-dir /path/to/raw_ciftis \
    --out-dir /path/to/preprocessed \
    --subjects-list subjects.txt \
    --sg-filter --psc --gsr \
    --save-individual
```

Output per subject:
- `{sub}_sg_psc_gsr_cortex_59k.dtseries.nii` — concatenated full-run CIFTI (4 runs)
- `{sub}_sg_psc_gsr_run_trs.npy` — TRs per run (needed for timing alignment)

Set `FMRI_SUFFIX="sg_psc_gsr"` in `run_analysis.sh` to match.

### Streaming mode (`--raw-dir`)

Preprocesses raw 7T CIFTIs on-the-fly via `preprocess_individual.preprocess_subject()`. No CIFTI is saved to disk.

## fMRI Preprocessing Convention

`preprocess_individual.py` applies per-run signal cleaning (SG→PSC→GSR) and concatenates all 4 runs into a single continuous CIFTI. **No timing filtering happens at preprocessing time.**

At analysis time, `run_encoding.py` loads `movie_timing.csv`, computes within-run onset TRs from global `onset_sec`, applies hemodynamic delay (`--delay-sec`), bins to `BIN_SEC` resolution, z-scores per run, and splits into train/test by video ID.

When `HRF=true`: set `--delay-sec 0` (no boxcar shift); embeddings are convolved with the SPM HRF inside `run_encoding.py`.

## Outputs

```
{OUTPUT_DIR}/{subject}/{model}/{config_label}/
    encoding_r_v.dscalar.nii     # Pearson r map — visual modality
    encoding_r_a.dscalar.nii     # Pearson r map — auditory modality
    encoding_r_av.dscalar.nii    # Pearson r map — audiovisual modality
```

`config_label` encodes parameters: `delay{D}s_nonorm_bin{B}s` or `hrf_norm_bin{B}s`.
