# RSA — Representational Similarity Analysis

Vertex-wise searchlight RSA and Glasser parcel RSA comparing cortical fMRI representations to model embeddings from audiovisual transformer models.

## Structure

```
rsa/
├── run_analysis.sh          # master runner — all config lives here
├── environment.yml          # conda env: analysis
├── run_searchlight.py       # geodesic neighbourhood RSA → .dscalar.nii
├── run_glasser.py           # Glasser parcel RSA → .dscalar.nii + ranked_report.csv
├── run_group_stats.py       # aggregate per-subject maps → group stats CIFTI
├── shared/
│   ├── rsa_utils.py         # fMRI extraction, embedding processing, RDM utilities
│   └── cifti_io.py          # CIFTI load/save helpers
```

## Usage

```bash
conda activate analysis
cd movie_watching   # run from repo root

bash rsa/run_analysis.sh avg                         # group-average, all methods
bash rsa/run_analysis.sh avg searchlight             # group-average, searchlight only
bash rsa/run_analysis.sh avg glasser                 # group-average, Glasser only
bash rsa/run_analysis.sh persubject                  # per-subject, streaming (default)
bash rsa/run_analysis.sh persubject all 4            # 4 parallel jobs
bash rsa/run_analysis.sh persubject all 8 100610     # resume from subject 100610
bash rsa/run_analysis.sh groupstats                  # re-run group stats on existing maps
bash rsa/run_analysis.sh all                         # avg + persubject + groupstats
```

## Configuration

All parameters are set in `run_analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `STREAM` | `true` | Streaming mode — preprocess raw CIFTIs on-the-fly (no pre-saved CIFTI needed) |
| `SG_FILTER` | `true` | Savitzky-Golay high-pass filter (removes scanner drift) |
| `PSC` | `true` | Percent signal change normalization |
| `GSR` | `true` | Global signal regression |
| `BIN_SEC` | 2.0 | Temporal bin size (seconds) |
| `DELAY_SEC` | 5.0 | Hemodynamic delay — boxcar shift applied at analysis time |
| `HRF` | `false` | Convolve embeddings with SPM HRF instead of boxcar delay |
| `METHOD` | `spearman` | RDM correlation method |
| `K` | 150 | Searchlight neighbourhood size (vertices) |
| `MODELS` | (array) | Model registry — add new models here |
| `TIMING_CSV` | `data/movie_timing.csv` | Authoritative timing file |

## Streaming vs Disk Mode

**Streaming (default, `STREAM=true`):** Raw 7T CIFTIs are loaded and preprocessed on-the-fly (SG→PSC→GSR per run). No preprocessed CIFTI is saved. Movie segment extraction, hemodynamic delay, binning, and z-scoring all happen at analysis time using `movie_timing.csv`.

**Disk mode (`STREAM=false`):** Reads pre-saved CIFTIs from `PREPROCESSED_DIR`. Use after running `preprocess_individual.py --save-individual`. The preprocessing suffix (`FMRI_SUFFIX`) must match what `preprocess_individual.py` used.

**Template CIFTI:** `TEMPLATE_CIFTI` points to the `sg_psc` group-average CIFTI. It is used only for its BrainModelAxis (grayordinate structure) — it does not need to match per-subject preprocessing.

## fMRI Preprocessing Convention

- **Continuous full-run CIFTIs** — `preprocess_individual.py` applies SG→PSC→GSR per run and concatenates all 4 runs. No timing filtering is done during preprocessing.
- **Timing applied at analysis time** — `run_searchlight.py` and `run_glasser.py` load `movie_timing.csv`, compute within-run onset TRs from global `onset_sec`, apply hemodynamic delay, bin to `BIN_SEC` resolution, z-score per run (training stats applied to test), and split train/test.
- **Z-scoring** — per run on training time bins; same mean/std applied to test bins from the same run (no leakage).

## Model Registry

```bash
MODELS=(
    "pe-av-small-16-frame:av"
    # "pe-av-small-16-frame:v,a,av"
    # "audiomae:a"
    # "videomaev2-large:v"
)
```

Any model whose embedding file is absent is silently skipped. Add a new model by appending a line.

## Outputs

```
{OUTPUT_DIR}/{subject}/{model}/{config_label}/
    rsa_59k_{prep}_k{K}_delay{D}s_bin{B}_{method}_maps.dscalar.nii   # searchlight
    glasser_rsa_{prep}_delay{D}s_bin{B}_{method}_maps.dscalar.nii    # Glasser
    ranked_report.csv                                                   # Glasser only

{OUTPUT_DIR}/group_stats/{model}/{config_label}/
    group_stats_{N}subs.dscalar.nii    # 8-map CIFTI: mean_rho, cohens_d, sigmaps, clusters
    summary.json
```

`config_label` for searchlight: `k{K}_delay{D}s_bin{B}s_{method}` (e.g. `k150_delay5s_bin2s_spearman`)  
`config_label` for Glasser: `delay{D}s_bin{B}s_{method}`

## Subjects List

`data/subjects.txt` — one subject ID per line. Supports:
- Full-line comments: `# comment`
- Inline comments: `100610  # note`
- Blank lines

The parser strips all comments and blank lines before feeding subject IDs to GNU parallel.

## Resume / Skip

Re-running skips any subject/model whose output CIFTI already exists.  
Delete the output CIFTI to force a rerun for that subject.

## Silence GNU parallel citation notice (one-time)

```bash
parallel --citation
```
