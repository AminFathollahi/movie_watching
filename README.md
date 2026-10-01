# Movie-watching fMRI: representation analyses

Analysis code for HCP 7T movie-watching fMRI (175 subjects, 4 runs, TR = 1 s) compared against audio, video and joint audio-video model embeddings. The brain data are the 59,412 cortical grayordinates of the 59k fsLR surface (plus the subcortical voxels in `subcortical/`). Each movie clip is cut into fixed windows (1, 2 or 5 s), every window is embedded by each model, and the embeddings are compared with the delayed fMRI response of the same window.

Analyses:

- Representational similarity analysis (RSA): searchlight and parcel-wise correlation between the model's representational dissimilarity matrix (RDM) and the brain RDM, plus noise ceilings, crossnobis distance, partial RSA, temporal-scramble and dummy-modality controls.
- Banded ridge encoding: cross-validated prediction of vertex responses from audio, video and joint embeddings, and the variance partition between them.
- Connective-field (CF) modeling: banded ridge models predicting cortex from the Laplace-Beltrami eigenfunctions of two region subsurfaces (after Hedger et al. 2025).
- Clustering of stimulus states and of vertex time series.
- Seed-based connectivity from RSA-defined regions.

## Repository layout

| Path | Contents |
|---|---|
| [`rsa/`](rsa/README.md) | RSA pipeline (`analysis.sh`) and its validation, control and integration analyses |
| [`encoding/`](encoding/README.md) | Banded ridge encoding and variance partition (`analysis.sh`) |
| [`cka/`](cka/README.md) | Searchlight centered kernel alignment, cross-validated across subject groups (`analysis.sh`) |
| [`cf_modeling/`](cf_modeling/README.md) | Connective-field models and ROI-mean connectivity maps (`run_analysis.sh`); retired analyses are in `cf_modeling/deprecated/` |
| [`cluster/`](cluster/README.md) | Temporal-state and spatial-network clustering, vertex time-series clustering (`cluster.sh`, `run_vertex_clustering.sh`) |
| [`connectivity/`](connectivity/README.md) | Seed-based whole-cortex connectivity from RSA-derived border ROIs (`seed_connectivity.py`) |
| [`subcortical/`](subcortical/README.md) | RSA on the subcortical voxels of the raw CIFTIs (`analysis.sh`) |
| [`notebooks/feature_extraction/`](notebooks/feature_extraction/README.md) | Stimulus segmentation and embedding extraction for every model |
| [`notebooks/visualization/`](notebooks/visualization/README.md) | Result-comparison and RDM notebooks |
| [`notebooks/RidgeEncodingForSpeechEnvMusic_BrainModel/`](notebooks/RidgeEncodingForSpeechEnvMusic_BrainModel) | Exploratory vertexwise ridge-encoding notebook (not used by the pipeline) |
| [`viz/`](viz/README.md) | Cortical map plotting, Workbench bundles, Lorax map conversion |
| [`reports/`](reports/av_map_interpretation/README.md) | Lorax-map audit ([`av_map_interpretation`](reports/av_map_interpretation/README.md), [`lorax_reproduction`](reports/lorax_reproduction/README.md)), the RSA, CKA and encoding map comparison ([`measure_comparison`](reports/measure_comparison/README.md)) and a note on the scramble permutation test |
| [`tests/`](tests/README.md) | Unit tests |
| `preprocess_individual.py` | Per-run signal cleaning of raw CIFTIs; also importable for streaming use |
| `official_timing.py` | Parser for the official HCP clip-timing sheet (run-local seconds to global seconds) |
| `cifti_io.py` | CIFTI helpers shared across directories |
| `make_average.sh` | Group-average midthickness and inflated surfaces from the subject list |
| `verify_subjects.py` | Checks that every listed subject has all four raw CIFTIs and both midthickness surfaces |
| `SETUP.md` | Environment and path setup |
| `PROJECT_OVERVIEW.md` | Scientific scope and where each result lives |

## Environments

The `movie` environment runs everything except embedding extraction. Set it up as described in [`SETUP.md`](SETUP.md):

```bash
conda env create -f cf_modeling/environment.yml
conda activate movie
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"
```

Embedding extraction uses separate environments because the model repositories pin incompatible package versions:

| Environment | Definition | Used for |
|---|---|---|
| `avtransformer` | `notebooks/feature_extraction/environment.yml` | PE-AV, Nemotron, Omni3B, unimodal encoders, Whisper, InternVL, Gemma |
| `topo_omni` | `notebooks/feature_extraction/topo_omni_environment.yml` | Topo-Omni |
| `cav-mae-sync` | `notebooks/feature_extraction/cav_mae_sync_environment.yml` | CAV-MAE-Sync (needs a clone of the official repository) |
| `audiocaption` | `notebooks/feature_extraction/audiocaption_environment.yml` | CLAP-Cap audio captions and the LLM caption rewrite |

## Data and output locations

Paths are set in the `CONFIG` block at the top of each runner script. Defaults:

| Item | Default location |
|---|---|
| Data root | `/home/amin/Research/Representation/Movie/data` |
| Output root | `/home/amin/Research/Representation/Movie/outputs` |
| Raw CIFTIs, one file per subject and run (`{sub}_tfMRI_MOVIE{1-4}_7T_{AP\|PA}_Atlas_1.6mm_hp2000_clean.dtseries.nii`) | `individual-59k/` on the external drive |
| Subject list (one ID per line, `#` comments) | `data/subjects.txt` |
| Per-subject midthickness and inflated surfaces | `data/midthickness_1.6/`, `data/Inflated_1.6/` |
| Group-average surfaces (`CohortAvg.{L,R}.{midthickness,inflated}_MSMAll.59k_fs_LR.surf.gii`) and Glasser parcellation | `data/HCP_S1200_GroupAvg_v1/` |
| Analysis clip timing: 18 clips, global `onset_sec`, `end_sec`, `duration_sec`, `run_id` | `data/movie_timing.csv` |
| Official HCP clip timing (run-local, includes 20 s rest blocks) | `data/HCP_7T_Movie_Clip_Timing.csv` |
| Full run movies and the cut stimulus windows | `data/segmented_stimulus/{full,filtered}/` |
| Preprocessed per-subject CIFTIs | `data/preprocessed/{flag}/{sub}_{flag}_cortex_59k.dtseries.nii` |
| Preprocessed group average and run lengths | `data/preprocessed/average_sub/{flag}/group_average_{flag}_cortex_59k.dtseries.nii`, `group_average_{flag}_run_trs.npy` |
| Embeddings | `outputs/model_embeddings/{model}/bin{B}s_skip{B}s/{model}_{a\|v\|av}.npy` |
| Results | `outputs/{rsa,encoding,cf_modeling,cluster,connectivity,subcortical}/` |

`{flag}` names the preprocessing steps applied, joined by underscores: `sg`, `psc`, `gsr`, or `raw` when none. The `rsa/`, `encoding/`, `cluster/` and `connectivity/` runners default to `raw`; `cf_modeling/` defaults to `sg_psc`. `average_sub/raw/group_average_raw_cortex_59k.dtseries.nii` is also the grayordinate template for the other runners.

Embedding rows are ordered by clip (1 to 18), then by window start time. Window width `B` and stride `S` appear in the directory name as `bin{B}s_skip{S}s`; the pipeline default is `S = B` (non-overlapping).

## Pipeline order

1. **Check inputs.** `python verify_subjects.py` reports subjects with missing CIFTIs or surfaces.
2. **Preprocess fMRI.** Each of `rsa/analysis.sh`, `encoding/analysis.sh` and `cf_modeling/run_analysis.sh` has a `preprocess` mode that runs `preprocess_individual.py` with its own `SG_FILTER`, `PSC` and `GSR` settings, saves per-subject and group-average CIFTIs and moves the group average to `average_sub/{flag}/`. Run it once with the `raw` setting at least, because the group average is the template and run-length source for every later step:
   ```bash
   conda activate movie
   bash rsa/analysis.sh preprocess
   ```
   Direct use of the script (the per-step flags are `--sg-filter`, `--psc`, `--gsr`, each with a `--no-` form; `--dry-run` processes one subject without saving):
   ```bash
   python preprocess_individual.py \
       --raw-dir "<dir with raw CIFTIs>" --out-dir data/preprocessed/sg_psc \
       --subjects-list data/subjects.txt --sg-filter --psc --no-gsr \
       --save-individual --save-average
   ```
   Steps run per run in the order Savitzky-Golay high-pass (window 201 TRs, order 3), percent signal change, global signal regression. Percent signal change divides by the temporal mean taken before the high-pass step, because the filter removes the mean. Without `--save-individual` and `--save-average` nothing is written; the RSA runner uses that streaming mode (`STREAM=true`) to preprocess raw CIFTIs on the fly.
3. **Cut the stimulus.** `python notebooks/feature_extraction/segment_official.py` cuts the 18 clips and their 1, 2 and 5 s windows from the full run movies and writes `data/movie_timing.csv`. It reads `average_sub/raw/group_average_raw_run_trs.npy`, so step 2 comes first.
4. **Extract embeddings.** See [`notebooks/feature_extraction/README.md`](notebooks/feature_extraction/README.md).
5. **Build surfaces.** `bash make_average.sh` averages the per-subject surfaces listed in `subjects.txt` and writes `CohortAvg.*` surfaces to `data/GroupAverage_59k/`. The runners read them from `data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/`, so place or link them there. Then `bash rsa/analysis.sh neighbors_avg` builds the geodesic neighbor cache.
6. **Run the analyses** (each runner documents its modes in its directory README):
   ```bash
   bash rsa/analysis.sh avg                  # group-average RSA
   bash rsa/analysis.sh persubject all 4     # per-subject RSA, 4 subjects in parallel, then group statistics and noise ceiling
   bash encoding/analysis.sh variance_partition
   bash cf_modeling/run_analysis.sh all
   bash cluster/cluster.sh groupaverage
   bash subcortical/analysis.sh neighbors    # once, before the other subcortical modes
   ```
   `connectivity/seed_connectivity.py` takes border files written by `rsa/draw_rsa_borders.py`, so it runs after RSA.
7. **Inspect results.** `viz/plot_cortex_map.py` for single maps, `viz/prepare_wb_view_cortical.py` for a Workbench bundle, and the notebooks in `notebooks/visualization/`.

## Conventions

- **Timing.** `data/movie_timing.csv` is the analysis timing file: 18 clips of meaningful audiovisual content, global onsets across the four concatenated runs, in seconds (equal to TR indices). `data/HCP_7T_Movie_Clip_Timing.csv` is the official run-local sheet; `official_timing.py` converts it to global time, and `connectivity/` uses it to find inter-clip rest blocks. The two sources differ at clip ends (for example the end credits of clip 1 are excluded from `movie_timing.csv`).
- **Hemodynamic delay.** Applied at analysis time to the fMRI window, either as a boxcar shift (`--delay-sec`, default 5.0 s in the runners) or by convolving the model with the SPM hemodynamic response function (`--hrf`). Preprocessing does no timing selection.
- **Subjects.** `data/subjects.txt` lists the 175 subjects used. Subjects with no individual midthickness surface are listed in `data/excluded.txt`.
