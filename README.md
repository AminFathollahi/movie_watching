# Movie-watching fMRI: representation analyses

Analysis code for HCP 7T movie-watching fMRI (175 subjects, 4 runs, TR = 1 s) compared against audio, video and joint audio-video model embeddings. The brain data are the 108,441 cortical grayordinates of the 59k fsLR surface (plus the subcortical voxels in `subcortical/`). Each movie clip is cut into fixed windows (1, 2 or 5 s), every window is embedded by each model, and the embeddings are compared with the delayed fMRI response of the same window.

Analyses:

- Representational similarity analysis (RSA): searchlight and parcel-wise correlation between the model's representational dissimilarity matrix (RDM) and the brain RDM, plus noise ceilings, crossnobis distance, partial RSA, temporal-scramble and dummy-modality controls.
- Banded ridge encoding: cross-validated prediction of vertex responses from audio, video and joint embeddings, and the variance partition between them.
- Connective-field (CF) modeling: banded ridge models predicting cortex from the Laplace-Beltrami eigenfunctions of two region subsurfaces (after Hedger et al. 2025).
- Clustering of stimulus states and of vertex time series.
- Seed-based connectivity from RSA-defined regions.

## Progress (October 2026)

The current question is where cortex combines auditory and visual information during movie watching, and whether audiovisual models capture that combination. Each 5-s window is embedded three ways: audio alone (A), video alone (V) and the joint audiovisual embedding (J). The audiovisual models are PE-AV (small) and Nemotron (layer 18). Their A and V are also replaced by eight pairs of separately trained audio and video encoders.

Changes in this round:

- Embeddings are normalized the same way as the fMRI responses (within each run when a whole run is held out, with training-set statistics when a single clip is held out), and all encoding models were refitted under both splits.
- The control encoders were refitted at the layer that best predicts cortex on its own.
- Encoding now reports the variance A and V explain in common, for a model's own A and V and for each control pair.
- Reducing each embedding to 32 or 128 principal components changed prediction by less than 0.003 and did not make the J maps more reliable, so the full embeddings are kept.

Results so far (group average of 175 subjects, mean held-out R² over the 108,441 cortical grayordinates, leave-one-run-out / leave-one-clip-out):

- A and V together reach R² of about 0.10 (PE-AV 0.100 / 0.099, Nemotron 0.119 / 0.117).
- Adding J to the model's own A and V changes R² by less than 0.002 (PE-AV +0.0008 / −0.0003, Nemotron +0.0017 / +0.0013). Added to separately trained encoders, Nemotron's J contributes 0.015 to 0.041, with similar maps under both splits.
- Separately trained encoders match PE-AV. The best pair (w2v-BERT 2.0 audio, V-JEPA 2 ViT-L video) reaches 0.106 / 0.099, against 0.101 / 0.098 for PE-AV with A, V and J. Nemotron (0.121 / 0.118) is ahead of every pair on average.
- The variance shared by A and V is no larger for PE-AV's own encoders than for the best separately trained pair (OpenBEATs audio, V-JEPA 2 ViT-L video).

- RSA now uses Euclidean and, separately, noise-normalized Mahalanobis RDMs, with semi-partial (Tong et al. 2022) and partial (Clarke & Tyler 2014) Spearman maps. Correlation distance is no longer used: it removes the response shared by a whole searchlight, and Walther et al. (2016) found cross-validated Mahalanobis distance preferable to it. Beyond PE-AV's own A and V, the joint RDM explains about nothing on average (semi-partial -0.011 Euclidean, 0.003 Mahalanobis), in line with encoding.

Per-pair numbers are in [`encoding/README.md`](encoding/README.md).

References: Clarke & Tyler (2014) *J Neurosci* 34:4766; Tong et al. (2022) *J Neurosci* 42(37); Walther et al. (2016) *NeuroImage* 137:188. Method references for each analysis are in the module READMEs.

## Repository layout

| Path | Contents |
|---|---|
| [`encoding/`](encoding/README.md) | Banded ridge encoding and variance partition (`analysis.sh`) |
| [`cf_modeling/`](cf_modeling/README.md) | Connective-field models and ROI-mean connectivity maps (`run_analysis.sh`); archived analyses are in `cf_modeling/deprecated/` |
| [`cluster/`](cluster/README.md) | Temporal-state and spatial-network clustering, vertex time-series clustering (`cluster.sh`, `run_vertex_clustering.sh`) |
| [`experimental/`](experimental/README.md) | Label validation: speech, face and body hand labels as the model in encoding |
| [`connectivity/`](connectivity/README.md) | Seed-based whole-cortex connectivity from RSA-derived border ROIs (`seed_connectivity.py`) |
| [`subcortical/`](subcortical/README.md) | Subcortical voxels of the raw CIFTIs: neighbor caches, noise ceiling and Workbench bundle (`analysis.sh`); their encoding is run by `encoding/analysis.sh`; outputs in `outputs/subcortical/` |
| [`notebooks/feature_extraction/`](notebooks/feature_extraction/README.md) | Stimulus segmentation and embedding extraction for every model |
| [`notebooks/visualization/`](notebooks/visualization/README.md) | Figure notebooks: layer sweeps, gradient and myelin comparison, embedding and RDM plots, connective-field and time-resolved figures |
| [`notebooks/RidgeEncodingForSpeechEnvMusic_BrainModel/`](notebooks/RidgeEncodingForSpeechEnvMusic_BrainModel) | Exploratory vertexwise ridge-encoding notebook (not used by the pipeline) |
| [`viz/`](viz/README.md) | Cortical map plotting, Workbench bundles, Lorax map conversion |
| [`tests/`](tests/README.md) | Unit tests |
| [`shared/`](shared) | `paths.py` (data, output, external-drive and model-weight locations), `cifti_io.py` (CIFTI helpers), `official_timing.py` (parser for the official HCP clip-timing sheet), and the building blocks the live analyses use: `binning.py` (binning of responses and embeddings; `rsa_utils.py` re-exports it for `cluster/`), `naming.py`, `model_registry.py`, `residuals.py`, `searchlight.py`, `noise_ceiling.py`, `glasser.py` |
| [`preprocessing/`](preprocessing) | `preprocess_individual.py` (per-run signal cleaning of raw CIFTIs; also importable for streaming use), `make_average.sh` (group-average midthickness and inflated surfaces from the subject list), `verify_subjects.py` (checks that every listed subject has all four raw CIFTIs and both midthickness surfaces) |
| [`deprecated/rsa/`](deprecated/rsa/README.md) | RSA pipeline (`analysis.sh`) and its validation, control and integration analyses; nothing outside `deprecated/` imports it, and its tests are in `deprecated/rsa/tests/` |
| [`deprecated/cka/`](deprecated/cka/README.md) | Searchlight centered kernel alignment, its comparison with RSA and encoding (`measure_comparison/`) and its tests; RSA and CKA are retired and not part of the current analyses |

## Environments

The `movie` environment runs everything except embedding extraction:

```bash
conda env create -f cf_modeling/environment.yml
conda activate movie
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"
```

Connective-field modeling imports `vicsompy` (Hedger et al. 2025) from source, because its pinned torch version conflicts with the one above:

```bash
git clone https://github.com/nicholashedger/vicsompy.git ../Vicarious_somatotopy
```

`VICSOMPY_REPO` or `--vicsompy-repo` point to another checkout.

Embedding extraction uses separate environments because the model repositories pin incompatible package versions:

| Environment | Definition | Used for |
|---|---|---|
| `avtransformer` | `notebooks/feature_extraction/environment.yml` | PE-AV, Nemotron, Omni3B, unimodal encoders, Whisper, InternVL, Gemma |
| `topo_omni` | `notebooks/feature_extraction/topo_omni_environment.yml` | Topo-Omni |
| `cav-mae-sync` | `notebooks/feature_extraction/cav_mae_sync_environment.yml` | CAV-MAE-Sync (needs a clone of the official repository) |
| `audiocaption` | `notebooks/feature_extraction/audiocaption_environment.yml` | CLAP-Cap audio captions and the LLM caption rewrite |

## Data and output locations

All locations are relative to the directory that contains the repository (`MOVIE_ROOT` overrides it): `data/`, `outputs/`, `external/` for data kept on a separate drive (a link to that drive's copy of the same tree), and `hf_models/` for model weights (`MOVIE_MODELS_DIR` overrides it). Python scripts read them from `shared/paths.py`, runner scripts from their `CONFIG` block.

| Item | Location |
|---|---|
| Raw CIFTIs, one file per subject and run (`{sub}_tfMRI_MOVIE{1-4}_7T_{AP\|PA}_Atlas_1.6mm_hp2000_clean.dtseries.nii`) | `external/data/individual-59k/` |
| Subject list (one ID per line, `#` comments) | `data/subjects.txt` |
| Per-subject midthickness and inflated surfaces | `data/midthickness_1.6/`, `data/Inflated_1.6/` |
| Group-average surfaces (`CohortAvg.{L,R}.{midthickness,inflated}_MSMAll.59k_fs_LR.surf.gii`) and Glasser parcellation | `data/HCP_S1200_GroupAvg_v1/` |
| Analysis clip timing: 18 clips, global `onset_sec`, `end_sec`, `duration_sec`, `run_id` | `data/movie_timing.csv` |
| Official HCP clip timing (run-local, includes 20 s rest blocks) | `data/HCP_7T_Movie_Clip_Timing.csv` |
| Full run movies and the cut stimulus windows | `data/segmented_stimulus/{full,filtered}/` |
| Preprocessed per-subject CIFTIs | `data/preprocessed/{flag}/{sub}_{flag}_cortex_59k.dtseries.nii` |
| Preprocessed group average and run lengths | `data/preprocessed/average_sub/{flag}/group_average_{flag}_cortex_59k.dtseries.nii`, `group_average_{flag}_run_trs.npy` |
| Embeddings | `outputs/model_embeddings/{model}/bin{B}s_skip{B}s/{model}_{a\|v\|av}.npy` |
| Results | `outputs/{rsa,encoding,cf_modeling,cluster,connectivity,subcortical}/`; subcortical encoding in `outputs/subcortical/encoding/` |

`{flag}` names the preprocessing steps applied, joined by underscores: `sg`, `psc`, `gsr`, or `raw` when none. The `encoding/`, `cluster/` and `connectivity/` runners default to `raw`; `cf_modeling/` defaults to `sg_psc`. `average_sub/raw/group_average_raw_cortex_59k.dtseries.nii` is also the grayordinate template for the other runners.

RSA and encoding write, per configuration and for the group average as for each subject, one multi-map CIFTI per kind of map: `models`, `semipartial` and `partial` for RSA, `models` for encoding, from which the collect step builds one file per partition quantity (`unique_j_{J}_{scaling}.dscalar.nii`, `shared_av_{scaling}.dscalar.nii`, …, the pairs as maps). Significance maps and masks go into the file of the map they test. RSA names carry the brain RDM and the comparator: `pearson_spearman` for the correlation distance `1 - r`, `euclidean_spearman`, `mahalanobis_spearman`, `crossnobis_spearman`, `cosine_kendalltau`.

Encoding fits of PE-AV and Nemotron share `outputs/encoding/group_average/partitions/delay{D}s_bin{B}s_skip{B}s/` (the joint model, `peav` or `nemotron`, is in the file name), the time-resolved files are in `…/group_average/time_resolved/delay{D}s_bin{B}s_skip{B}s/`, and the subcortical counterparts are under `outputs/subcortical/encoding/group_average/` (`encoding/README.md`).

Embedding rows are ordered by clip (1 to 18), then by window start time. Window width `B` and stride `S` appear in the directory name as `bin{B}s_skip{S}s`; the pipeline default is `S = B` (non-overlapping).

## Pipeline order

1. Check inputs. `python preprocessing/verify_subjects.py` reports subjects with missing CIFTIs or surfaces.
2. Preprocess fMRI. Each of `encoding/analysis.sh`, `cf_modeling/run_analysis.sh` and `deprecated/rsa/analysis.sh` has a `preprocess` mode that runs `preprocessing/preprocess_individual.py` with its own `SG_FILTER`, `PSC` and `GSR` settings, saves per-subject and group-average CIFTIs and moves the group average to `average_sub/{flag}/`. Run it once with the `raw` setting at least, because the group average is the template and run-length source for every later step:
   ```bash
   conda activate movie
   bash encoding/analysis.sh preprocess
   ```
   Direct use of the script (the per-step flags are `--sg-filter`, `--psc`, `--gsr`, each with a `--no-` form; `--dry-run` processes one subject without saving):
   ```bash
   python preprocessing/preprocess_individual.py \
       --raw-dir "<dir with raw CIFTIs>" --out-dir data/preprocessed/sg_psc \
       --subjects-list data/subjects.txt --sg-filter --psc --no-gsr \
       --save-individual --save-average
   ```
   Steps run per run in the order Savitzky-Golay high-pass (window 201 TRs, order 3), percent signal change, global signal regression. Percent signal change divides by the temporal mean taken before the high-pass step, because the filter removes the mean. Without `--save-individual` and `--save-average` nothing is written; the RSA runner uses that streaming mode (`STREAM=true`) to preprocess raw CIFTIs on the fly.
3. Cut the stimulus. `python notebooks/feature_extraction/segment_official.py` cuts the 18 clips and their 1, 2 and 5 s windows from the full run movies and writes `data/movie_timing.csv`. It reads `average_sub/raw/group_average_raw_run_trs.npy`, so step 2 comes first.
4. Extract embeddings. See [`notebooks/feature_extraction/README.md`](notebooks/feature_extraction/README.md).
5. Build surfaces. `bash preprocessing/make_average.sh` averages the per-subject surfaces listed in `subjects.txt` and writes `CohortAvg.*` surfaces to `data/GroupAverage_59k/`. The runners read them from `data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/`, so place or link them there. Then `bash deprecated/rsa/analysis.sh neighbors_avg` builds the geodesic neighbor cache.
6. Run the analyses (each runner documents its modes in its directory README):
   ```bash
   bash deprecated/rsa/analysis.sh avg                  # group-average RSA
   bash deprecated/rsa/analysis.sh persubject all 4     # per-subject RSA, 4 subjects in parallel, then group statistics and noise ceiling
   bash encoding/run_all.sh                  # encoding fits, time-resolved analysis and collected files; resumable
   bash cf_modeling/run_analysis.sh all
   bash cluster/cluster.sh groupaverage
   bash subcortical/analysis.sh neighbors    # once, before the other subcortical modes
   ```
   `connectivity/seed_connectivity.py` takes border files written by `deprecated/rsa/draw_rsa_borders.py`, so it runs after RSA.
7. Inspect results. `viz/plot_cortex_map.py` for single maps, `viz/prepare_wb_view_cortical.py` for a Workbench bundle, and the notebooks in `notebooks/visualization/`.

## Conventions

- `data/movie_timing.csv` is the analysis timing file: 18 clips of meaningful audiovisual content, global onsets across the four concatenated runs, in seconds (equal to TR indices). `data/HCP_7T_Movie_Clip_Timing.csv` is the official run-local sheet; `shared/official_timing.py` converts it to global time, and `connectivity/` uses it to find inter-clip rest blocks. The two sources differ at clip ends (for example the end credits of clip 1 are excluded from `movie_timing.csv`).
- The hemodynamic delay is applied at analysis time to the fMRI window, either as a boxcar shift (`--delay-sec`, default 5.0 s in the runners) or by convolving the model with the SPM hemodynamic response function (`--hrf`). Preprocessing does no timing selection.
- `data/subjects.txt` lists the 175 subjects used. Subjects with no individual midthickness surface are listed in `data/excluded.txt`.
