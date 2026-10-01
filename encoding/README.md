# Encoding: banded-ridge variance partition of audio, video and joint embeddings

Predicts cortical responses of the 7T HCP movie-watching data from model embeddings and partitions the held-out variance explained among three feature spaces:

- **A**: embedding of the clip's audio alone.
- **V**: embedding of the clip's video alone.
- **J**: the model's joint audiovisual embedding of the same clip.

An encoding model is a regression from an embedding to the response of each cortical grayordinate (a surface vertex of the cortical template, 108,441 in total, medial wall excluded). If it predicts the responses of clips it was not fitted on, the embedding carries information that the cortex also carries. One analysis, `variance_partition.py`, answers two questions:

1. **How well does each embedding predict each grayordinate?** The single-band models A, V and J are three of the seven models fitted below; there is no separate plain-encoding script.
2. **Where does J add to A and V?** With A and V already in the model, how much extra held-out variance does J explain (the unique-J region of the partition)?

## Method

### Stimulus, timing and embeddings

| Item | Location | Content |
|------|----------|---------|
| Full run movies | `data/segmented_stimulus/full/7T_MOVIE{1..4}_*.mp4` and `*_audio.wav` | Whole run, including the 20 s rest blocks. `data/segmented_stimulus` is a link to the external drive |
| Official timing | `data/HCP_7T_Movie_Clip_Timing.csv` (parsed by `official_timing.py`) | Start, end and duration of every clip and rest block, in run-local seconds |
| Clip windows | `data/segmented_stimulus/filtered/` | `Video{N}/Video{N}_chunks_{B}s`, `Audio{N}/Audio{N}_chunks_{B}s`, `Video{N}/Video{N}_av_chunks_{B}s` for B = 1, 2, 5 s |
| Timing table | `data/movie_timing.csv` | One row per clip (18): `video_id`, `onset_sec` (global: run offset plus run-local start), `end_sec`, `duration_sec`, `run_id` |
| Embeddings | `outputs/model_embeddings/{model}/bin{B}s_skip{B}s/{model}_{a,v,av}.npy` | One row per window, clips in timing order, windows in time order. `_a` is A, `_v` is V, `_av` is J |
| Responses | `data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii` and `group_average_raw_run_trs.npy` | Group-average (175 subjects, `data/subjects.txt`) continuous response of the four runs concatenated, repetition time 1 s, no Savitzky-Golay filter, percent-signal-change or global-signal regression (`raw`). The file name says `59k`; the file holds 108,441 grayordinates. `run_trs.npy` holds the number of repetition times in each run |

`notebooks/feature_extraction/segment_official.py` writes the clip windows and `data/movie_timing.csv`. It cuts each clip from the full run movie at the official start time and duration, snapped to the movie's 24 frames per second grid. Only the rest blocks and the end credits of `video1` (cut at frame 5579) are removed. Window i of a clip starts at the clip start plus i times B seconds; the last partial window is dropped. A clip of duration D yields floor((D − B) / skip) + 1 windows.

The last clip of each run (`video5`, `video9`, `video14`, `video18`) is the same 83.375 s stimulus, shown once per run. These four repeated clips are the held-out set of the `fixed` split and are dropped from the `runwise` split.

Embedding extraction (environment `avtransformer`; each script header gives the exact command):

| Script | Writes |
|--------|--------|
| `notebooks/feature_extraction/pe_av_extract_intact.py --unimodal own` (default) | `pe-av-small-16-frame_{a,v,av}.npy`: A is `audio_embeds` and V is `video_embeds` of the joint forward call (audio tower and video tower alone), J is `audio_video_embeds` |
| `pe_av_extract_intact.py --unimodal dummy` | `pe-av-small-16-frame_dummy_av_{a,v}.npy` (definition under Tags) |
| `notebooks/feature_extraction/nemotron_extract_intact.py` | `nemotron_layer{N}_{mp,lt}_{a,v,av}.npy` for layers 9, 18, 27, 35, 36; `mp` is mean pooling, `lt` last token. A and V are separate audio-only and video-only passes through the whole model, pooled at the same layer as J |

Environment variables of the extraction scripts: `STIMULUS_DIR`, `EMBEDDINGS_BASE`, `BIN_SEC`, `SKIP_SEC` (set equal to `BIN_SEC` for the encoding layout; the PE-AV default is 1), `BATCH_SIZE` (PE-AV), `MODELS_HOME` (Nemotron).

### Responses

For a clip with global onset t0 in run r (run start s_r), the response of window i is the mean of the preprocessed signal over round(B / TR) consecutive repetition times starting at round((t0 − s_r + delay) / TR) + i · round(skip / TR), where `delay` is 5 s (a boxcar shift standing in for the hemodynamic lag), B is the window length, skip the stride and TR the repetition time. Rows of the response matrix Y have shape (windows, 108441) and align one to one with the embedding rows after the repeated clips are separated.

Shape example at B = skip = 5 s: 626 windows over all clips (562 outside the repeated clips: 135, 145, 140 and 142 in runs 1 to 4; 64 in the repeated clips). At 2 s: 1583 windows (164 repeated); at 1 s: 3175 (332 repeated). Embedding dimension: 1024 for PE-AV, 2048 for Nemotron.

`--hrf` additionally convolves each clip's embeddings with the SPM canonical hemodynamic response function (off by default).

### Banded ridge

For a set of feature bands such as A and V, the prediction is Ŷ = A·W_A + V·W_V, with W_A and W_V of shape (embedding dimension, 108441). The weights minimize the squared error ‖Y − A·W_A − V·W_V‖² plus a separate squared-norm penalty on each band (banded ridge, himalaya `GroupRidgeCV(groups="input", solver="random_search")`). Each grayordinate has its own penalties, chosen by inner cross-validation on the training rows only: `--n-iter` (20) random draws of the relative band scalings, each combined with `--n-alphas` (23) overall penalties `logspace(--alpha-min, --alpha-max)` (default `logspace(-2, 9, 23)`); for each grayordinate the setting with the best score on the held-out training run is kept. The training mean of Y is subtracted before the fit and added back to the prediction (`fit_intercept=False`). `--model-random-state` (0) seeds the random draws.

### Splits

| Split | Test rows | Training rows | Inner cross-validation | Outer fits |
|-------|-----------|---------------|------------------------|------------|
| `fixed` | the four repeated clips, concatenated and not averaged | all other clips | leave one run out over the four runs | 1 |
| `runwise` | one run, repeated clip excluded | the other three runs | leave one run out over the three training runs | 4 |

For `runwise`, the held-out predictions of the four folds are pooled before scoring. `fixed` follows the held-out validation design of Hedger et al. (2025). Which clips are held out and dropped is set by `--exclude-video-ids` (default the four repeated clips); `fixed` requires it to be non-empty.

### Normalization (after the split)

| Quantity | Rule | Statistics from |
|----------|------|-----------------|
| Embeddings, `zscore` (`StandardScaler` with mean and standard deviation, as in Hedger et al.) | subtract the per-feature mean, divide by the per-feature standard deviation | all training rows of the fold, pooled over runs; applied unchanged to the test rows |
| Embeddings, `center` (as in the Gallant-lab voxelwise tutorials) | subtract the per-feature mean only | the same training rows |
| Responses, training rows | z-score each grayordinate within each run | that run's training windows |
| Responses, held-out rows, `fixed` | z-score each grayordinate within each run | that run's repeated clip alone (its own mean and standard deviation) |
| Responses, held-out rows, `runwise` | z-score each grayordinate within the run | that run's retained windows |

Each run is z-scored separately because a grayordinate's mean and scale differ between runs for reasons unrelated to the stimulus. No test feature or test response enters a fitted quantity.

### Scoring

For every grayordinate, on the held-out rows (pooled over folds for `runwise`):

- R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², where ŷ is the prediction including the training mean and ȳ is the mean of the held-out responses. Negative values (worse than predicting the held-out mean) are kept. R² is undefined (NaN) where the held-out responses are constant.
- Pearson r between y and ŷ. R² penalizes offset and amplitude errors; r does not.

### The seven models and the partition

Three bands give seven non-empty subsets: A, V, J, AV, AJ, VJ, AVJ. Each is a separate banded ridge fitted on identical training rows with identical penalty search and seed, and scored on identical held-out rows. Each model's R² is treated as the size of the union of the variance sets its bands explain. With R²(S) the R² of subset S:

- unique J = R²(AVJ) − R²(AV); unique A = R²(AVJ) − R²(VJ); unique V = R²(AVJ) − R²(AJ)
- pairwise intersection I(X, Y) = R²(X) + R²(Y) − R²(XY)
- triple intersection T = R²(AVJ) − R²(A) − R²(V) − R²(J) + I(A,V) + I(A,J) + I(V,J)
- shared A and V only = I(A,V) − T; shared A and J only = I(A,J) − T; shared V and J only = I(V,J) − T; shared A, V and J = T

The seven regions sum to R²(AVJ), not to 1; the remainder 1 − R²(AVJ) is response the three bands do not explain. Regions can be negative where adding a band lowers held-out prediction. The overlaps are long chains of subtractions and are the least reliable. No partition of Pearson r is produced, because correlations are not additive.

## Running

Environment: conda `movie` (`encoding/environment.yml`; himalaya, torch, nibabel).

### `analysis.sh`

```bash
bash encoding/analysis.sh preprocess
bash encoding/analysis.sh variance_partition [--models M...] [--bins B...] [--variants split:scaling...] [--controls SET...|none|all]
bash encoding/analysis.sh screen [--audio-models M...] [--video-models M...] [--bins B...] [--variants split:scaling...]
bash encoding/analysis.sh factorial_interaction [--models M...] [--bins 5]
```

Each option takes a space-separated list. Defaults: `--models pe-av-small-16-frame nemotron_layer18_mp`, `--bins 5 2 1` (skip equals bin), `--variants fixed:center`, `--controls all`; `--audio-models` and `--video-models` default to empty. Variants are `split:scaling` with split `fixed` or `runwise` and scaling `zscore` or `center`. A failed fit is logged and the remaining fits continue; the script exits non-zero if any failed.

| Mode | Action |
|------|--------|
| `preprocess` | Runs `preprocess_individual.py --save-individual --save-average` on the subjects in `SUBJECTS_LIST`: per-subject files to `PREPROCESSED_INDIV_DIR`, group average to `PREPROCESSED_DIR` |
| `variance_partition` | For each model and bin, seven-model fits with J from the model: first with A and V from the model itself (tag `unimodal_own`), then once per matching `OWN_SETS` entry, then once per selected control set; each fit for every variant |
| `screen` | Single-feature ridge (tag `screen`) on each listed audio model (`--subsets a`) and video model (`--subsets v`); a model without `{model}_{a,v}.npy` at the bin is skipped and logged. For depth selection (below) use `--variants runwise:<scaling>` |
| `factorial_interaction` | Crossed-pair interaction control, 5 s bins only (pass `--bins 5`; other bins abort). For each model, each seed in the environment variable `PAIRING_SEEDS` (default `0`) and each run 1 to 4: extracts the interaction embedding with `pe_av_extract_scramble.py` or `nemotron_extract_scramble.py --factorial-run RUN --seed SEED` (environment `avtransformer`), then fits `runwise:zscore` with J taken per held-out run from `{model}_interaction_run{run}_seed{seed}`. The interaction embedding of a window is J(target video, target audio) + J(reference video, reference audio) − J(target video, reference audio) − J(reference video, target audio), with the reference windows drawn by `notebooks/feature_extraction/av_pairing.py`. A and V are the model's own, tag `unimodal_own`; output directory `outputs/encoding/factorial_interaction/group_average/{model}_interaction_seed{seed}/` |

### `variance_partition.py` directly

```bash
python encoding/variance_partition.py \
  --preprocessed-dir data/preprocessed/average_sub/raw --timing-csv data/movie_timing.csv \
  --embeddings-dir outputs/model_embeddings --output-dir outputs/encoding \
  --template-cifti data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \
  --model pe-av-small-16-frame --split fixed --feature-scaling center --tag unimodal_own
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--preprocessed-dir`, `--timing-csv`, `--embeddings-dir`, `--output-dir`, `--template-cifti` | required | Inputs and output root |
| `--subject`, `--fmri-suffix` | `group_average`, `raw` | Response file `{preprocessed-dir}/{subject}_{suffix}_cortex_59k.dtseries.nii` and `..._run_trs.npy` |
| `--model` | none | Model supplying J, and A and V unless overridden |
| `--audio-model`, `--video-model` | `--model` | Model supplying A, V |
| `--subsets` | all seven | Band subsets to fit (`a v j av aj vj avj`); bands not involved are not loaded. Partition files are written only when all seven are fit |
| `--split` | `runwise` | `fixed` or `runwise` |
| `--feature-scaling` | `center` | `zscore` or `center` |
| `--tag` | none | Inserted after the scaling in every output file name |
| `--output-name` | none | Output model directory name instead of the model name |
| `--joint-model-template` | none | J model name with a `{run}` placeholder, one J per held-out run (`runwise` only) |
| `--bin-sec`, `--skip-sec`, `--delay-sec`, `--tr` | 5, 5, 5, 1 | Window length, stride, response delay, repetition time (seconds) |
| `--hrf` | off | Convolve embeddings with the SPM hemodynamic response function |
| `--exclude-video-ids` | `video5,video9,video14,video18` | Held-out and dropped clips |
| `--alpha-min`, `--alpha-max`, `--n-alphas` | -2, 9, 23 | Overall penalty grid `logspace(min, max, n)` |
| `--n-iter` | 20 | Random draws of band scalings |
| `--backend` | `torch_cuda` | himalaya backend; falls back to `torch` when CUDA is unavailable |
| `--model-random-state` | 0 | Seed of the random search |

Embeddings whose model name contains `avscramble` are rejected (they mix training and held-out clips). Fits are deterministic up to GPU non-determinism.

### Differences against dummy-modality conditions

`run_diff_study.sh [partition|consolidate|all]` (default `partition`) fits and then contrasts intact joint embeddings against dummy-modality conditions.

- `partition`: for every base model in `BASE_MODELS` (PE-AV; Nemotron layers 9, 18, 27, 36; omni3b layers 9, 18, 27; topoomni layers 9, 18, 27, and their `sheet` variants; `_mp` pooling) and for the intact condition and each of `clsav_from_a` (dummy video, real audio) and `clsav_from_v` (dummy audio, real video): seven-model fit, 5 s bins, tag `unimodal_own`, all four split by scaling variants. A and V come from the base model; J comes from `{base}_{condition}` (its `_av.npy`, written by `notebooks/feature_extraction/{pe_av,nemotron,omni3b,topo_omni}_extract_dummy_modality.py`). Output directory `outputs/encoding/group_average/{base}_{condition}/`.
- `consolidate`: runs `diff_maps.py`, which for each model and variant writes (maps merged into existing files by `cifti_io.merge_into_combined`; missing inputs are skipped and logged):
  - `{base}_{condition}/delay5s_bin5s_skip5s/encoding_diff_{split}_{scaling}_unimodal_own_dummy.dscalar.nii` with maps `joint_r_dummy` (r_j of the dummy condition), `unique_j_dummy` and `diff_joint_r` = r_j(intact) − r_j(dummy);
  - `{base}/delay5s_bin5s_skip5s/encoding_diff_{split}_{scaling}_unimodal_own_modality_presence.dscalar.nii` with map `modality_presence_diff` = unique J(intact) − max over conditions of unique J(dummy).
- Environment overrides: `DIFF_STUDY_MODELS`, `DIFF_STUDY_CONDITIONS` (space-separated subsets of the lists above).

## Configuration

Set in the CONFIG block of `analysis.sh`:

| Variable | Value | Meaning |
|----------|-------|---------|
| `DATA_BASE`, `OUTPUTS_BASE` | `.../Movie/data`, `.../Movie/outputs` | Roots; `OUTPUT_DIR` is `OUTPUTS_BASE/encoding` |
| `SG_FILTER`, `PSC`, `GSR` | `false` | Preprocessing flags; all false gives the `raw` directory |
| `DELAY_SEC`, `TR` | 5.0, 1.0 | Response delay and repetition time |
| `PARTITION_N_ITER`, `PARTITION_MODEL_RANDOM_STATE` | 20, 0 | Passed as `--n-iter`, `--model-random-state` |
| `TIMING_CSV`, `EMBEDDINGS_DIR`, `TEMPLATE_CIFTI`, `SUBJECTS_LIST` | see script | Inputs |
| `MODELS`, `BINS`, `VARIANTS`, `CONTROLS` | see Defaults | Overridden by the command-line options |
| `OWN_SETS` | one entry | `"model tag audio_model video_model"`; extra fits with A and V from the entry's models and J from `model`, tagged `tag`. Entry: `pe-av-small-16-frame dummy_av pe-av-small-16-frame_dummy_av pe-av-small-16-frame_dummy_av` |
| `CONTROL_SETS` | eight entries | `"name audio_model video_model"`; the name is the output tag. Fits use cross-model A and V with J unchanged (table below) |
| `PAIRING_SEEDS`, `MOVIE_SEGMENTED_DIR` | `0`, filtered stimulus directory | Environment variables for `factorial_interaction` |
| `CONDA_ENV` | `movie` | Environment for `conda run` |

`--backend` is not set by `analysis.sh`; `variance_partition.py` uses its default.

### Control sets

`-d50` and `-d75` name the encoder layer at 50% and 75% of the model's depth; the plain name is the final output. Each model's depth is the one with the highest `runwise` 5 s `screen` R², averaged over the 5% of grayordinates where the model family's best depth predicts best. `runwise` never uses the four repeated clips of `fixed`, so the choice does not touch the `fixed` test set.

| Set (tag) | Audio | Video | Pairing |
|-----------|-------|-------|---------|
| `mae` | dasheng-0.6b-d75 (layer 24 of 32) | videomaev2-large-d75 (layer 18 of 24) | both masked autoencoders |
| `mae-large` | dasheng-1.2b-d75 (layer 30 of 40) | videomaev2-giant-d50 (layer 20 of 40) | both masked autoencoders, larger |
| `latent` | openbeats-large-i2-d75 (layer 18 of 24) | vjepa2-vitl (final, 24 of 24) | both predict masked content in a learned target space |
| `large-mixed` | dasheng-1.2b-d75 (layer 30 of 40) | vjepa2-vitg (final, 40 of 40) | size-matched at about 1B parameters, different objectives |
| `speech-wavlm` | wavlm-large-d75 (layer 18 of 24) | vjepa2-vitl (final, 24 of 24) | speech self-supervised audio with a fixed video model |
| `speech-w2vbert` | w2v-bert-2.0-d75 (layer 18 of 24) | vjepa2-vitl (final, 24 of 24) | speech self-supervised audio with a fixed video model |
| `text-contrastive` | clap-larger | pe-core-l14 | both contrastive with text |
| `text-asr` | whisper-large-v3 (final, 32 of 32) | pe-core-l14 | text-supervised speech recognition with a contrastive image-text model |

## Outputs

```
outputs/encoding/{subject}/{model}/delay{D}s_bin{B}s_skip{S}s/
```

`subject` is `group_average`. `{model}` is the J model, except `screen` files, which sit in the directory of the single feature model. Delay, bin and skip are written as integers.

File naming rule: `{quantity}_{metric}_{split}_{scaling}_{tag}_{content}`. `{split}` is `fixed` or `runwise`, `{scaling}` is `zscore` or `center`, and `{tag}` names the definition of A and V. Every `analysis.sh` mode passes a tag (`variance_partition.py` omits it only when called directly without `--tag`). Example: `encoding_r2_fixed_center_unimodal_own_models.dscalar.nii`.

| File | Contents |
|------|----------|
| `encoding_r2_{split}_{scaling}_{tag}_models.dscalar.nii` | Maps `r2_a, r2_v, r2_j, r2_av, r2_aj, r2_vj, r2_avj`: held-out R² of each fitted subset |
| `encoding_pearson_r_{split}_{scaling}_{tag}_models.dscalar.nii` | Maps `r_a ... r_avj`: held-out Pearson correlation of the same pooled predictions |
| `encoding_r2_{split}_{scaling}_{tag}_partition.dscalar.nii` | Ten maps, only when all seven subsets are fit: `unique_a`, `unique_v`, `unique_j`; `gain_j_over_a` = R²(AJ) − R²(A); `gain_j_over_v` = R²(VJ) − R²(V); `j_minus_av` = R²(J) − R²(AV); overlaps `shared_av_only`, `shared_aj_only`, `shared_vj_only`, `shared_avj` |
| `encoding_r2_{split}_{scaling}_{tag}_per_fold.npz` | `folds` (labels); per subset, `{subset}` = R² per outer fold (folds by grayordinates); `alphas_{subset}` and `deltas_{subset}`: selected overall penalties and himalaya band scalings per fold and grayordinate; for `fixed` also `clips` (the repeated clips) and `clip_r2_{subset}` (clips by grayordinates): R² of each subset on one clip's rows alone |
| `encoding_r2_{split}_{scaling}_{tag}_provenance.json` | Split, scaling, folds, `n_train_pool`, `n_test_rows`, repeated clip ids, audio, video and joint model names, tag, subsets, metric and normalization descriptions, `n_iter`, `model_random_state`, penalty grid |
| `encoding_diff_{split}_{scaling}_unimodal_own_{dummy,modality_presence}.dscalar.nii` | From `diff_maps.py` (see above) |

For `screen` (`--subsets a` or `v`), files are `encoding_{r2,pearson_r}_{split}_{scaling}_screen_models.dscalar.nii` (one map: `r2_a` / `r_a` or `r2_v` / `r_v`), `..._screen_per_fold.npz` and `..._screen_provenance.json`, in the feature model's directory; no partition is written.

### Tags

J is always the fitted model's joint embedding (PE-AV `audio_video_embeds`; Nemotron joint pass) in the J model's own directory; the tag changes only A and V.

| Tag | A | V | Produced by |
|-----|---|---|-------------|
| `unimodal_own` | the model's own audio-only representation (PE-AV: `audio_embeds`; Nemotron: audio-only pass through the whole model, mean-pooled at the same layer as J) | the model's own video-only representation (PE-AV: `video_embeds`; Nemotron: video-only pass, same pooling and layer) | every model, from its own `{model}_{a,v}.npy` |
| `dummy_av` | `audio_video_embeds` of (real audio, blank video). Blank video is the processor output for all-black frames: every value of `pixel_values_videos` set to -1.0 (pixel 0 after rescaling by 1/255 and normalizing with mean 0.5, standard deviation 0.5), shape and padding mask unchanged | `audio_video_embeds` of (silent audio, real video); silent audio is `zeros_like(input_values)` with the real padding mask | `OWN_SETS` entry (PE-AV); embeddings from `pe_av_extract_intact.py --unimodal dummy` in model directory `pe-av-small-16-frame_dummy_av` |
| `mae`, `mae-large`, `latent`, `large-mixed`, `speech-wavlm`, `speech-w2vbert`, `text-contrastive`, `text-asr` | embeddings of the set's audio model | embeddings of the set's video model | `CONTROL_SETS` |
| `screen` | one model's embedding alone (`--subsets a`) | one model's embedding alone (`--subsets v`) | `analysis.sh screen`; no J, no partition |

## Tests

```bash
python -m pytest tests/test_fold_evaluator.py tests/test_no_leakage.py tests/test_encoding.py tests/test_variance_partition.py
```

| File | Guards |
|------|--------|
| `tests/test_fold_evaluator.py` | R² keeps negative values and marks constant targets; partition regions sum to R²(AVJ) and match the unique definitions; a purely joint signal lands in unique J; fold masks of both splits; Pearson r against NumPy |
| `tests/test_no_leakage.py` | Perturbing test features or test responses leaves fitted quantities and predictions unchanged (both feature scalings); a residual fitted before the split is detected; global-scramble models are rejected; `center` subtracts the training mean without dividing; training-only feature statistics; `fixed` response normalization |
| `tests/test_encoding.py` | Response binning (`build_fmri_arrays`): shapes, run boundaries, bin totals |
| `tests/test_variance_partition.py` | A, V and J loaded from separate models; tag placement in file names; per-clip R²; single-feature screening outputs; the ten partition maps |

## Code

| File | Purpose |
|------|---------|
| `variance_partition.py` | Fits, partition and output writing |
| `diff_maps.py` | Intact versus dummy-modality contrasts |
| `analysis.sh`, `run_diff_study.sh` | Runners |
| `shared/` | `fold_evaluator.py` (banded ridge, standardization, partition, scoring), `splits.py` (argument parser, inputs, folds), `encoding_utils.py` (response binning, hemodynamic response function, CIFTI writing); see `shared/README.md` |

## Reference: two variance decompositions used in the Gallant lab

1. **Inclusion-exclusion on separate fits** (Lescroart et al. 2015; Deniz et al. 2019): what this directory computes. One model per band subset, then set arithmetic on held-out scores. The earlier papers applied it to signed r² (r·|r|); here it is applied to R².
2. **Split R² inside one joint banded fit** (Dupre la Tour et al. 2022, himalaya `r2_score_split`): one model on [A, V, J]; each band's prediction is scored separately and the allocation sums to the full model's R². It allocates the joint model's variance among bands, including cross-products, and is not the gain from removing a band. Not computed here.
