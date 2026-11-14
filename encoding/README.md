# Encoding: banded-ridge variance partition

Voxelwise encoding models predict the fMRI response of each cortical grayordinate (108,441) or subcortical voxel from model embeddings of the same movie window. Three feature spaces are compared: **A**, an embedding of the window's audio; **V**, of its video; and **J**, the joint audiovisual embedding of a multimodal model (PE-AV or Nemotron). One banded ridge model is fitted for each of the seven non-empty subsets of {A, V, J}, and their held-out R² values are partitioned into unique and shared parts (Lescroart et al. 2015; de Heer et al. 2017).

## Method

- **Data.** Group-average responses of 175 HCP 7T subjects (4 runs, TR = 1 s). Each clip is cut into non-overlapping windows of B = 1, 2, 5 or 10 s; the response of a window is the mean signal over the B seconds starting 5 s after the window onset (hemodynamic delay).
- **Cross-validation.** Leave one run out (`--split loro`). The repeated clip that ends every run is kept once (its first showing) and its three later showings are excluded.
- **Scaling.** Responses and every embedding channel are z-scored within each run (`--feature-scaling zscore`; `demean` is a robustness variant). Embeddings get the same rows and statistics as the responses.
- **Model.** Banded ridge (himalaya `GroupRidgeCV`, Dupré la Tour et al. 2022): one penalty per band, chosen per grayordinate by inner leave-one-run-out cross-validation over 20 random draws of the band weights × 23 overall penalties.
- **Score.** R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)² on the pooled held-out windows; negative values are kept.
- **Partition.** With R²(S) the held-out R² of subset S:

| Quantity | Definition |
|---|---|
| `shared_av` | R²(A) + R²(V) − R²(AV) |
| `gain_a_over_v`, `gain_v_over_a` | R²(AV) − R²(V), R²(AV) − R²(A) |
| `gain_a_over_v_times_gain_v_over_a` | max(gain_a_over_v, 0) · max(gain_v_over_a, 0) |
| `unique_j` | R²(AVJ) − R²(AV) |
| `unique_a`, `unique_v` | R²(AVJ) − R²(VJ), R²(AVJ) − R²(AJ) |
| `shared_avj`, `shared_av_only`, `shared_aj_only`, `shared_vj_only` | the remaining regions of the three-set Venn diagram (inclusion–exclusion) |

## Feature sets

J is always the multimodal model's joint embedding. A and V come from a pair named `{audio}_{video}`. The audio and video encoders of a pair are matched in training objective and size, and each encoder appears in one pair only.

| Pair | Training objective | Audio | Video |
|---|---|---|---|
| `dasheng12b_videomaev2gd50` | masked autoencoding | Dasheng 1.2B | VideoMAE V2 giant (50 % depth) |
| `dasheng06bd75_videomaev2ld75` | masked autoencoding | Dasheng 0.6B (75 %) | VideoMAE V2 large (75 %) |
| `openbeatsli2_vjepa2ld75` | masked token / latent prediction | OpenBEATs large | V-JEPA 2 ViT-L (75 %) |
| `w2vbert2d75_vjepa2gd75` | masked token / latent prediction | w2v-BERT 2.0 (75 %) | V-JEPA 2 ViT-g (75 %) |
| `languagebinda_languagebindv` | contrastive with text | LanguageBind audio | LanguageBind video |
| `clap_viclip` | contrastive with text | CLAP | ViCLIP L/14 |
| `whisperl30_qwen25vl3bvitl32` | text-generative | Whisper large-v3 encoder (hidden state 30 of 32) | Qwen2.5-VL 3B vision encoder (hidden state 32 of 32, the last block) |
| `peavs16fa_peavs16fv` | own towers | PE-AV audio-only | PE-AV video-only |
| `nemotronl18mpcata_nemotronl18mpcatv` | own towers | Nemotron audio-only pass | Nemotron video-only pass |
| `peavs16fdummya_peavs16fdummyv` | dummy modality | PE-AV joint, blank video | PE-AV joint, silent audio |

The dummy pair is fitted for PE-AV only. In the Whisper/Qwen2.5-VL pair each encoder is taken at the hidden state of highest mean held-out Pearson r in its layer sweep (see Layer sweeps): hidden state 30 for Whisper, 32 (the last block, mean held-out r 0.243) for the Qwen2.5-VL vision encoder.

## Usage

Dependencies: `encoding/environment.yml`. Input and output roots: `shared/paths.py` (`MOVIE_ROOT`).

```bash
bash encoding/analysis.sh preprocess          # group-average responses
bash encoding/run_all.sh                      # all fits, time-resolved analysis, collected files
python encoding/variance_partition.py --help  # a single fit
```

## Running

`run_all.sh` calls three resumable runners in order: `run_variance_partition.sh` (pair fits), `run_time_resolved.sh` (time-resolved analysis) and `run_collect.sh` (multi-map files, including the layer sweeps). A unit of work is skipped when its output exists and was made with the requested settings, so after a stop or a reboot the same command carries on.

```bash
bash encoding/run_all.sh --status     # done / to run per runner, and the running PIDs
bash encoding/run_all.sh --dry-run    # what would run
setsid nohup bash encoding/run_all.sh >> run_all.log 2>&1 < /dev/null &   # detached
```

Stop with `kill <runner PID> <job PID>` (both printed by `--status`); start again to resume. Options (flags or environment variables): `--spaces`, `--bins`, `--scalings`, `--models`, `--n-iter`; `GPU_LOCK` (run each GPU job under `flock` on that file), `ON_OOM` (command run before retrying a job that ran out of GPU memory), `MIN_FREE_GB` (default 60). A failed job is retried once and then listed in `{runner}.failed.tsv`.

## Outputs

```
outputs/encoding/group_average/partitions/delay5s_bin{B}s_skip{B}s/        # cortex fits and collected files
outputs/encoding/group_average/time_resolved/delay5s_bin{B}s_skip{B}s/     # time-resolved files
outputs/encoding/group_average/layers/delay5s_bin{B}s_skip{B}s/            # layer sweeps
outputs/subcortical/encoding/group_average/...                             # the same for subcortex
```

`{J}` is `peav` or `nemotron`, `{scaling}` is `zscore` or `demean`. Multi-map files hold one map per pair, named by the pair.

| File | Contents |
|---|---|
| `encoding_{r2,pearson_r}_loro_{scaling}_{J}_{pair}_models.dscalar.nii` | one fit: `r2_a` … `r2_avj` (held-out R² of the seven subsets), `r_a` … `r_avj` |
| `encoding_r2_loro_{scaling}_{J}_{pair}_provenance.json`, `…_per_fold.npz` | settings and embedding sources of the fit; per-fold R², penalties and band scalings |
| `{quantity}_{scaling}.dscalar.nii` | `shared_av`, `gain_a_over_v`, `gain_v_over_a`, `gain_a_over_v_times_gain_v_over_a`: quantities without J, all pairs |
| `{quantity}_{J}_{scaling}.dscalar.nii` | `unique_j`, `unique_a`, `unique_v`, `shared_avj`, `shared_av_only`, `shared_aj_only`, `shared_vj_only`, `j_minus_av`, `gain_j_over_a`, `gain_j_over_v`, `own_avj_minus_av`, `own_shared_av_minus_shared_av`, `own_vs_controls`: quantities with J, all pairs |
| `av_bivariate_{r2,gain}_{scaling}.dlabel.nii` | colour maps, red = audio, blue = video (`r2`: R²(A) vs R²(V); `gain`: `gain_a_over_v` vs `gain_v_over_a`); white = both high; fixed limits 0.20 (`r2`) and 0.10 (`gain`) for every pair, window length, scaling and space |
| `av_bivariate_{r2,gain}_legend.png`, `av_bivariate_provenance.json`, `partitions_provenance.json` | colour keys; limits and colour-region shares; sources of the J-free quantities |
| `…_{observed,predictions_{a,v,av,j,avj}}.npy`, `…_rows.csv` | held-out responses and predictions of a fit made with `--save-predictions` (kept only while the time-resolved analysis needs them) |

## Layer sweeps

The single-model (`screen`) fit of every hidden layer of a model shows at which depth its features predict the response. The embeddings of every layer are written by `notebooks/feature_extraction/extract_embeddings.py --all-layers` as `{model}-l{k:02d}` (hidden state `k`, 0 the embedding-layer output), each layer is fitted with the `screen` runner, and `collect_layers.py` gathers the fits. The sweep of another model changes only the model name; the Whisper large-v3 encoder (33 hidden states, `--audio-models whisper-large-v3-l{00..32}`) is swept the same way.

```bash
python notebooks/feature_extraction/extract_embeddings.py --models qwen2.5-vl-3b-vit --bins 5 --all-layers
bash encoding/analysis.sh screen --video-models qwen2.5-vl-3b-vit-l{00..32} qwen2.5-vl-3b-vit-merger --bins 5 --variants loro:zscore
bash encoding/run_collect.sh --bins 5 --scalings zscore     # collected files and their provenance
```

Each layer's fit is stored in its own folder like any single-model fit (`outputs/encoding/group_average/{model}-l{k:02d}/delay5s_bin{B}s_skip{B}s/`). `run_collect.sh` collects every model that has such folders; its work list marks a model's collection as to run when a layer fit is newer than the provenance file.

| File (`outputs/encoding/group_average/layers/delay5s_bin{B}s_skip{B}s/`) | Contents |
|---|---|
| `encoding_{pearson_r,r2}_loro_{scaling}_screen_{model}_layers.dscalar.nii` | held-out Pearson r and R² of the single-model fit of every layer, one map per layer: `layer_01` is hidden state 0, `layer_02` hidden state 1, and so on, in layer order |
| `encoding_loro_{scaling}_screen_{model}_layers_provenance.json` | per map the hidden-state index and the fit it came from; the fit settings, which are checked to be identical across the layers |

`notebooks/visualization/main_figures.ipynb` plots the mean and the maximum of the held-out r over the cortical grayordinates against layer.

## Time-resolved analysis

Following Pushpita and Wehbe (bioRxiv 2025.11.05.686819), `time_resolved.py` asks window by window which feature set predicts each grayordinate. From the held-out predictions of a fit it computes per window:

- c_S(t) = z(y)(t) · z(ŷ_S)(t), the product of response and prediction z-scored within run (the paper's measure);
- rho_S(t) = [(y(t) − ȳ)² − (y(t) − ŷ_S(t))²] / Σ(y − ȳ)², the window's contribution to the held-out R² (Σ_t rho_S(t) = R²_S), and the partition terms built from it.

A window is flagged when a series exceeds chance: block permutation of the response (1,000 shuffles, 10 s blocks), threshold-free cluster enhancement over time, and a maximum-statistic test per run (`_fwe20`: family-wise error 0.2, the paper's criterion; `_fwe05`). Flagged windows are counted in seconds, and the series are related to speech and face/body labels by regression. Run for four pairs (`peavs16fa_peavs16fv`, `clap_viclip`, `languagebinda_languagebindv`, `dasheng12b_videomaev2gd50`) at 1 s, for both scalings, in cortex and subcortex. `notebooks/visualization/time_resolved_figures.ipynb` draws the figures from these files: audio and video series, shared and joint-embedding series before and after label removal with their flag strips, label-correlation bars and flat maps (`SCALING` selects the scaling of the maps and bars, `SERIES_SCALING` the fit whose saved predictions feed the area-level permutation test, which runs on the CPU). `notebooks/visualization/time_resolved_partition_figures.ipynb` draws, for `peavs16fa_peavs16fv` at 1 s with `zscore`, the area means (A5, FFC, STSdp) of `rho_a`, `rho_v`, `rho_j` and `shared_av` with the four runs on one axis, before and after leave-one-run-out removal of the speech and face labels, label-part bars with a block-shuffle null and partial correlations, and the table `partition_label_loro_*.csv` (`partition_figures/` in the time_resolved folder).

The joint model alone is added for the pairs `peavs16fa_peavs16fv` and `clap_viclip` at 1 s (`JOINT_SERIES_*` in `job_list.py`): a fit of subsets `av` and `j` with saved predictions in `outputs/encoding/joint_fits/` (the stored partitions stay untouched), then `time_resolved.py --joint-series` writes `rho_j_{J}(t)` of the joint model and `delta_j_{J}(t) = rho_j(t) - rho_av(t)` next to the other series, and the predictions are removed. `run_time_resolved.sh --bins 1 --scalings zscore` carries out both jobs.

| File | Contents |
|---|---|
| `encoding_loro_{scaling}_{pair}_{B}s_{series}.dtseries.nii` | one series per file: `c_a`, `c_v`, `rho_a`, `rho_v`, `unique_a`, `unique_v`, `unique_j_{J}`, `rho_j_{J}` and `delta_j_{J}` (joint model alone, and its gap to the pair's own audio-video fit), `shared_av`, the `joint_c` and `joint_rho` indicators, and the `_resid_speech` and `_resid_speech_fob` versions of `shared_av` and `unique_j_{J}` |
| `encoding_loro_{scaling}_{pair}_{B}s_rows.csv` | one row per held-out window: run, clip, response and stimulus onset, fold |
| `maps/{pair}_{scaling}.{npz,json}` | per-grayordinate maps and run settings of one pair |
| `encoding_{pair}_delay5s_bin{B}s_skip{B}s_{scaling}.dscalar.nii` | every per-grayordinate map of one pair, named without the pair: the time-resolved maps plus the pair's `r2_*`, partition and J maps from the partitions folder |
| `time_resolved_{scaling}.dscalar.nii` | the time-resolved maps of every pair, `{pair}_{map}` |
| `av_bivariate_seconds_{scaling}.dlabel.nii`, `av_bivariate_seconds_legend.png` | flagged audio (red) vs video (blue) seconds, five maps per pair: `timeresolved_*` (c) and `partition_*` (rho); colour key (symmetric: grey when both low, red for audio-higher, blue for video-higher, equal luminance so swapping audio and video swaps hue only); shared limit per window length and space: cortex 200 s at 1 s windows and 250 s at 2 s windows, subcortex 60 s at 1 s and 100 s at 2 s (`av_bivariate_provenance.json`) |
| `README_files.md` | list of the file families of the folder |

## Tests

```bash
python -m pytest tests/test_fold_evaluator.py tests/test_no_leakage.py tests/test_encoding.py \
    tests/test_variance_partition.py tests/test_time_resolved.py tests/test_runners.py
```
