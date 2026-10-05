# Encoding: banded-ridge variance partition

Encoding models predict the response of each of the 108,441 cortical grayordinates in the HCP 7T movie-watching data from model embeddings of the same movie window. Three feature spaces are compared: A, the embedding of the window's audio alone; V, of its video alone; and J, the model's joint audiovisual embedding. Seven banded ridge models, one per non-empty subset of {A, V, J}, are fitted and scored on the same rows, and their held-out R² values are partitioned into unique and shared parts (Lescroart et al. 2015; Deniz et al. 2019). `unique_j` is the variance J explains beyond A and V together.

## Method

### Responses and embeddings

Responses are the group average of 175 subjects (four runs, TR = 1 s, `raw` preprocessing). The 18 clips of `data/movie_timing.csv` are cut into non-overlapping windows of B = 5, 2 or 1 s (626 at 5 s). The response of window i of a clip with run-local onset t is the mean signal over the B seconds starting at t + 5 s + i·B; the shift accounts for the hemodynamic delay (`--hrf` also convolves the embeddings with the SPM canonical hemodynamic response function).

Embeddings are read from `outputs/model_embeddings/{model}/bin{B}s_skip{B}s/{model}_{a,v,av}.npy` (A, V, J), one row per window (extraction: `notebooks/feature_extraction/README.md`). A and V are the audio and video tower outputs for PE-AV, and audio-only and video-only passes pooled at J's layer for Nemotron. The last clip of each run (`video5`, `video9`, `video14`, `video18`) is the same 83 s stimulus, shown once per run.

### Cross-validation

`--split loro` (default) holds out each run in turn, with the repeated clips excluded; `loco` holds out each clip in turn; `fixed` holds out the four repeated clips (Hedger et al. 2025). Held-out predictions are pooled over outer folds before scoring.

### Normalization

Embeddings are scaled over the same rows and with the same statistics as the responses.

- Leave-one-run-out (`--split loro`, response scaling `run`, the default): each grayordinate is z-scored within each run on that run's own rows, the held-out run on its own. Each embedding channel is demeaned (`--feature-scaling demean`, the default) or z-scored (`zscore`) within each run in the same way. Per-run z-scoring of responses and features follows Huth et al. (2016) and Deniz et al. (2019). Under `fixed`, the repeated clip of each run is scaled separately from the run's training rows.
- Leave-one-clip-out (`--split loco`, requires `--response-scaling train`): in each fold, each run is scaled with the mean and standard deviation of its training rows, applied unchanged to the held-out clip, for responses and embeddings alike; no statistic is computed from held-out rows (Poldrack, Huckins & Varoquaux 2020). `loco` with per-run scaling is refused, because the held-out clip would enter its run's statistics.

`--response-scaling clip` and `none` are also available. `--n-components N` replaces each band by its first N principal components, fitted on the fold's training rows after scaling and applied to the held-out rows.

### Banded ridge

For a subset S of bands, Ŷ = Σ_{b∈S} X_b W_b, with W minimizing ‖Y − Σ_b X_b W_b‖² + Σ_b λ_b ‖W_b‖² (Dupré la Tour et al. 2022; himalaya `GroupRidgeCV`). Penalties are chosen per grayordinate by leave-one-run-out cross-validation within the training rows: 20 random draws of the relative band scalings, each with 23 overall penalties in logspace(−2, 9). The training mean of Y serves as the intercept.

### Scoring and partition

For each grayordinate, on the pooled held-out rows, R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², with ȳ the mean of the held-out responses. Negative values are kept; R² is undefined where the held-out response is constant. With R²(S) the held-out R² of subset S:

- unique A = R²(AVJ) − R²(VJ); unique V = R²(AVJ) − R²(AJ); unique J = R²(AVJ) − R²(AV)
- pairwise overlap I(X, Y) = R²(X) + R²(Y) − R²(XY)
- triple overlap T = R²(AVJ) − R²(A) − R²(V) − R²(J) + I(A,V) + I(A,J) + I(V,J)
- shared A and V only = I(A,V) − T, likewise for A–J and V–J; shared A, V and J = T

The seven regions sum to R²(AVJ) and can be negative where adding a band lowers held-out accuracy. The partition uses separate fits per subset, not the split R² of one joint fit (Dupré la Tour et al. 2022). Pearson r is not partitioned.

### Feature sets

J is always the fitted model's joint embedding; the tag names the source of A and V. The tag `screen` marks single-encoder fits (`--subsets a` or `v`).

| Tag | A | V |
|---|---|---|
| `unimodal_own` | the model's own audio-only embedding | the model's own video-only embedding |
| `dummy_av` (PE-AV) | joint embedding of real audio and blank video (all pixels at the normalized value of black, −1) | joint embedding of silent audio (zero waveform) and real video |
| control set name | the set's audio encoder | the set's video encoder |

Control sets take A and V from independent unimodal encoders. `-d50` and `-d75` denote the layer at 50% and 75% of the encoder's depth, a plain name the final layer; for each encoder, the depth with the highest leave-one-run-out single-model R² (`screen`), averaged over the 5% of grayordinates where the encoder family predicts best, was used.

| Set | Audio | Video |
|---|---|---|
| `mae` | dasheng-0.6b-d75 | videomaev2-large-d75 |
| `mae-large` | dasheng-1.2b | videomaev2-giant-d50 |
| `latent` | openbeats-large-i2 | vjepa2-vitl-d75 |
| `large-mixed` | dasheng-1.2b | vjepa2-vitg-d75 |
| `speech-wavlm` | wavlm-large-d75 | vjepa2-vitl-d75 |
| `speech-w2vbert` | w2v-bert-2.0-d75 | vjepa2-vitl-d75 |
| `text-contrastive` | clap-larger | pe-core-l14 |
| `text-asr` | whisper-large-v3 | pe-core-l14 |

## Usage

Dependencies are listed in `encoding/environment.yml`; input and output roots are set in the `CONFIG` block of `encoding/analysis.sh`.

```bash
bash encoding/analysis.sh variance_partition --models pe-av-small-16-frame --bins 5 --variants loro:demean
```

The runner fits each model with `unimodal_own`, `dummy_av` (PE-AV) and every control set (`--controls SET...|all|none`). `--variants` takes `split:feature_scaling` pairs; `loco` needs `--response-scaling train` and keeps one showing of the repeated clip (15 folds). `screen --audio-models M... --video-models M...` fits single encoders. `factorial_interaction` replaces J by the crossed-pair interaction J(v, a) + J(v′, a′) − J(v, a′) − J(v′, a), (v′, a′) being a training-run window of another clip. Each fit calls `encoding/variance_partition.py` (see `--help`).

### Dummy-modality contrasts

`bash encoding/run_diff_study.sh [partition|consolidate|all]` refits each joint model with J from a dummy-modality condition (`clsav_from_a`: dummy video; `clsav_from_v`: dummy audio) and the intact A and V. `encoding/diff_maps.py` then writes `encoding_diff_..._dummy.dscalar.nii` per condition (`joint_r_dummy`, `unique_j_dummy`, `diff_joint_r` = r_J(intact) − r_J(dummy)) and `encoding_diff_..._modality_presence.dscalar.nii` per model (`modality_presence_diff` = unique J(intact) − the larger dummy unique J).

## Outputs

Files go to `outputs/encoding/group_average/{model}/delay{D}s_bin{B}s_skip{B}s/`, `{model}` being the J model (the encoder for `screen`). Names follow `encoding_{metric}_{split}_{scaling}_{tag}_{content}`, where `{scaling}` is the feature scaling followed by `_trainscaled`, `_clipdemeaned` or `_rawresponse` for response scalings other than `run`, and by `_pca{N}` with `--n-components`.

| File | Contents |
|---|---|
| `encoding_r2_..._models.dscalar.nii` | `r2_a`, `r2_v`, `r2_j`, `r2_av`, `r2_aj`, `r2_vj`, `r2_avj`: held-out R² of each subset |
| `encoding_pearson_r_..._models.dscalar.nii` | `r_a` … `r_avj`: held-out Pearson r |
| `encoding_r2_..._partition.dscalar.nii` | the seven regions (`unique_a`, `unique_v`, `unique_j`, `shared_av_only`, `shared_aj_only`, `shared_vj_only`, `shared_avj`); `gain_j_over_a` = R²(AJ) − R²(A); `gain_j_over_v` = R²(VJ) − R²(V); `j_minus_av` = R²(J) − R²(AV); `shared_av` = R²(A) + R²(V) − R²(AV). For tags other than `unimodal_own` also `own_avj_minus_av` = R²(AVJ) of `unimodal_own` − R²(AV) of this tag, and `own_shared_av_minus_shared_av` = `shared_av` of `unimodal_own` − `shared_av` of this tag |
| `encoding_r2_{split}_{scaling}_own_vs_controls.dscalar.nii` | mean (`mean_…`) and minimum (`min_…`) of `own_shared_av_minus_shared_av` over the control sets, and `n_controls_own_shared_av_greater`, the number of control sets with a lower `shared_av` than `unimodal_own` |
| `encoding_r2_..._per_fold.npz` | R², penalties and band scalings per fold and grayordinate; per-clip R² for `fixed` |
| `encoding_r2_..._provenance.json` | settings, embedding sources and definitions |

## Current results (5 October 2026)

Group average, 5-s windows, demeaned embeddings. Each cell is the held-out value averaged over the 108,441 cortical grayordinates, leave-one-run-out / leave-one-clip-out. Shared AV = R²(A) + R²(V) − R²(AV); unique J = R²(AVJ) − R²(AV). The first three rows take A and V from the J model itself; the last eight from the control sets above.

| A and V from | R²(A) | R²(V) | R²(AV) | shared AV | R²(AVJ), J = PE-AV | unique J, PE-AV | R²(AVJ), J = Nemotron | unique J, Nemotron |
|---|---|---|---|---|---|---|---|---|
| PE-AV (`unimodal_own`) | 0.0482 / 0.0472 | 0.0754 / 0.0790 | 0.1003 / 0.0986 | +0.0233 / +0.0275 | 0.1011 / 0.0983 | +0.0008 / −0.0003 | – | – |
| Nemotron (`unimodal_own`) | 0.0468 / 0.0503 | 0.0872 / 0.0926 | 0.1190 / 0.1171 | +0.0150 / +0.0259 | – | – | 0.1207 / 0.1184 | +0.0017 / +0.0013 |
| PE-AV (`dummy_av`) | 0.0426 / 0.0453 | 0.0807 / 0.0831 | 0.1009 / 0.1015 | +0.0224 / +0.0270 | 0.0998 / 0.1016 | −0.0010 / +0.0001 | – | – |
| `mae` | 0.0290 / 0.0344 | 0.0727 / 0.0809 | 0.0908 / 0.0943 | +0.0108 / +0.0209 | 0.0983 / 0.0927 | +0.0075 / −0.0016 | 0.1175 / 0.1158 | +0.0266 / +0.0215 |
| `mae-large` | 0.0325 / 0.0346 | 0.0636 / 0.0662 | 0.0823 / 0.0795 | +0.0137 / +0.0214 | 0.0944 / 0.0888 | +0.0120 / +0.0093 | 0.1237 / 0.1193 | +0.0414 / +0.0397 |
| `latent` | 0.0334 / 0.0390 | 0.0880 / 0.0842 | 0.0984 / 0.0960 | +0.0230 / +0.0273 | 0.1037 / 0.0994 | +0.0053 / +0.0034 | 0.1151 / 0.1175 | +0.0167 / +0.0215 |
| `large-mixed` | 0.0325 / 0.0346 | 0.0879 / 0.0833 | 0.1051 / 0.0930 | +0.0153 / +0.0250 | 0.1067 / 0.0930 | +0.0017 / +0.0001 | 0.1211 / 0.1109 | +0.0161 / +0.0179 |
| `speech-wavlm` | 0.0307 / 0.0360 | 0.0880 / 0.0842 | 0.1031 / 0.0952 | +0.0156 / +0.0250 | 0.1090 / 0.1008 | +0.0059 / +0.0057 | 0.1194 / 0.1155 | +0.0163 / +0.0203 |
| `speech-w2vbert` | 0.0315 / 0.0388 | 0.0880 / 0.0842 | 0.1061 / 0.0991 | +0.0134 / +0.0240 | 0.1098 / 0.1029 | +0.0037 / +0.0039 | 0.1207 / 0.1182 | +0.0146 / +0.0192 |
| `text-contrastive` | 0.0269 / 0.0276 | 0.0643 / 0.0651 | 0.0741 / 0.0712 | +0.0171 / +0.0216 | 0.0881 / 0.0885 | +0.0140 / +0.0174 | 0.1090 / 0.1105 | +0.0349 / +0.0393 |
| `text-asr` | 0.0396 / 0.0489 | 0.0643 / 0.0651 | 0.0835 / 0.0877 | +0.0205 / +0.0263 | 0.0851 / 0.0863 | +0.0016 / −0.0013 | 0.1102 / 0.1142 | +0.0267 / +0.0265 |

Against the model's own A and V, unique J is within 0.002 of zero for both models, and its map correlates only 0.40 (PE-AV) and 0.30 (Nemotron) between the two splits. Using PE-AV's joint embedding with the other modality blanked as A and V (`dummy_av`) gives the same result. Against the control pairs, Nemotron's J adds 0.015 to 0.041 and its maps correlate 0.72 to 0.86 between splits; PE-AV's J adds −0.002 to +0.017.

Three control pairs have a higher mean R²(AV) than PE-AV's R²(AVJ) under leave-one-run-out, and one under leave-one-clip-out. PE-AV's full model is above all eight pairs at 26% / 25% of grayordinates, Nemotron's at 36% / 36%.

Where PE-AV's own R²(AV) exceeds 0.05, its own A and V share 0.038 / 0.043. The `latent` pair shares as much (PE-AV larger at 50% / 48% of these grayordinates), and the mean over all eight pairs is lower by 0.012 / 0.006. A model's own A and V share more than all eight pairs at 17% / 15% of cortex for PE-AV and 14% / 16% for Nemotron.

Between the two splits, the R²(AVJ) maps correlate 0.93 to 0.96 and the shared AV maps 0.51 to 0.82. Reducing each band to 32 or 128 principal components (fitted on training rows) lowers R² and leaves the between-split agreement of unique J unchanged.

## Tests

`tests/test_fold_evaluator.py`, `tests/test_no_leakage.py`, `tests/test_encoding.py`, `tests/test_variance_partition.py`.
