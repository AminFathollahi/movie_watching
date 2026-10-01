# CKA: searchlight centered kernel alignment

Compares the geometry of cortical responses with the geometry of model embeddings in the HCP 7T movie-watching data using centered kernel alignment (CKA), a measure of the similarity of two window-by-window Gram matrices. CKA is not representational similarity analysis (RSA): it compares inner products, not rank-ordered distances, and the cross-validated variants remove the noise bias that a Gram matrix of noisy responses carries. Everything here is a searchlight over cortical grayordinates (108,441, medial wall excluded) with `k` geodesic neighbors, as in `rsa/`.

Run from the repository root in the `movie` conda environment.

## What is computed

**Windows.** Binning, delay and per-run normalization are those of `rsa/` (`preprocess_fmri`, `process_model_embeddings`): K windows of `bin_sec` seconds (626 at 5 s) over the 18 clips, each response z-scored per run and each embedding dimension normalized per run (`--feature-scaling center` subtracts the mean, `zscore` also divides by the standard deviation). The embedding of a window is the joint audiovisual embedding `{model}_av.npy`. `window_index` returns the clip and run of every window, in the order of the response rows.

**Model Gram matrix.** With X the (K, d) embedding matrix and H = I − 11ᵀ/K the centering matrix over windows, G_model = H X Xᵀ H. It holds the centered inner product of every pair of windows; it is scaled to unit Frobenius norm.

**CKA.** For two (K, K) centered Gram matrices A and B, CKA(A, B) = ⟨A, B⟩_F / (‖A‖_F ‖B‖_F), the cosine between the two matrices read as vectors; it lies between −1 and 1.

**Non-cross-validated CKA (`noncv`).** For a searchlight with P neighbors, Y is the (K, P) matrix of group-average responses. G_brain = H Y Yᵀ H and the value is CKA(G_brain, G_model). Products Y Yᵀ of noisy responses contain noise-times-noise terms (a positive bias on the diagonal and, through the temporal autocorrelation of the response, between neighboring windows), so this is a diagnostic. It is also written with the four repeated clips (`video5`, `video9`, `video14`, `video18`, the same 83.375 s stimulus shown once per run) dropped (`noncv_..._norepeats`): their windows are 64 of 626, they are identical stimulus, and they sit at fixed places in each run. The per-run response normalization is that of all windows; the model Gram matrix is recomputed on the retained windows.

**Subject partitions (`partitions`).** The listed subjects are assigned to M disjoint groups (subject i to group i mod M; M = 25, 7 subjects per group), and each group's preprocessed responses are binned like the group average (per-run mean subtracted), averaged within the group, then divided by the per-run, per-grayordinate standard deviation of the all-subject average. The result is an array of shape (M, K, 108441), float32, 6.8 GB at M = 25 and K = 626. After building it, the script prints the per-grayordinate correlation between the mean over groups and the z-scored group-average responses (a plumbing check; it should be near 1).

**Cross-validated CKA (`cv`).** For a searchlight, let Y_m be the (K, P) responses of group m. The cross-validated Gram matrix over windows is

G_cv = Σ_{m≠n} Y_m Y_nᵀ / (M (M − 1) P),

the sum over ordered pairs of different groups. The noise of different groups is independent, so the noise-times-noise terms of Y Yᵀ drop out in expectation and G_cv estimates the stimulus-driven second-moment matrix per neighbor. The value is CKA(H G_cv H, G_model); it can be negative. In the code, with S = Σ_m Y_m, G_cv = (SᵀS − Σ_m Y_mᵀ Y_m) / (M (M − 1) P) (the explicit pair sum is checked in `tests/test_cka.py`).

**Whitened cross-validated CKA (`cv-ar`).** Noise is autocorrelated across the windows of a run. Its covariance between windows is estimated from the partitions as Σ = (1 / ((M − 1) V)) Σ_v Σ_m (y_vm − ȳ_v)(y_vm − ȳ_v)ᵀ, with y_vm the length-K response of grayordinate v in group m, ȳ_v the mean over groups and V the number of grayordinates. W is the symmetric inverse square root of HΣH: its eigenvector of the constant direction (the null space of the centering) is dropped and every other eigenvalue is floored at 1e-3 times their mean (the number floored is in the noise statistics). The value is CKA(W G_cv W, W Xc (W Xc)ᵀ), with Xc the column-centered embedding. In the low-signal limit this inner product equals the likelihood-ratio statistic of the pairwise window contrasts under noise covariance Σ (identity checked numerically in `tests/test_cka.py`); `cv` is the same statistic with Σ = I. The noise statistics (mean lag-1 and lag-2 correlation within runs, mean absolute correlation across runs, eigenvalue range, number floored) are written to `work/`.

**Log-likelihood (`cv-loglik`, `cv-ar-loglik`).** With J = K (K − 1) / 2 window pairs, Λ = −(J / 2) log(1 − r²), where r is the `cv` or `cv-ar` value (r² capped at 1 − 10⁻¹²). Λ is a monotone function of r² scaled by the nominal number of pairs; pairs are not independent, so Λ ranks searchlights and models and is not a calibrated likelihood.

**Model difference (`cv-loglik-diff`, `cv-ar-loglik-diff`).** For each unordered pair of models (A, B) in the order given, Λ_A − Λ_B; positive favors A.

**Interaction decomposition (`multimodal_decomposition.py`).** For a joint embedding J and unimodal embeddings A and V, each z-scored per feature, R = J − Ĵ, where Ĵ is the ridge prediction of J from [A | V] with the penalty chosen by 5-fold cross-validation over windows. The residual is fit and evaluated on the same windows, so it is not out-of-fold. The global score is ‖R‖² / ‖J‖². Non-cross-validated CKA of the response Gram matrix with the Gram matrices of J, of [A | V] and of R is computed per Glasser parcel and per searchlight; the specificity index is CKA(brain, R) / CKA(brain, J). Outputs: `cka_decomp_{target}.dscalar.nii`, `cka_decomp_parcels_{target}.csv`, `cka_decomp_summary_{target}.json` in `--output-dir`.

## Running

### `analysis.sh`

```bash
bash cka/analysis.sh partitions [--bins B...]
bash cka/analysis.sh run [--models M...] [--bins B...] [--variants SCALING...] [--max-vertices N] [--output-dir DIR]
```

Defaults: `--models pe-av-small-16-frame nemotron_layer18_mp`, `--bins 5` (stride equals bin), `--variants center`, `--max-vertices 0` (all grayordinates). A failed step is logged and the script exits non-zero.

| Mode | Action |
|------|--------|
| `partitions` | Builds the partition array of each bin from the raw per-subject CIFTIs in `RAW_DIR` (subjects in `SUBJECTS_LIST`; subjects with missing runs are skipped and recorded). Resumable: a checkpoint is saved every 20 subjects. Run once per bin before `run` |
| `run` | For each bin and variant: non-cross-validated, cross-validated and whitened maps, log-likelihoods and model differences for all models in one pass over the partition array |

### `cka_searchlight.py` directly

```bash
python cka/cka_searchlight.py run \
  --preprocessed-dir data/preprocessed/average_sub/raw --timing-csv data/movie_timing.csv \
  --template-cifti data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \
  --partitions-dir <cache> --embeddings-dir outputs/model_embeddings --output-dir outputs/cka \
  --left-surface <L.surf.gii> --right-surface <R.surf.gii> --workbench <wb_command> \
  --geodesic-cache-dir outputs/rsa/_geodesic_cache --models pe-av-small-16-frame nemotron_layer18_mp
python cka/cka_searchlight.py partitions <same common arguments> --raw-dir <raw CIFTIs> --subjects-list data/subjects.txt
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--preprocessed-dir`, `--fmri-suffix`, `--subject` | required, `raw`, `group_average` | Group-average response `{dir}/{subject}_{suffix}_cortex_59k.dtseries.nii` and `..._run_trs.npy` |
| `--timing-csv`, `--template-cifti`, `--embeddings-dir` | required | Clip timing, grayordinate template, embeddings root (`{model}/bin{B}s_skip{S}s/{model}_av.npy`) |
| `--partitions-dir`, `--n-partitions` | required, 25 | Directory and number of groups M of the partition array |
| `--bin-sec`, `--skip-sec`, `--delay-sec`, `--tr` | 5, 5, 5, 1 | Window length, stride, response delay, repetition time (seconds) |
| `--models` | required (`run`) | Embedding models; one `{model}_av` directory each |
| `--output-dir` | required (`run`) | Output root |
| `--left-surface`, `--right-surface`, `--workbench`, `--geodesic-cache-dir` | required (`run`) | Midthickness surfaces and the geodesic neighbor cache of `rsa/searchlight.py::get_neighbors` |
| `--k` | 100 | Searchlight neighbors |
| `--feature-scaling` | `center` | `zscore` or `center` |
| `--exclude-video-ids` | the four repeated clips | Clips dropped in the `norepeats` diagnostic |
| `--max-vertices` | 0 | Evaluate this many evenly spaced grayordinates (others stay 0); for tests |
| `--raw-dir`, `--subjects-list`, `--limit`, `--checkpoint-every` | required, required, 0, 20 | `partitions` only: raw CIFTIs, subject list, first N subjects only, checkpoint interval |

Configuration lives in the `CONFIG` block of `analysis.sh`; `PARTITIONS_DIR` points to the external drive because one array is 6.8 GB. Memory: the partition array is held in RAM (about 7 GB) and the searchlight runs on the GPU when CUDA is available (a batch of 32 searchlights is about 0.2 GB of partition data), on the CPU otherwise.

## Outputs

`{OUTPUT_DIR}/{flag}/{subject}/`, with `{stem} = cka_59k_{flag}_k{K}_delay{D}s_bin{B}s_skip{S}s_{measure}_{scaling}`:

```
{model}_av/{stem}_maps.dscalar.nii               measure cv, cv-ar, cv-loglik, cv-ar-loglik; map names cka-cv, cka-cv-ar, cka-cv-loglik, cka-cv-ar-loglik
{model}_av/diagnostics/{stem}_maps.dscalar.nii   measure noncv (map name cka); {stem}_norepeats_maps.dscalar.nii without the repeated clips
{stem}_maps.dscalar.nii                          measure cv-loglik-diff_{A}_av_minus_{B}_av, cv-ar-loglik-diff_...; map names cka-cv-loglik-diff, cka-cv-ar-loglik-diff
work/                                            every map as .npy ({model}_av/{stem}.npy), vertices.npy, noise covariance, whitener and noise statistics
{PARTITIONS_DIR}/partitions_M{M}_delay{D}s_bin{B}s_skip{S}s.npy   (M, K, 108441) float32, and .json with the subject-to-group map and the validation correlation
```

All maps are 108,441 grayordinates in the template's brain-model axis. CKA does not use the `rsa_` prefix.

## Tests

```bash
conda run -n movie python -m pytest tests/test_cka.py
```

Synthetic, CPU, no data files: the likelihood identities above, whitener, cross-validated Gram matrix, log-likelihood, window indexing against `preprocess_fmri`, and both searchlight passes against a direct computation per vertex.
