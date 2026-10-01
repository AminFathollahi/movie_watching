# rsa/shared: shared representational similarity analysis utilities

Library modules imported by the scripts in `rsa/` (and by the residual-embedding generators in `notebooks/feature_extraction/`). Nothing here is run directly except the self-checks listed under Tests. Import from the repository root as `from rsa.shared.<module> import ...`.

| Module | Contents |
|---|---|
| `rsa_utils.py` | Binning of fMRI and embeddings, representational dissimilarity matrices (RDMs), correlation of RDMs, group-level bootstrap helpers |
| `naming.py` | Searchlight output names and the shared `--model-norm` argument |
| `model_registry.py` | Model catalogue, embedding paths, partial-RSA run definitions, audio-video reference models |
| `residuals.py` | Linear and projection residualization of joint embeddings |
| `sheet_rsa.py` | Topo-Omni cortical-sheet geometry, plotting and hotspot statistics used by `rsa/full_sheet_rsa.py` |

## rsa_utils.py

Binning (all functions use the clip table `movie_timing.csv` with columns `onset_sec` (global time), `duration_sec`, `run_id`, and `run_trs`, the number of TRs per run):

| Function | What it computes |
|---|---|
| `preprocess_fmri(fmri_continuous, timing_df, run_trs, bin_sec, tr, delay_sec=0.0, skip_sec=None, normalize=True)` | For each clip, windows of `bin_sec` seconds starting at `onset_sec - run_start + delay_sec`, stride `skip_sec` (default `bin_sec`); each window is the mean over its TRs; windows that cross the run end are dropped. Per run, each vertex is z-scored over windows (`normalize=True`) or only mean-subtracted. Returns `(n_windows, n_vertices)` float32 |
| `process_model_embeddings(emb_path, timing_df, bin_sec, tr, run_trs, delay_sec=0.0, hrf=False, skip_sec=None, model_norm="zscore")` | Selects the rows of the precomputed embedding file that correspond to the surviving windows, optionally convolves each clip with the SPM canonical hemodynamic response (`hrf=True`), and normalizes each dimension per run: `"zscore"` (subtract mean, divide by standard deviation) or `"center"` (subtract mean). Returns `(n_windows, n_features)` float32. The default here is `"zscore"`; the command-line default in `rsa/` scripts is `"center"` |
| `process_model_embeddings_with_hrf(emb_path_tr, timing_df, bin_sec, tr)` | Variant that convolves 1 s-resolution embeddings first, then bins and z-scores per run |
| `get_run_bin_counts(timing_df, run_trs, bin_sec, tr, delay_sec, skip_sec)` | Number of windows per run, mirroring `preprocess_fmri` exactly (used to align temporal blocks and permutation shifts with runs) |
| `align_and_assert_bins(fmri_binned, model_binned)` | Raises if the brain and model window counts differ |
| `assert_segment_timing(timing_df, bin_sec, tr, delay_sec, skip_sec, run_trs)` | Raises if any window would fall outside its run |
| `spm_hrf(tr, oversampling=16)` | SPM canonical HRF kernel (gamma densities with shape 6 and 16, ratio 1/6), normalized to sum 1 |
| `load_fmri_cifti(path)` | Loads a dtseries as `(n_vertices, T)` float32 |

RDMs and comparison:

| Function | What it computes |
|---|---|
| `compute_rdm(data, method="correlation")` | For `(n_conditions, n_features)` data, the symmetric `n x n` matrix of correlation distances `1 - r` (or cosine distance with `method="cosine"`) |
| `correlate_rdms(rdm1, rdm2, method="spearman")` | Correlation (`spearman`, `pearson`, or `rho_a`) of the strict lower triangles; returns `(r, p)`, with `p = nan` for `rho_a` |
| `correlate_rdms_rho_a(rdm1, rdm2)` | Kendall's tau (`scipy.stats.kendalltau`) of the lower triangles |

Group-level helpers:

| Function | What it computes |
|---|---|
| `corrected_2factor_bootstrap(rho_stack, block_stack, n_boot=2000, rng=None)` | Corrected two-factor bootstrap variance of the group-mean `rho` (Schutt et al. 2023, Eq. 5). Resamples subjects, blocks, and both, giving `var_subj`, `var_block`, `var_both`; returns `var_c2f = clip(2 (var_subj + var_block) - var_both, lower, upper)` with `lower = max(var_subj, var_block)` and `upper = max(var_both, lower)`, `var_subj`, `var_block`. Inputs: `rho_stack (n_subjects, n_vertices)`, `block_stack (n_subjects, n_blocks, n_vertices)` |
| `aggregate_blocks(block_stack, n_target)` | Averages adjacent blocks to reduce `N_file` blocks to `n_target` (`N_file` must be divisible by `n_target`) |

## naming.py

`MODEL_NORMS = ("zscore", "center")`, `DEFAULT_MODEL_NORM = "center"`.

| Function | Result |
|---|---|
| `add_model_norm_arg(parser)` | Adds `--model-norm {zscore,center}` (default `center`) |
| `searchlight_config(k, delay_sec, bin_sec, skip_sec, method, model_norm, hrf=False)` | `k{k}_delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}`; with `hrf=True`, `delay{D}s` becomes `hrf` |
| `searchlight_stem(fmri_tag, k, delay_sec, bin_sec, skip_sec, method, model_norm)` | `rsa_59k_{fmri_tag}_{searchlight_config}` |
| `method_label(method, model_norm)` | Comparator label for group-level names: `{method}_{model_norm}`, or just `rho_a` (crossnobis does not depend on the model normalization) |

## model_registry.py

Embedding location convention: `{embeddings_dir}/{model}/bin{B}s_skip{S}s/{model}_{modality}.npy`.

| Name | Meaning |
|---|---|
| `BIN_SEC_DEFAULT`, `DELAY_SEC_DEFAULT`, `TR_DEFAULT` | `5.0`, `5.0`, `1.0` |
| `emb_path(embeddings_dir, model, modality, bin_sec, skip_sec=None)` | Resolves the path above (`skip_sec` defaults to `bin_sec`) |
| `check_embeddings_exist(...)` | Same, raises `FileNotFoundError` if missing |
| `MODELS` | Dict `name -> {modalities, joint, description}`. Omni-family names carry the pooling suffix `_mp` (mean pool) or `_lt` (last token), e.g. `nemotron_layer18_mp`, `omni3b_layer18_lt`, `topoomni_layer18_sheet_mp`. Includes scramble (`_avscramble`) and dummy-modality (`_clsav_from_{a,v}`) controls, unimodal reference models, and the residual pseudo-models `{model}_av_linear_resid_unimodal` and `pe-av-small-16-frame_av_linear_resid_own` |
| `validate_model(model, modality)` | Raises `ValueError` if the model or modality is not registered |
| `NATIVE_AV_MODELS` | Registered joint audio-video models, excluding scramble and dummy-modality controls |
| `LEGACY_BARE_AV_MODELS`, `RESIDUALIZED_AV_MODELS` | Bare-name omni probes, and the roster used by the partial-correlation, linear-residual and projection-residual analyses |
| `AV_DERIVED_COMMON_BASELINES`, `av_derived_baselines(model)` | The three audio/video reference pairs (`own`, `unimodal` = AudioMAE + VideoMAEv2-Large, `text_aligned` = WavLM-Large + PE-Core ViT-L/14) for a natural joint model; `None` for scramble, dummy-modality and residual pseudo-models |
| `PartialRSARun` | Named tuple `(target, nuisance, label, description, kind)`; `kind` is `"cross_baseline"` (nuisance from other models) or `"integration"` (nuisance is the target model's own audio and video embeddings) |
| `PARTIAL_RSA_RUNS`, `validate_run(run_name)` | Run keys accepted by `rsa/partial_rsa.py --run`: `integration_*`, `partial_corr_*`, scramble / dummy-modality integration runs, and the unimodal-reference families |
| `DIAGONAL_MASK_DEFAULT` | Default model, modality and bin size for `rsa/rdm_diagonal.py` |

## residuals.py

| Function | What it computes |
|---|---|
| `linear_residual(J, nuisance_list, alpha_grid=None, cv_folds=5)` | Ridge-regression residual `R` of the joint embedding `J (n, d_J)` after regressing on the concatenated, z-scored nuisance embeddings, with one linear map fit across all samples and the ridge penalty chosen by `cv_folds`-fold cross-validation (`alpha_grid` default `logspace(0, 8, 30)`). Returns `(R, ms_score, best_alpha)` with `ms_score = ||R||_F^2 / ||J||_F^2`. Implemented by `cka.multimodal_decomposition.compute_interaction_residual_cv` |
| `projection_residual(av, a, v, eps=1e-8)` | For each row `t` independently: Gram-Schmidt orthonormalize `(a_t, v_t)` into `u1, u2` (a vector with norm below `eps` is dropped), and return `av_t - (av_t . u1) u1 - (av_t . u2) u2`. `av`, `a`, `v` must have the same shape |

The generator scripts save results as ordinary `{model}_{modality}.npy` embedding files, which `searchlight.py` and `partial_rsa.py` then read like any other model.

## sheet_rsa.py

Constants: `SHEET_ROWS = 304`, `SHEET_COLS = 512`, `N_UNITS = 155648`, `ENCODER_ROWS = 160`, towers `0` vision, `1` audio, `2` thinker (`TOWER_NAMES`).

| Function | What it computes |
|---|---|
| `tower_id()` | Per-unit tower id from the raster position `(row, col) = (k // 512, k % 512)`: rows 0-159 are vision (columns 0-255) or audio (columns 256-511); rows 160-303 are the thinker stack |
| `load_true_coords(cache_path)` | `(N_UNITS, 2)` true `(row, col)` per raster index; regenerates `init_coords.permute_coordinates(seed=42)` from the Topo-Omni repository (`TOPO_REPO`) on a cache miss and saves it. The permutation moves units only within their own architectural block |
| `knn_on_sheet(coords, k)` | `k` nearest sheet neighbors per unit in `(row, col)` space (self excluded), via `scipy.spatial.cKDTree` |
| `random_neighbors_within_tower(tid, k, rng)` | `k` random neighbors per unit drawn from the same tower, coordinates ignored (control for `knn_on_sheet`) |
| `sanity_corr(sheet_emb, seed_emb)` | Pearson correlation of each unit with the seed region's mean time series (a side check, not the RSA result) |
| `mean_nn_dist(xy)` | Mean distance to the nearest other point |
| `robust_vlim(values, pct=1.0)`, `plot_sheet_map(...)` | Percentile color limits; heat map of per-unit values on true coordinates with tower boundaries |
| `characterize(results, coords, tid, args)` | Cross-seed contrast, per-tower mean `rho`, hotspot composition by tower (top `args.pref_top_decile` of units), nearest-neighbor contiguity of hotspots against 2000 random subsets, hotspot Jaccard overlap against a random-subset null |

## Tests

```bash
pytest tests/test_rsa.py tests/test_model_norm.py tests/test_sheet_rsa.py tests/test_av_derived_maps.py tests/test_residualized_maps.py
python rsa/shared/residuals.py     # projection residual is orthogonal to a and v at every row; vanishes for av inside span{a, v}
```
