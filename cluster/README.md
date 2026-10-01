# cluster: temporal-state / spatial-network interaction and clustering-method selection

Two independent pipelines share this directory.

| Pipeline | Question | Entry points |
|----------|----------|--------------|
| Interaction | Do movie-driven temporal states of a model's embedding differ in how strongly they drive each brain network? | `cluster.sh groupaverage`, `run_cluster.py`, `state_content.py`, `av_integration.py` |
| Method selection | Which dimensionality-reduction and clustering methods give useful partitions of grayordinate time series, and of embedding-channel time series? | `run_vertex_clustering.sh`, `run_vertex_model_selection.sh`, `cluster.sh channel_model_selection`, `consolidate_vertex_outputs.py`, `screen_temporal_differentiation.py` |

Environment: conda environment `movie`. All outputs go under `outputs/cluster/` (`--output-dir`). Numerical results of the method-selection pipeline are in `RESULTS.md`; `cluster.txt` is an archived planning note, not documentation.

## 1. Interaction pipeline

### What is computed

Notation: `V` = 108,441 cortical grayordinates (medial wall excluded), `S` = number of movie segments (the temporal grid below), `K` = number of temporal states, `G` = number of spatial networks.

1. **Segment grid.** The group-average fMRI time series `X` (`V x T`, `T` time points of 1 s) is cut into windows of `--bin-sec` seconds spaced `--skip-sec` apart, delayed by `--delay-sec`, within each run (runs never share a window). Each window is averaged over time points and z-scored within each run, giving `fmri_binned` (`S x V`). The model embedding of the same windows (`process_model_embeddings`, no hemodynamic convolution, z-scored within run) gives `emb` (`S x n_features`). The two must have the same `S`.
2. **Temporal states.** `emb` is reduced by PCA to `--temporal-n-components` (default 50) components (randomized SVD, seed 0). A Gaussian hidden Markov model with `--n-temporal-states` (`K`, default 10) states and covariance `--hmm-covariance` (default `diag`) is fit with `hmmlearn`; run lengths are passed so transitions never cross runs. The model is refit from `--hmm-n-init` (default 10) random seeds and the restart with the highest log-likelihood is kept. Output: a hard state label per segment and the state posteriors (`S x K`). A diagnostic sweep over `K = k-min..k-max` (defaults 2 to 15) always runs and is written to `temporal_hmm_diagnostics.json`: for each `K`, leave-one-run-out held-out log-likelihood, Bayesian information criterion of the full fit, and adjusted Rand index between the best and second-best restart; it does not influence `K`.
3. **Spatial networks (brain only, cached).** `X` (`V x T`) is reduced over time by PCA to `--spatial-n-components` (default 100) features per grayordinate; HDBSCAN (`--min-cluster-size` default 100, `--min-samples` default 10) clusters grayordinates into networks; label `-1` is noise. Because this does not depend on the model it is computed once per spatial configuration and reused.
4. **Interaction matrix.** For state `k` and network `g`: `M[k, g] = mean over segments s in state k of ( mean over grayordinates v in network g of fmri_binned[s, v] )`. Noise grayordinates are excluded. The deviation is `D[k, g] = M[k, g] - m[g]`, with `m[g]` the mean of the network's segment means over all segments.
5. **Block-circular permutation test.** Null: within each run, circularly shift the state labels by an independent random offset (keeps run boundaries and the state sequence's autocorrelation), recompute `D`; repeat `--n-permutations` (default 1000) times. Per-cell two-sided p = fraction of permutations with `|D_null| >= |D_observed|`, floored at `1 / n_permutations`. `--perm-correction fdr` (default) applies Benjamini-Hochberg across all `K x G` cells; `maxstat` compares each `|D|` with the permutation distribution of the maximum `|D|` over cells.
6. **State content (`state_content.py`).** For each state: number of segments, total duration, dominant `video_id` and its share, segment counts per video, and merged stimulus time ranges per video. Only video identity and timing are available; no caption or transcript text is used.
7. **Audiovisual integration cartography (`av_integration.py`).** Six conditions are fit with identical settings (`a`, `v`, `av`, and three control embeddings for the audiovisual modality: `avscramble` = real audio with temporally permuted video, `clsav_from_a` = real audio with blanked video, `clsav_from_v` = real video with blanked audio). For each network `g` and condition `c`: `n_sig[c]` = number of states with corrected p below 0.05 in column `g`; `effect[c]` = mean over states of `|M[k, g] - mean_k M[k, g]|`. Category: `unclassified` if no cell has corrected p below 0.05 in any condition; otherwise the best condition is the one with the largest `(n_sig, effect)`; `av_integrative` if the best condition is `av` and `n_sig[av]` is strictly greater than `n_sig` of each of the three controls; `av_nonspecific` if the best condition is `av` or one of the controls without that property; `auditory` if best is `a`; `visual` if best is `v`. Each network is also cross-referenced with the Glasser parcellation: top three parcels by share of the network's grayordinates and verdict `known:{parcel}` if the top parcel's share is at least 0.4, else `novel/mixed(...)`.

### Running

```bash
bash cluster/cluster.sh groupaverage      # default; loops MODELS x modalities in cluster.sh
```

`cluster.sh` modes: `groupaverage` (calls `run_cluster.py` once per model/modality), `persubject` (calls `run_cluster.py --mode persubject`, which only logs that it is not implemented and exits), `temporal_differentiation` (section 2), `channel_model_selection` (section 2). All settings (`MODELS`, `BIN_SEC=5`, `SKIP_SEC=5`, `DELAY_SEC=5`, `TR=1.0`, `TEMPORAL_NCOMP=50`, `SPATIAL_NCOMP=100`, `N_STATES=10`, `HMM_COVARIANCE=diag`, `HMM_N_INIT=10`, `MIN_CLUSTER_SIZE=100`, `MIN_SAMPLES=10`, `N_PERM=1000`, `PERM_CORRECTION=fdr`, input and output paths) are variables at the top of `cluster.sh`. `MODELS` entries read `model:modalities`; the control conditions of `av_integration.py` are on-disk models named `{model}_avscramble`, `{model}_clsav_from_a`, `{model}_clsav_from_v` with modality `av`.

Single run:

```bash
python cluster/run_cluster.py --model nemotron_layer27_mp --modality av \
    [--group-avg-cifti F --group-avg-trs F --timing-csv F --embeddings-dir D --template-cifti F --output-dir D] \
    [--bin-sec 5 --skip-sec 5 --delay-sec 5 --tr 1] \
    [--temporal-reduction pca|tphate --temporal-n-components 50 --adv-n-components 10] \
    [--spatial-reduction pca|phate --spatial-n-components 100] \
    [--spatial-cluster hdbscan|nmf --min-cluster-size 100 --min-samples 10 --nmf-rank 20] \
    [--n-temporal-states 10 --hmm-covariance diag --hmm-n-init 10 --k-min 2 --k-max 15] \
    [--temporal-source embedding|brain] [--n-permutations 1000 --perm-correction fdr|maxstat]
```

`--model` and `--modality {a,v,av}` are required. `tphate`/`phate` need those packages. With `--temporal-source brain` the states come from `fmri_binned` and the interaction matrix is descriptive only (no permutation test, no p-values). `--select-states` and `--stability` are accepted and have no effect. A run is skipped when `temporal_state_labels.npy`, `temporal_state_posteriors.npy`, `interaction_matrix.npy` and `manifest.json` already exist.

```bash
python cluster/state_content.py --model nemotron_layer27_mp --modality av --config CONFIG_TAG   # no arguments: self-check
python cluster/av_integration.py --model nemotron_layer27_mp --config CONFIG_TAG --spatial-config SPATIAL_TAG   # no arguments: self-check
```

`CONFIG_TAG` is the directory name under `{model}_{modality}/` (`treduce-pca_tnc50_K10_sreduce-pca_snc100_scluster-hdbscan_bin5s_skip5s_delay5s`); `SPATIAL_TAG` is the directory under `_spatial/` (`sreduce-pca_snc100_scluster-hdbscan_mcs100_ms10_bin5s_skip5s_delay5s`).

### Outputs

```text
outputs/cluster/group_average/
├── _spatial/{SPATIAL_TAG}/
│   ├── spatial_vertex_labels.npy        (V,) raw labels, -1 = noise
│   ├── spatial_vertex_labels.dlabel.nii key 0 = unassigned, 1..G = networks
│   └── spatial_report.json              n_networks, noise_fraction, subsampled silhouette, network sizes, PCA variance
├── {model}_{modality}/{CONFIG_TAG}/
│   ├── temporal_state_labels.npy        (S,)
│   ├── temporal_state_posteriors.npy    (S, K)
│   ├── temporal_hmm_diagnostics.json
│   ├── interaction_matrix.npy           M, (K, G)
│   ├── interaction_pvals_fdr.npy        corrected p, (K, G) (named "fdr" whichever correction is used)
│   ├── interaction_stats.json           correction, permutations, K, G, number of corrected p < 0.05, null |D| mean and sd
│   ├── manifest.json                    arguments, array shapes, variance explained, number of networks, noise fraction
│   └── state_content_summary.json       from state_content.py
└── {model}_av_integration/{CONFIG_TAG}/
    ├── vertex_av_integration_category.dlabel.nii
    ├── vertex_av_integration_code.npy   category code per grayordinate (-1 = noise)
    └── network_av_integration_report.json   per network: category, n_sig, effect, binding_gain = n_sig[av] - n_sig[avscramble], atlas cross-reference
```

Embeddings are read from `{embeddings-dir}/{model}/bin{N}s_skip{N}s/{model}_{modality}.npy`.

### Helper modules

`io_cluster.py` (group-average loader, segment metadata, `.dlabel.nii` and per-channel CSV writers; re-exports the binning functions of `rsa/shared/rsa_utils.py`), `reduce.py` (`reduce_temporal`, `reduce_spatial`), `cluster_temporal.py` (`fit_hmm`, `sweep_states`), `cluster_spatial.py` (`cluster_spatial`, `spatial_report`), `interaction.py` (`interaction_matrix`, `block_permutation_test`). Running `reduce.py`, `cluster_temporal.py`, `cluster_spatial.py`, `interaction.py` or `io_cluster.py` directly executes a built-in self-check (`io_cluster.py` loads the real group average).

## 2. Clustering-method selection

Rows are units (grayordinates, or embedding channels) and columns are time points (fMRI time points, or the 626 aligned 5-second movie bins). Each unit's time series is z-scored over time (`--no-zscore-timeseries` disables it) and reduced directly, with no preliminary PCA. Six reductions: PCA, metric multidimensional scaling (MDS), Isomap, t-distributed stochastic neighbor embedding (t-SNE), FastICA, UMAP. MDS, t-SNE, UMAP and Isomap are fit on `--n-landmarks` (default 2000) randomly chosen landmark units (exact MDS is quadratic in the number of units and scikit-learn t-SNE has no transform); the remaining units are placed by a distance-weighted k-nearest-neighbor extension (`--extension-neighbors`, default 8; Isomap uses its own transform). Three clusterers act on each standardized embedding: k-means (`--kmeans-clusters` default 4), HDBSCAN (`--min-cluster-size` 100, `--min-samples` 10), BIRCH with `n_clusters=None` (`--birch-threshold` 0.5, `--birch-branching-factor` 50).

Grayordinates are restricted to a stimulus-driven mask: `--mask-cifti` (default `outputs/sitmulus_regressor_cifti/HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii`; the directory name is spelled that way on disk), kept where value `> --mask-threshold` (default 0). Label convention in every vertex `.npy` and `.dlabel.nii`: key 0 = outside the mask; keys `1..K` = clusters in increasing raw-label order, named `cluster_1`, ...; HDBSCAN noise among masked-in grayordinates gets one reserved key after the last cluster (recorded as `noise_label` in the report together with `n_masked_in` and `n_masked_out`). Channel results use CSV instead of dlabel (0 = noise).

### Fixed-parameter sweep

```bash
bash cluster/run_vertex_clustering.sh     # grayordinates, group average
python cluster/channel_timeseries_clustering.py --family peav   # channels (families: peav, nemotron_layer18_mp, topoomni_layer18_sheet_mp)
```

Channel family names (`--family`) and the embedding file lookup come from `MODEL_CONFIGS` in `cf_modeling/deprecated/channel_cca_analysis.py`; the channel input is the intact audiovisual embedding of the family, binned to the same 626 segments (5 s windows, 5 s delay).

The shell runner reads environment variables: `N_COMPONENTS=3`, `N_LANDMARKS=2000`, `KMEANS_CLUSTERS=4`, `MIN_CLUSTER_SIZE=100`, `MIN_SAMPLES=10`, `BIRCH_THRESHOLD=0.5`, `REDUCTIONS="pca mds isomap tsne fastica"` (UMAP is left out of the shell default; the Python default includes it), `CLUSTERERS="kmeans hdbscan birch"`, `DATA_BASE`, `OUTPUTS_BASE`, `GROUP_AVG_CIFTI`, `OUTDIR`, `CONDA_ENV`. The Python entry points accept all settings as flags (`--n-components {2,3}`, `--tsne-perplexity`, `--umap-neighbors`, `--force`, ...; see `--help`). Each result is cached and a rerun fills in missing combinations.

```text
outputs/cluster/group_average/_vertex/norm-zscore_raw/fixed_nc3_landmarks2000/
├── manifest.json
└── sreduce-pca_snc3/
    ├── spatial_vertex_components.npy
    ├── reduction_report.json
    └── sreduce-pca_snc3_scluster-kmeans_k4/
        ├── spatial_vertex_labels.npy
        ├── spatial_vertex_labels.dlabel.nii
        └── spatial_report.json
```

The tags are `sreduce-{method}_snc{components}[_method-specific settings]` and `scluster-{method}_{parameters}`, so changed settings never reuse a cached result. Channel results are written to `outputs/cluster/{family}/_channel_timeseries/norm-zscore_nc3_landmarks2000/` (`channel_timeseries_features.npy`, `channel_ids.json`, then per reduction `channel_components.npy` and per clustering `channel_labels.npy`, `channel_labels.csv`, `channel_report.json`).

### Hyperparameter selection

```bash
bash cluster/run_vertex_model_selection.sh [extra vertex_model_selection.py flags]
python cluster/channel_timeseries_clustering.py --family F   # first: creates the channel feature cache
bash cluster/cluster.sh channel_model_selection              # families peav and nemotron_layer18_mp in parallel
python cluster/channel_timeseries_model_selection.py --family F [--features FILE] [--regress-global]
```

Stage 1, reducers. Every configuration in the grid (landmarks, neighbors, perplexity, FastICA algorithm and contrast function, UMAP neighbors and minimum distance, for display dimensions `--components` 2 and 3 and latent dimensions `--latent-components-grid`, default 2,3,4,5,6,8,10) is embedded and scored on one fixed random sample of `--quality-sample-size` (default 2000) units with `quality = 0.4 * trustworthiness + 0.4 * continuity + 0.2 * (rho + 1) / 2`. Trustworthiness and continuity are averaged over `--quality-neighbors` (default 10, 30) nearest neighbors; continuity is trustworthiness with source and embedding swapped; `rho` is the Spearman correlation between pairwise distances before and after reduction. For each method and display dimension the configuration with the highest quality is selected (role `display_2d`, `display_3d`). The method-specific latent dimension (role `latent_best`) is chosen from a native criterion per dimension: PCA explained variance (maximize), MDS normalized stress, Isomap geodesic reconstruction error, t-SNE Kullback-Leibler divergence, FastICA normalized reconstruction mean squared error (all minimize), UMAP the quality above (maximize). The criterion curve over dimensions is made monotone and normalized to [0, 1] between its first and last grid points, and the dimension with the largest gap above the straight line between the endpoints (elbow) is selected.

Stage 2, clusterers. For every selected embedding, a grid is evaluated on a random sample of `--cluster-sample-size` (default 12,000) units: k-means `--kmeans-k-grid` (default 2,3,4,5,6,8,10,12,16,20), HDBSCAN `--hdbscan-min-cluster-size-grid` x `--hdbscan-min-samples-grid`, BIRCH `--birch-threshold-grid` x `--birch-branching-factor-grid`. Each candidate's selection score (defined only for 2 to `--max-selected-clusters` = 100 clusters) is `0.35 * (silhouette + 1) / 2 + 0.20 / (1 + Davies-Bouldin) + 0.15 * CH / (CH + 1000) + 0.15 * balance + 0.15 * assigned_fraction`, where CH is the Calinski-Harabasz index, `balance` is the normalized entropy of cluster sizes `-sum(p log p) / log(n_clusters)`, and `assigned_fraction` is one minus the noise fraction. The best candidate per selected embedding and clusterer is refit on all units (HDBSCAN `min_cluster_size` is multiplied by the ratio of all units to sampled units).

This internal score rewards few, well-separated clusters, so it is a candidate generator and not a criterion for the number of clusters. Downstream choice is by temporal differentiation (section below).

Outputs, `outputs/cluster/group_average/_vertex/norm-zscore_raw/` (channels: `outputs/cluster/{family}/_channel_timeseries_model_selection/norm-zscore_raw/`, or `..._globalregressed` with `--regress-global`):

| Path | Content |
|------|---------|
| `reduction_sweep.csv`, `reduction_quality_indices.npy` | quality and criterion value of every reducer configuration; evaluation sample indices |
| `dimension_selection_sweep.csv`, `selected_latent_dimensions.csv` | best configuration per method and dimension; elbow distance per dimension |
| `selected_reducers.csv`, `selected_reducers/`, `reduction_candidates/` | selected reducers (with `selection_role`), their embeddings, every candidate embedding |
| `clustering_sweep.csv`, `clustering_evaluation_indices.npy` | metrics and selection score of every clustering candidate |
| `selected_clusterings.csv` | winning configuration per selected embedding and clusterer (`full_fit_cluster_tag`, rescaled HDBSCAN size) |
| `selected_maps/{reducer_tag}/{reducer_tag}_{cluster_tag}/` | `spatial_vertex_labels.{npy,dlabel.nii}`, `spatial_report.json` (channels: `channel_labels.{npy,csv}`, `channel_report.json`) |
| `selected_maps_manifest.json`, `selection_manifest.json` | index of selected maps; objective weights, arguments, counts |
| `selected_clusterings_combined.csv` | channels only: one column of cluster keys per selected map |

`--skip-reduction-sweep` and `--skip-clustering-sweep` reuse existing tables; `--force` recomputes. `channel_timeseries_model_selection.py --features` defaults to the cache written by `channel_timeseries_clustering.py`; `--regress-global` first regresses the across-channel mean time series out of every channel and z-scores again. `python cluster/channel_timeseries_model_selection.py demo` runs a self-check of that regression.

### Review files for grayordinate maps

```bash
python cluster/consolidate_vertex_outputs.py [--selection-root DIR] [--stimulus-map FILE] [--delete-duplicates]
```

Merges the selected one-map dlabels into `review_ciftis/` under the selection root: `best_vertex_clusterings.dlabel.nii` (map names `{reducer}_{2d|3d|best}_{clusterer}_best`), `selected_clusterings_all-2d_3d_bestdim.dlabel.nii`, `selected_{kmeans,hdbscan,birch}_clusterings_2d_3d_bestdim.dlabel.nii`, `selected_clusterings_all-2d_3d_stimregressor.dlabel.nii` (same maps restricted to the stimulus mask), plus `map_index.csv`, `combined_maps_manifest.json` and, in the selection root, `LOOK_HERE.md`. Each map keeps its own label table. With `--delete-duplicates`, the per-run dlabels are removed after the merged files pass a round-trip check; labels `.npy`, reports, embeddings and sweep tables are kept.

### Temporal differentiation screen

```bash
bash cluster/cluster.sh temporal_differentiation     # python cluster/screen_temporal_differentiation.py
```

For every selected clustering (vertex side: group-average grayordinates binned to the 626-segment grid; channel side: families `peav`, `nemotron_layer18_mp`), compute each cluster's mean time series, z-score it, form the Pearson correlation matrix between cluster means, and report `mean_abs_off_diag` and `max_abs_off_diag`, the mean and maximum absolute off-diagonal correlation. Values near 1 mean the clusters are copies of one time course. Output: `outputs/cluster/_channel_vertex_alignment_screen.csv` (columns include `side`, `family`, `reducer_tag`, `cluster_tag`, `n_clusters`, `n_noise`, `selection_score`, `mean_abs_off_diag`, `max_abs_off_diag`).

### Notebooks

`vertex_cluster_scatterplots.ipynb` (parameter `MODEL`, default `nemotron_layer18_mp`, only namespaces the figure directory) and `channel_cluster_scatterplots.ipynb` (parameter `FAMILY`, default `nemotron_layer18_mp`; also `peav`) read the selection outputs above and draw: reducer sweeps and dimension criteria, unclustered embeddings, clustering metric sweeps, selected cluster plots, and a comparison of each solution's silhouette against its temporal differentiation. Figures go to `{selection root}/figures/`.

## Tests

```bash
pytest tests/test_vertex_clustering.py tests/test_vertex_model_selection.py -v
```
