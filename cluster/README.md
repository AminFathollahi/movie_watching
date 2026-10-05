# cluster: temporal states, spatial networks and clustering-method selection

Two independent pipelines share this directory.

| Pipeline | Question | Entry points |
|---|---|---|
| Interaction | Do movie-driven temporal states of a model embedding differ in how strongly they drive each brain network? | `cluster.sh groupaverage`, `run_cluster.py`, `state_content.py`, `av_integration.py` |
| Method selection | Which dimensionality-reduction and clustering methods give useful partitions of grayordinate time series and of embedding-channel time series? | `run_vertex_clustering.sh`, `run_vertex_model_selection.sh`, `cluster.sh channel_model_selection`, `consolidate_vertex_outputs.py`, `screen_temporal_differentiation.py` |

The environment is `movie`. Outputs go under `outputs/cluster/` (`--output-dir`).

## 1. Interaction pipeline

Notation: `V` = 108,441 cortical grayordinates (medial wall excluded), `S` = number of movie segments, `K` = number of temporal states, `G` = number of spatial networks.

### Method

1. Segment grid. The group-average fMRI time series `X` (`V x T`, one time point per second) is cut into windows of `--bin-sec` seconds spaced `--skip-sec` apart and delayed by `--delay-sec`, within each run (no window spans two runs). Each window is averaged over time points and z-scored within run, giving `fmri_binned` (`S x V`). The model embedding of the same windows (`process_model_embeddings`, no hemodynamic convolution, z-scored within run) gives `emb` (`S x n_features`).
2. Temporal states. `emb` is reduced by PCA to `--temporal-n-components` (default 50) components. A Gaussian hidden Markov model with `--n-temporal-states` (`K`, default 10) states and covariance `--hmm-covariance` (default `diag`) is fit with `hmmlearn`, with run lengths passed so that transitions never cross runs. The model is refit from `--hmm-n-init` (default 10) random seeds and the restart with the highest log-likelihood is kept. The outputs are a hard state label per segment and the state posteriors (`S x K`). A sweep over `K = k-min..k-max` (defaults 2 to 15) is written to `temporal_hmm_diagnostics.json` and does not influence `K`; for each `K` it records the leave-one-run-out held-out log-likelihood, the Bayesian information criterion of the full fit, and the adjusted Rand index between the best and second-best restart.
3. Spatial networks. `X` is reduced by PCA to `--spatial-n-components` (default 100) features per grayordinate, and HDBSCAN (`--min-cluster-size` 100, `--min-samples` 10) clusters grayordinates into networks; label `-1` is noise. This does not depend on the model, so it is computed once per spatial configuration and cached.
4. Interaction matrix. For state `k` and network `g`, `M[k, g]` is the mean over segments `s` in state `k` of the mean over grayordinates `v` in network `g` of `fmri_binned[s, v]`; noise grayordinates are excluded. The deviation is `D[k, g] = M[k, g] - m[g]`, with `m[g]` the mean of the network's segment means over all segments.
5. Block-circular permutation test. Within each run, the state labels are circularly shifted by an independent random offset (which keeps run boundaries and the autocorrelation of the state sequence) and `D` is recomputed, `--n-permutations` times (default 1000). The two-sided p-value of a cell is the fraction of permutations with `|D_null| >= |D_observed|`, floored at `1 / n_permutations`. `--perm-correction fdr` (default) applies Benjamini-Hochberg across all `K x G` cells; `maxstat` compares each `|D|` with the permutation distribution of the maximum `|D|` over cells.
6. State content (`state_content.py`). For each state: number of segments, total duration, dominant `video_id` and its share, segment counts per video, and merged stimulus time ranges per video.
7. Audiovisual integration (`av_integration.py`). Six conditions are fit with identical settings: `a`, `v`, `av`, and three control embeddings for the audiovisual modality, `avscramble` (real audio, temporally permuted video), `clsav_from_a` (real audio, blank video) and `clsav_from_v` (real video, blank audio). For network `g` and condition `c`, `n_sig[c]` is the number of states with corrected p below 0.05 in column `g` and `effect[c]` is the mean over states of `|M[k, g] - mean_k M[k, g]|`. A network is `unclassified` if no cell has corrected p below 0.05 in any condition. Otherwise the best condition is the one with the largest `(n_sig, effect)`: the network is `av_integrative` if the best condition is `av` and `n_sig[av]` is strictly greater than `n_sig` of each of the three controls, `av_nonspecific` if the best condition is `av` without that property or one of the controls, `auditory` if it is `a`, and `visual` if it is `v`. Each network is also compared with the Glasser parcellation: the top three parcels by share of the network's grayordinates, and the verdict `known:{parcel}` if the top share is at least 0.4, else `novel/mixed(...)`.

### Usage

```bash
bash cluster/cluster.sh groupaverage      # loops over MODELS and modalities set in cluster.sh
```

`cluster.sh` modes are `groupaverage`, `persubject` (not implemented), `temporal_differentiation` and `channel_model_selection` (section 2). Paths, `MODELS` (entries `model:modalities`) and the analysis settings are variables at the top of `cluster.sh`. The control conditions of `av_integration.py` are embeddings stored as `{model}_avscramble`, `{model}_clsav_from_a` and `{model}_clsav_from_v`, each with modality `av`.

Single run:

```bash
python cluster/run_cluster.py --model nemotron_layer27_mp --modality av \
    [--bin-sec 5 --skip-sec 5 --delay-sec 5] [--n-temporal-states 10] \
    [--temporal-source embedding|brain] [--n-permutations 1000 --perm-correction fdr|maxstat]
```

`--model` and `--modality {a,v,av}` are required; `--help` lists the input paths and the reduction and clustering options (`tphate` and `phate` need those packages). With `--temporal-source brain` the states come from `fmri_binned` and the interaction matrix is descriptive only (no permutation test). A run is skipped when `temporal_state_labels.npy`, `temporal_state_posteriors.npy`, `interaction_matrix.npy` and `manifest.json` already exist.

```bash
python cluster/state_content.py --model nemotron_layer27_mp --modality av --config CONFIG_TAG
python cluster/av_integration.py --model nemotron_layer27_mp --config CONFIG_TAG --spatial-config SPATIAL_TAG
```

`CONFIG_TAG` is the directory name under `{model}_{modality}/` (for example `treduce-pca_tnc50_K10_sreduce-pca_snc100_scluster-hdbscan_bin5s_skip5s_delay5s`) and `SPATIAL_TAG` the directory under `_spatial/` (for example `sreduce-pca_snc100_scluster-hdbscan_mcs100_ms10_bin5s_skip5s_delay5s`). Without arguments, both scripts run a self-check.

Embeddings are read from `{embeddings-dir}/{model}/bin{N}s_skip{N}s/{model}_{modality}.npy`.

### Outputs

```text
outputs/cluster/group_average/
├── _spatial/{SPATIAL_TAG}/
│   ├── spatial_vertex_labels.npy        (V,) raw labels, -1 = noise
│   ├── spatial_vertex_labels.dlabel.nii key 0 = unassigned, 1..G = networks
│   └── spatial_report.json              networks, noise fraction, silhouette, sizes, PCA variance
├── {model}_{modality}/{CONFIG_TAG}/
│   ├── temporal_state_labels.npy        (S,)
│   ├── temporal_state_posteriors.npy    (S, K)
│   ├── temporal_hmm_diagnostics.json
│   ├── interaction_matrix.npy           M, (K, G)
│   ├── interaction_pvals_fdr.npy        corrected p, (K, G); the name is the same for either correction
│   ├── interaction_stats.json           correction, permutations, K, G, number of corrected p < 0.05, null |D| mean and sd
│   ├── manifest.json                    arguments, array shapes, variance explained, networks, noise fraction
│   └── state_content_summary.json       from state_content.py
└── {model}_av_integration/{CONFIG_TAG}/
    ├── vertex_av_integration_category.dlabel.nii
    ├── vertex_av_integration_code.npy   category code per grayordinate (-1 = noise)
    └── network_av_integration_report.json   per network: category, n_sig, effect, binding_gain = n_sig[av] - n_sig[avscramble], atlas comparison
```

The helper modules (`io_cluster.py`, `reduce.py`, `cluster_temporal.py`, `cluster_spatial.py`, `interaction.py`) run a self-check when executed directly.

## 2. Clustering-method selection

Rows are units (grayordinates or embedding channels) and columns are time points (fMRI time points, or the 626 aligned 5-second movie bins). Each unit's time series is z-scored over time (`--no-zscore-timeseries` disables this) and reduced directly, without preliminary PCA. The reductions are PCA, metric multidimensional scaling (MDS), Isomap, t-distributed stochastic neighbor embedding (t-SNE), FastICA and UMAP. MDS, t-SNE, UMAP and Isomap are fit on `--n-landmarks` (default 2000) randomly chosen landmark units, because exact MDS is quadratic in the number of units and scikit-learn t-SNE has no transform; the remaining units are placed by a distance-weighted k-nearest-neighbor extension (`--extension-neighbors`, default 8; Isomap uses its own transform). Each standardized embedding is clustered with k-means (`--kmeans-clusters`, default 4), HDBSCAN (`--min-cluster-size` 100, `--min-samples` 10) and BIRCH with `n_clusters=None` (`--birch-threshold` 0.5, `--birch-branching-factor` 50).

Grayordinates are restricted to a stimulus-driven mask, `--mask-cifti` (default `outputs/sitmulus_regressor_cifti/HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii`, with the directory name spelled as shown), kept where the value exceeds `--mask-threshold` (default 0). In every vertex `.npy` and `.dlabel.nii`, key 0 is outside the mask and keys `1..K` are the clusters in increasing raw-label order, named `cluster_1`, ...; HDBSCAN noise among masked-in grayordinates gets one reserved key after the last cluster (reported as `noise_label`, with `n_masked_in` and `n_masked_out`). Channel results use CSV instead of dlabel (0 = noise).

### Fixed-parameter sweep

```bash
bash cluster/run_vertex_clustering.sh     # grayordinates, group average
python cluster/channel_timeseries_clustering.py --family peav   # channels; families: peav, nemotron_layer18_mp, topoomni_layer18_sheet_mp
```

The channel input is the intact audiovisual embedding of the family, binned to the same 626 segments (5 s windows, 5 s delay); family names and embedding files come from `MODEL_CONFIGS` in `cf_modeling/deprecated/channel_cca_analysis.py`. The shell runner takes its settings from environment variables (listed at the top of the script; its default `REDUCTIONS` omits UMAP, which the Python default includes), and the Python entry points take them as flags (see `--help`). Each result is cached and a rerun fills in missing combinations.

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

Tags have the form `sreduce-{method}_snc{components}[_method-specific settings]` and `scluster-{method}_{parameters}`, so a changed setting never reuses a cached result. Channel results are written to `outputs/cluster/{family}/_channel_timeseries/norm-zscore_nc3_landmarks2000/` (`channel_timeseries_features.npy`, `channel_ids.json`, then per reduction `channel_components.npy` and per clustering `channel_labels.npy`, `channel_labels.csv`, `channel_report.json`).

### Hyperparameter selection

```bash
bash cluster/run_vertex_model_selection.sh [extra vertex_model_selection.py flags]
python cluster/channel_timeseries_clustering.py --family F   # first: creates the channel feature cache
bash cluster/cluster.sh channel_model_selection              # families peav and nemotron_layer18_mp in parallel
python cluster/channel_timeseries_model_selection.py --family F [--features FILE] [--regress-global]
```

Stage 1, reducers. Every configuration in the grid (landmarks, neighbors, perplexity, FastICA algorithm and contrast function, UMAP neighbors and minimum distance; display dimensions `--components` 2 and 3; latent dimensions `--latent-components-grid`, default 2,3,4,5,6,8,10) is embedded and scored on one fixed random sample of `--quality-sample-size` (default 2000) units:

`quality = 0.4 * trustworthiness + 0.4 * continuity + 0.2 * (rho + 1) / 2`

Trustworthiness and continuity are averaged over `--quality-neighbors` (default 10 and 30) nearest neighbors; continuity is trustworthiness with source and embedding swapped; `rho` is the Spearman correlation between pairwise distances before and after reduction. For each method and display dimension the configuration with the highest quality is selected (role `display_2d`, `display_3d`). The latent dimension (role `latent_best`) is chosen from a method-specific criterion per dimension: explained variance for PCA (maximize); normalized stress for MDS, geodesic reconstruction error for Isomap, Kullback-Leibler divergence for t-SNE and normalized reconstruction mean squared error for FastICA (minimize); the quality above for UMAP (maximize). The criterion curve is made monotone and normalized to [0, 1] between its first and last grid points, and the dimension with the largest gap above the straight line between the endpoints (the elbow) is selected.

Stage 2, clusterers. For every selected embedding, a grid is evaluated on a random sample of `--cluster-sample-size` (default 12,000) units: k-means over `--kmeans-k-grid` (default 2,3,4,5,6,8,10,12,16,20), HDBSCAN over `--hdbscan-min-cluster-size-grid` x `--hdbscan-min-samples-grid`, and BIRCH over `--birch-threshold-grid` x `--birch-branching-factor-grid`. A candidate with 2 to `--max-selected-clusters` (100) clusters receives

`score = 0.35 * (silhouette + 1) / 2 + 0.20 / (1 + Davies-Bouldin) + 0.15 * CH / (CH + 1000) + 0.15 * balance + 0.15 * assigned_fraction`

where CH is the Calinski-Harabasz index, `balance = -sum(p log p) / log(n_clusters)` is the normalized entropy of cluster sizes, and `assigned_fraction` is one minus the noise fraction. The best candidate per selected embedding and clusterer is refit on all units (HDBSCAN `min_cluster_size` is multiplied by the ratio of all units to sampled units). Because this score favors few, well-separated clusters, it generates candidates and is not a criterion for the number of clusters; the downstream choice uses temporal differentiation (below).

Outputs are in `outputs/cluster/group_average/_vertex/norm-zscore_raw/` (channels: `outputs/cluster/{family}/_channel_timeseries_model_selection/norm-zscore_raw/`, or `..._globalregressed` with `--regress-global`):

| Path | Content |
|---|---|
| `reduction_sweep.csv`, `reduction_quality_indices.npy` | quality and criterion value of every reducer configuration; evaluation sample indices |
| `dimension_selection_sweep.csv`, `selected_latent_dimensions.csv` | best configuration per method and dimension; elbow distance per dimension |
| `selected_reducers.csv`, `selected_reducers/`, `reduction_candidates/` | selected reducers (with `selection_role`), their embeddings, every candidate embedding |
| `clustering_sweep.csv`, `clustering_evaluation_indices.npy` | metrics and selection score of every clustering candidate |
| `selected_clusterings.csv` | winning configuration per selected embedding and clusterer |
| `selected_maps/{reducer_tag}/{reducer_tag}_{cluster_tag}/` | `spatial_vertex_labels.{npy,dlabel.nii}` and `spatial_report.json` (channels: `channel_labels.{npy,csv}`, `channel_report.json`) |
| `selected_maps_manifest.json`, `selection_manifest.json` | index of selected maps; objective weights, arguments, counts |
| `selected_clusterings_combined.csv` | channels only: one column of cluster keys per selected map |

`--skip-reduction-sweep` and `--skip-clustering-sweep` reuse existing tables and `--force` recomputes. For channels, `--features` defaults to the cache written by `channel_timeseries_clustering.py`, and `--regress-global` first regresses the across-channel mean time series out of every channel and z-scores again; `python cluster/channel_timeseries_model_selection.py demo` self-checks that regression.

### Review files for grayordinate maps

```bash
python cluster/consolidate_vertex_outputs.py [--selection-root DIR] [--stimulus-map FILE] [--delete-duplicates]
```

Merges the selected one-map dlabels into `review_ciftis/` under the selection root: `best_vertex_clusterings.dlabel.nii`, `selected_clusterings_all-2d_3d_bestdim.dlabel.nii`, `selected_{kmeans,hdbscan,birch}_clusterings_2d_3d_bestdim.dlabel.nii` and `selected_clusterings_all-2d_3d_stimregressor.dlabel.nii` (the same maps restricted to the stimulus mask), plus `map_index.csv` and `combined_maps_manifest.json`; `LOOK_HERE.md` is written to the selection root. Each map keeps its own label table. With `--delete-duplicates`, the per-run dlabels are removed after the merged files pass a round-trip check; labels, reports, embeddings and sweep tables are kept.

### Temporal differentiation screen

```bash
bash cluster/cluster.sh temporal_differentiation     # runs cluster/screen_temporal_differentiation.py
```

For every selected clustering (grayordinates binned to the 626-segment grid; channel families `peav` and `nemotron_layer18_mp`), each cluster's mean time series is z-scored and the Pearson correlation matrix between cluster means is computed. The script reports `mean_abs_off_diag` and `max_abs_off_diag`, the mean and maximum absolute off-diagonal correlation; values near 1 mean that the clusters are copies of one time course. Output: `outputs/cluster/_channel_vertex_alignment_screen.csv` (columns include `side`, `family`, `reducer_tag`, `cluster_tag`, `n_clusters`, `n_noise`, `selection_score`, `mean_abs_off_diag`, `max_abs_off_diag`).

### Notebooks

`vertex_cluster_scatterplots.ipynb` (parameter `MODEL`, default `nemotron_layer18_mp`, which only names the figure directory) and `channel_cluster_scatterplots.ipynb` (parameter `FAMILY`, default `nemotron_layer18_mp`) read the selection outputs and draw the reducer sweeps, clustering metric sweeps, selected clusterings, and each solution's silhouette against its temporal differentiation, into `{selection root}/figures/`.

## Tests

```bash
pytest tests/test_vertex_clustering.py tests/test_vertex_model_selection.py -v
```
