# Clustering analyses

This directory holds two independent pipelines that answer different questions:

- **Stimulus-state x brain-network interaction analysis** — `run_cluster.py` +
  `cluster.sh`, with helpers `reduce.py`, `cluster_temporal.py`,
  `cluster_spatial.py`, `interaction.py`, `state_content.py`, and
  `av_integration.py`. This crosses temporal stimulus states (HMM over reduced
  movie-embedding features) against one spatial brain-network solution
  (HDBSCAN over reduced grayordinate features), tests their interaction via
  block-permutation, characterizes what each state represents
  (`state_content.py`), and tests AV-integration cartography against
  scrambled/dummy controls (`av_integration.py`). This is a **complete, frozen
  scientific result** — not deprecated, just not where new work happens.

- **Clustering-method-selection pipeline** — `voxel_timeseries_clustering.py` /
  `voxel_timeseries_model_selection.py` and their channel-embedding analogues
  `channel_timeseries_clustering.py` / `channel_timeseries_model_selection.py`,
  plus `consolidate_voxel_timeseries_outputs.py`. This compares
  dimensionality-reduction and clustering methods directly on raw
  grayordinate or embedding-channel time series, independent of the HMM/
  interaction machinery above. **This is the entry point for new clustering
  work.**

## Script reference

- `state_content.py` — characterizes what each HMM temporal state represents in
  stimulus terms (dominant video_id(s) and time ranges; no caption/transcript
  text exists for this stimulus set, so "meaning" here is video identity + timing).
- `av_integration.py` — tests AV-integration cartography per brain network: a
  network is "av_integrative" only if the real-AV state partition beats all
  three controls (avscramble, clsav_from_a, clsav_from_v) in FDR-significant
  cell count, then cross-references surviving networks against the Glasser atlas.
- `channel_timeseries_clustering.py` — channel analogue of
  `voxel_timeseries_clustering.py`: reduces and clusters embedding channels
  (rows) across the 626 aligned 5-second movie bins (columns) instead of
  grayordinates across fMRI TRs; reuses the same reduction/clustering code,
  writes labels to CSV (no grayordinate axis to save as `.dlabel.nii`).
- `channel_timeseries_model_selection.py` — channel analogue of
  `voxel_timeseries_model_selection.py`; same sweep/selection-score math, CSV
  output instead of CIFTI.
- `consolidate_voxel_timeseries_outputs.py` — merges the selected voxel-timeseries
  cluster maps into five authoritative review CIFTIs (see the "Hyperparameter
  selection workflow" section below for details); `--delete-duplicates` removes
  the superseded per-run dlabels after a round-trip validation passes.

## Voxel-timeseries dimensionality reduction and clustering

Run the full group-average sweep with:

```bash
bash cluster/run_voxel_timeseries_clustering.sh
```

The script z-scores each grayordinate over time, fits one shared 50-component
PCA for denoising and computational tractability, and creates three-dimensional
embeddings with PCA, metric MDS, Isomap, t-SNE, FastICA, and UMAP. Set
`N_COMPONENTS=2` to use two dimensions instead. Every embedding is clustered
with k-means (`k=4`), HDBSCAN, and BIRCH with `n_clusters=None`.

MDS, t-SNE, and UMAP are fitted to a reproducible landmark sample because exact MDS
is quadratic in the number of grayordinates and scikit-learn t-SNE has no
out-of-sample transform. Their remaining grayordinates are embedded with a
distance-weighted k-nearest-neighbor extension. Isomap uses the same landmarks
and its native transform. Every approximation and parameter is recorded in
JSON alongside the outputs.

The default output root is:

```text
outputs/cluster/group_average/_voxel_timeseries/
└── norm-zscore_prepca50_nc3_landmarks2000/
    └── sreduce-pca_snc3/
        ├── spatial_vertex_components.npy
        ├── reduction_report.json
        └── sreduce-pca_snc3_scluster-kmeans_k4/
            ├── spatial_vertex_labels.npy
            ├── spatial_vertex_labels.dlabel.nii
            └── spatial_report.json
```

The other reduction/clustering combinations follow the same config-tag grammar
already used by `run_cluster.py`: `sreduce-{method}_snc{components}` and
`scluster-{method}_{parameters}`. Method-specific manifold settings are added
to the reduction tag (for example, `_landmarks2000_nn15` for Isomap), ensuring
that changed settings never collide with cached results. The internal CIFTI
LabelAxis map name is the full config tag, label key 0 is transparent
`unassigned`, and cluster keys are consecutive one-based values named
`cluster_1`, `cluster_2`, and so on. Raw estimator labels remain available in
`spatial_vertex_labels.npy`; the JSON report records the raw-to-CIFTI key
mapping.

All settings are CLI flags. The shell runner also exposes the main settings as
environment variables, for example:

```bash
N_COMPONENTS=2 N_LANDMARKS=3000 \
REDUCTIONS="pca isomap fastica" \
bash cluster/run_voxel_timeseries_clustering.sh
```

Because each result is independently cached, rerunning resumes missing
reduction or clustering combinations. Pass `--force` to the Python entry point
to overwrite complete results.

## Hyperparameter selection workflow

To tune the reducers before clustering, and then tune the clustering methods
on each selected 2-D, 3-D, and method-specific latent-best embedding, run:

```bash
bash cluster/run_voxel_timeseries_model_selection.sh
```

The reducer sweep evaluates geometry preservation on a fixed grayordinate
sample. Its selection score weights trustworthiness (40%), continuity (40%),
and Spearman agreement between high- and low-dimensional distances (20%). It
saves every candidate embedding and selects one configuration for each
reducer at each dimensionality. The default dimension grid is
`2,3,4,5,6,8,10`. PCA uses an explained-variance elbow, MDS normalized stress,
Isomap geodesic reconstruction error, t-SNE KL divergence, FastICA normalized
reconstruction MSE, and UMAP geometry-quality saturation to select one
latent-best dimension per method. UMAP additionally sweeps landmark count,
`n_neighbors`, and `min_dist`.

The clustering sweep then tests multiple `k` values for k-means, multiple
`min_cluster_size`/`min_samples` combinations for HDBSCAN, and multiple
threshold/branching-factor combinations for BIRCH. Selection combines
silhouette, Davies-Bouldin, Calinski-Harabasz, cluster-size balance, and
assigned-data coverage. The full sweep tables are retained, while full-cortex
dlabels are generated for the selected configuration of each clustering
algorithm on every selected reducer/dimensionality pair.

HDBSCAN's selected `min_cluster_size` is scaled by the ratio of full-cortex to
evaluation-sample size before the final fit. This preserves the selected
minimum cluster fraction instead of making the full fit artificially more
fragmented. Both the evaluation and scaled full-fit values are recorded in
`selected_clusterings.csv`.

For review, consolidate the selected one-map dlabels with:

```bash
python cluster/consolidate_voxel_timeseries_outputs.py --delete-duplicates
```

This produces five authoritative files under `review_ciftis/`: one 18-map
CIFTI for each clustering algorithm,
`selected_clusterings_all-2d_3d_bestdim.dlabel.nii` with all 54 selected maps,
and `selected_clusterings_all-2d_3d_stimregressor.dlabel.nii` with those same
maps multiplied by the binary positive stimulus-regressor mask.
`review_ciftis/combined_maps_manifest.json` and `map_index.csv`
are the authoritative map indices. `LOOK_HERE.md` identifies the intended
review files; redundant individual and superseded dlabels are removed only
after the consolidated files pass a round-trip validation.

The executed notebook `cluster/voxel_timeseries_cluster_scatterplots.ipynb`
displays, in plain English grounded in the sweep CSVs, which hyperparameter
each reducer/clusterer selected and why, then plots only the 2-D and 3-D
embeddings:

- **Dimensionality selection plots**: 6 figures, one per reducer, showing its
  selection criterion against candidate dimensions with the chosen dimension
  and elbow distance annotated.
- **Clustering selection plots**: 3 figures, one per clusterer (k-means,
  HDBSCAN, BIRCH), each with one subplot per reducer showing the selection
  score across that clusterer's hyperparameter grid, with the 2-D and 3-D
  winners starred.
- **Individual scatterplots**: 36 figures (6 reducers × 3 clusterers × 2
  roles — 2-D and 3-D only), each named
  `scatterplot_<reducer>_<clusterer>_<2d|3d>.png`.

The method-specific latent-best ("bestdim") embeddings and their selected
clustering hyperparameters are still computed and their labels are included
in the consolidated dlabel CIFTIs above, but are not plotted or saved as
figures. All figures are saved in the model-selection output's `figures/`
directory with an index at `figures/scatterplot_index.csv`.

Outputs are written under:

```text
outputs/cluster/group_average/_voxel_timeseries_model_selection/
└── norm-zscore_prepca50/
    ├── reduction_sweep.csv
    ├── selected_reducers.csv
    ├── clustering_sweep.csv
    ├── selected_clusterings.csv
    ├── reduction_candidates/
    ├── review_ciftis/
    │   ├── selected_clusterings_all-2d_3d_bestdim.dlabel.nii
    │   ├── selected_clusterings_all-2d_3d_stimregressor.dlabel.nii
    │   ├── selected_kmeans_clusterings_2d_3d_bestdim.dlabel.nii
    │   ├── selected_hdbscan_clusterings_2d_3d_bestdim.dlabel.nii
    │   └── selected_birch_clusterings_2d_3d_bestdim.dlabel.nii
    └── figures/
```

All grids are configurable from the Python CLI. The workflow is resumable and
reuses the preliminary PCA generated by the baseline runner.
