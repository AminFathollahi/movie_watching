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

- **Clustering-method-selection pipeline** — `vertex_clustering.py` /
  `vertex_model_selection.py` and their channel-embedding analogues
  `channel_timeseries_clustering.py` / `channel_timeseries_model_selection.py`,
  plus `consolidate_vertex_outputs.py`. This compares
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
  `vertex_clustering.py`: reduces and clusters embedding channels
  (rows) across the 626 aligned 5-second movie bins (columns) instead of
  grayordinates across fMRI TRs; reuses the same reduction/clustering code,
  writes labels to CSV (no grayordinate axis to save as `.dlabel.nii`). Like
  the vertex side, reductions are fit directly on the movie-bin time series —
  no preliminary PCA; there is no stimulus-driven mask either (channels
  aren't spatial).
- `channel_timeseries_model_selection.py` — channel analogue of
  `vertex_model_selection.py`; same sweep/selection math, CSV
  output instead of CIFTI.
- `consolidate_vertex_outputs.py` — merges the selected vertex
  cluster maps into review CIFTIs (see the "Hyperparameter
  selection workflow" section below for details); `--delete-duplicates` removes
  the superseded per-run dlabels after a round-trip validation passes.
- `screen_temporal_differentiation.py` — screens selected clusterings by temporal distinctness: computes mean/max off-diagonal Pearson correlation among cluster-mean profiles. Solutions with near-redundant profiles (r > 0.9) are flagged as uninformative. Writes `outputs/cluster/_channel_vertex_alignment_screen.csv`.

## Vertex dimensionality reduction and clustering

Run the full group-average sweep with:

```bash
bash cluster/run_vertex_clustering.sh
```

Vertices are first restricted to the stimulus-driven mask (`--mask-cifti`,
default `outputs/sitmulus_regressor_cifti/HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii`,
value > `--mask-threshold`, default 0 — 68885/108441 cortical grayordinates show positive stimulus correlation). The script
z-scores each masked-in grayordinate over time and creates three-dimensional
embeddings directly from the raw time series with PCA, metric MDS, Isomap,
t-SNE, FastICA, and UMAP — no preliminary PCA reduction. Set `N_COMPONENTS=2`
to use two dimensions instead. Every embedding is clustered with k-means
(`k=4`), HDBSCAN, and BIRCH with `n_clusters=None`.

MDS, t-SNE, and UMAP are fitted to a reproducible landmark sample because exact MDS
is quadratic in the number of grayordinates and scikit-learn t-SNE has no
out-of-sample transform. Their remaining grayordinates are embedded with a
distance-weighted k-nearest-neighbor extension. Isomap uses the same landmarks
and its native transform. Every approximation and parameter is recorded in
JSON alongside the outputs.

The default output root is:

```text
outputs/cluster/group_average/_vertex/norm-zscore_raw/
└── fixed_nc3_landmarks2000/
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
LabelAxis map name is the full config tag; label key 0 is transparent
`not_stimulus_driven` (every vertex outside the mask), cluster keys are
consecutive one-based values named `cluster_1`, `cluster_2`, and so on, and
HDBSCAN noise among masked-in vertices (if any) gets its own reserved key
(`noise`) one past the last cluster — never 0, never a cluster id. This same
convention is baked into `spatial_vertex_labels.npy` (via
`vertex_clustering.expand_masked_labels`), not just the dlabel. Each report
records `noise_label`, `n_masked_in`, and `n_masked_out`.

All settings are CLI flags. The shell runner also exposes the main settings as
environment variables, for example:

```bash
N_COMPONENTS=2 N_LANDMARKS=3000 \
REDUCTIONS="pca isomap fastica" \
bash cluster/run_vertex_clustering.sh
```

Because each result is independently cached, rerunning resumes missing
reduction or clustering combinations. Pass `--force` to the Python entry point
to overwrite complete results.

## Hyperparameter selection workflow

To tune the reducers before clustering, and then tune the clustering methods
on each selected 2-D, 3-D, and method-specific latent-best embedding, run:

```bash
bash cluster/run_vertex_model_selection.sh
```

The reducer sweep evaluates geometry preservation on a fixed grayordinate
sample. Its selection score combines trustworthiness, continuity, and
Spearman agreement between high- and low-dimensional distances. It
saves every candidate embedding and selects one configuration for each
reducer at each dimensionality. The default dimension grid is
`2,3,4,5,6,8,10`. Latent-dimension selection is a diminishing-returns elbow
criterion: each method's per-dimension quality curve is compared against the
straight chord connecting its first and last grid points, and the dimension
with the largest gap above that chord wins. PCA uses an explained-variance
curve, MDS normalized stress, Isomap geodesic reconstruction error, t-SNE KL
divergence, FastICA normalized reconstruction MSE, and UMAP geometry-quality
saturation as the underlying per-dimension quality metric. UMAP additionally
sweeps landmark count, `n_neighbors`, and `min_dist`.

The clustering sweep then tests multiple `k` values for k-means, multiple
`min_cluster_size`/`min_samples` combinations for HDBSCAN, and multiple
threshold/branching-factor combinations for BIRCH, recording silhouette,
Davies-Bouldin, Calinski-Harabasz, cluster-size balance, and assigned-data
coverage for every candidate. Each candidate's selection score is a composite
of these internal validity indices and its cluster-size balance. The
pipeline's internal ranking of these candidates is not a defensible way to
choose cluster granularity (see
`RESULTS.md`'s methodological caveat) and is never used downstream to pick a
final solution — that selection is instead by temporal differentiation (mean
absolute off-diagonal correlation between cluster mean timecourses, gated by
assigned fraction >= 0.5; `screen_temporal_differentiation.py`). The full
sweep tables are retained, while full-cortex dlabels are generated for every
selected reducer/dimensionality x clustering-algorithm pair so any candidate
can be pulled up for downstream screening.

HDBSCAN's selected `min_cluster_size` is scaled by the ratio of full-cortex to
evaluation-sample size before the final fit. This preserves the selected
minimum cluster fraction instead of making the full fit artificially more
fragmented. Both the evaluation and scaled full-fit values are recorded in
`selected_clusterings.csv`.

For review, consolidate the selected one-map dlabels with:

```bash
python cluster/consolidate_vertex_outputs.py --delete-duplicates
```

This produces `best_vertex_clusterings.dlabel.nii` with all 54 selected maps,
plus algorithm-specific and stimulus-masked review CIFTIs. Its map names follow
`<reducer>_<2d|3d|best>_<clusterer>_best`, for example
`umap_best_kmeans_best` and `isomap_3d_hdbscan_best`.
`review_ciftis/combined_maps_manifest.json` and `map_index.csv`
are the authoritative map indices. `LOOK_HERE.md` identifies the intended
review files; redundant individual and superseded dlabels are removed only
after the consolidated files pass a round-trip validation.

The executed notebooks are `cluster/vertex_cluster_scatterplots.ipynb` and
`cluster/channel_cluster_scatterplots.ipynb`. Each has a single `MODEL`/`FAMILY`
parameter at the top (default `nemotron_layer18_mp`; the channel notebook also
accepts `peav`) that namespaces its figure output directory, so both models can
be executed into separate figure trees without editing anything else. Both
notebooks include:

- reducer hyperparameter and dimension sweeps showing trustworthiness,
  continuity, distance-rank correlation, and each method's named criterion;
- unclustered 2-D and 3-D embeddings containing every channel or vertex;
- separate 2-D, 3-D, and latent-best sweeps showing silhouette,
  Davies–Bouldin, Calinski–Harabasz, cluster-size entropy, and assigned fraction;
- 36 selected display-space cluster plots;
- 36 latent-best label projections onto 2-D and 3-D spaces;
- 2 projections of the global latent-best winner;
- cluster-evidence diagnostics comparing each selected solution's internal
  validity (silhouette) against its temporal differentiation (mean absolute
  off-diagonal correlation between cluster mean profiles).

The executed vertex notebook contains 114 indexed figures (identical content
per model, since vertex clustering is shared across families); the executed
channel notebook contains 112 (peav) to 114 (nemotron_layer18_mp), since the
active set of reducer hyperparameters swept differs slightly by family.

Outputs are written under:

```text
outputs/cluster/group_average/_vertex/norm-zscore_raw/
├── reduction_sweep.csv
├── selected_reducers.csv
├── clustering_sweep.csv
├── selected_clusterings.csv
├── reduction_candidates/
├── review_ciftis/
│   ├── best_vertex_clusterings.dlabel.nii
│   ├── selected_clusterings_all-2d_3d_bestdim.dlabel.nii
│   ├── selected_clusterings_all-2d_3d_stimregressor.dlabel.nii
│   ├── selected_kmeans_clusterings_2d_3d_bestdim.dlabel.nii
│   ├── selected_hdbscan_clusterings_2d_3d_bestdim.dlabel.nii
│   └── selected_birch_clusterings_2d_3d_bestdim.dlabel.nii
└── figures/
```

All grids are configurable from the Python CLI. The workflow is resumable.
