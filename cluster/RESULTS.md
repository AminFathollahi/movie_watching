# cluster/ results — 2026-09-10

Covers the clustering-method-selection pipeline (`vertex_*` /
`channel_timeseries_*`), not the separate `run_cluster.py` interaction
analysis (see "Older pipeline" below). This pipeline produces exactly two
things: cluster selection scores (section 1) and the off-diagonal
temporal-differentiation screen (section 2).

**All numbers below are current.** The channel-side sweep was re-run without
any preliminary PCA (channel embeddings are reduced directly from their
movie-bin time series, matching the vertex side) under
`outputs/cluster/<family>/_channel_timeseries_model_selection/norm-zscore_raw/`,
for **peav** and **nemotron_layer18_mp** only — topoomni_layer18_sheet_mp is
no longer part of this pipeline.

## Methodological caveat (read this before the numbers)

The clustering pipeline (`cluster/vertex_model_selection.py`, shared by the
channel-side script) picks a winning configuration per selection role via an
internal selection criterion, never reported here as a score (see below).
That criterion rewards few, well-separated clusters, so it is structurally
biased toward low k. Effect on disk: the **vertex-side** winner (`group_average`,
shared across both model families) is k=2 at **every** selection role:

| role | config | k | silhouette |
|---|---|---|---|
| display_2d | umap(2d) + kmeans | 2 | 0.8273 |
| display_3d | mds(3d) + hdbscan | 2 | 0.7472 |
| latent_best | mds(4d) + hdbscan | 2 | 0.7305 |

Source: `outputs/cluster/group_average/_vertex/norm-zscore_raw/selected_clusterings.csv`.

The **channel-side** `latent_best` winners show the same bias but not
uniformly: nemotron collapses to k=2, peav does not.

| family | config | k | silhouette |
|---|---|---|---|
| nemotron_layer18_mp | umap(5d) + kmeans | 2 | 0.3049 |
| peav | mds(5d) + kmeans | 16 | 0.1999 |

Source: `outputs/cluster/{nemotron_layer18_mp,peav}/_channel_timeseries_model_selection/norm-zscore_raw/selected_clusterings.csv`.

**Conclusion:** the pipeline's internal composite is valid for comparing
reducers at a fixed k (that is what it was built for — see
`cluster/README.md`'s reducer-then-clusterer workflow). It must **not** be
used to choose granularity. Every downstream analysis in this document
instead screens candidate clusterings by temporal differentiation (next
section) before using them.

## 1. Channel clustering hyperparameter optimization

Two families, channel counts confirmed from `n_assigned` in each
`selected_clusterings.csv` (noise-free rows): peav 1024, nemotron_layer18_mp
2048. Winning configuration per `selection_role` (argmax of the pipeline's
internal composite over all reducer x clusterer rows in each family's CSV):

| family | role | reducer | clusterer | k | silhouette |
|---|---|---|---|---|---|
| peav | display_2d | mds(2d) | kmeans | 8 | 0.3871 |
| peav | display_3d | umap(3d) | kmeans | 8 | 0.3130 |
| peav | latent_best | mds(5d) | kmeans | 16 | 0.1999 |
| nemotron_layer18_mp | display_2d | umap(2d) | kmeans | 5 | 0.4140 |
| nemotron_layer18_mp | display_3d | umap(3d) | kmeans | 2 | 0.4092 |
| nemotron_layer18_mp | latent_best | umap(5d) | kmeans | 2 | 0.3049 |

Source (one row per family): `outputs/cluster/{peav,nemotron_layer18_mp}/_channel_timeseries_model_selection/norm-zscore_raw/selected_clusterings.csv`.

## 2. Temporal-differentiation screen

`outputs/cluster/_channel_vertex_alignment_screen.csv`, produced by
`cluster/screen_temporal_differentiation.py`: 161 rows (54 vertex rows,
tagged `family=group_average`, shared across model families; 107 channel
rows, 53 peav + 54 nemotron_layer18_mp). Metric: mean |off-diagonal Pearson r|
among a candidate clustering's own cluster-mean profiles on the 626-bin movie
timeline — low mean = temporally distinct clusters, high mean = the "solution"
is one signal split into near-duplicate pieces (script docstring,
`cluster/screen_temporal_differentiation.py:1-6`).

- **Vertex side, k=2 solutions** (23 rows at k=2): mean |off-diag r|
  0.934–0.984 — near-total redundancy at k=2.
- **Vertex side, higher k**: noisy but broadly improving with k, never fully
  differentiating — best solutions are k=6 (t-SNE(4d)+HDBSCAN, 0.357) and
  k=34 (UMAP(4d)+HDBSCAN, 0.360); k=70 (PCA(5d) or FastICA(5d)+BIRCH) plateaus
  at 0.460.
- **Channel side, peav**: best-differentiated configurations (k=4–19) reach
  0.093–0.110 mean |off-diag r|.
- **Channel side, nemotron_layer18_mp**: best configurations (k=4–33) reach
  0.108–0.148 — differentiates less cleanly than peav but still far better
  than any vertex solution.

## Older pipeline (not superseded)

`cluster/run_cluster.py` and its helpers (`reduce.py`, `cluster_temporal.py`,
`cluster_spatial.py`, `interaction.py`, `state_content.py`,
`av_integration.py`) answer a different question — HMM stimulus-state x
HDBSCAN brain-network interaction, plus AV-integration cartography against
scrambled/dummy controls. Per `cluster/README.md:13-14`, this is "a complete,
frozen scientific result — not deprecated, just not where new work happens."
Nothing above touches or revises it.

## Open questions

- **Per-subject version worth running?** Evidence favors it for both
  families now that both have well-differentiated channel-side configurations
  (section 2): peav differentiates best (0.093 minimum), nemotron less
  cleanly (0.108 minimum) but still far better than any vertex solution.
