# cluster/ results — 2026-09-02

Covers the clustering-method-selection pipeline (`voxel_timeseries_*` /
`channel_timeseries_*` / `channel_vertex_alignment.py`), not the separate
`run_cluster.py` interaction analysis (see "Older pipeline" below).

## Methodological caveat (read this before the numbers)

The clustering selection score (`cluster/voxel_timeseries_model_selection.py:557-558`,
shared by the channel-side script) is:

```
0.35*silhouette + 0.20*(1/(1+davies_bouldin)) + 0.15*calinski_harabasz_unit
+ 0.15*cluster_size_balance + 0.15*assigned_fraction
```

Silhouette dominates (35%), and silhouette is maximized by few, well-separated
clusters. Effect on disk: the **vertex/voxel-side** winner (`group_average`,
shared across all three model families — see "Why this analysis takes this
form" below) is k=2 at **every** selection role:

| role | config | k | score | silhouette |
|---|---|---|---|---|
| display_2d | mds(2d) + hdbscan mcs50_ms50 | 2 | 0.9331 | 0.8852 |
| display_3d | mds(3d) + hdbscan mcs50_ms50 | 2 | 0.9217 | 0.8649 |
| latent_best | mds(4d) + hdbscan mcs50_ms25 | 2 | 0.8608 | 0.7145 |

Source: `outputs/cluster/group_average/_voxel_timeseries_model_selection/norm-zscore_prepca50/selected_clusterings.csv`.

The **channel-side** `latent_best` winners show the same bias but not
uniformly: nemotron and topoomni also collapse to k=2, peav does not.

| family | config | k | score |
|---|---|---|---|
| nemotron_layer18_mp | umap(5d) + kmeans | 2 | 0.6831 |
| topoomni_layer18_sheet_mp | umap(4d) + kmeans | 2 | 0.7095 |
| peav | mds(5d) + kmeans | 16 | 0.6203 |

Source: `outputs/cluster/{nemotron_layer18_mp,topoomni_layer18_sheet_mp,peav}/_channel_timeseries_model_selection/norm-zscore_prepca50/selected_clusterings.csv`.
(These three numbers match the task brief's figures exactly, but the brief
labeled them "vertex side" — on disk they are the per-family **channel**-side
`latent_best` winners; the true vertex-side selection is the k=2-at-every-role
table above. Flagging this as a mislabel, not a numeric discrepancy.)

**Conclusion:** the score is valid for comparing reducers at a fixed k (that
is what it was built for — see `cluster/README.md`'s reducer-then-clusterer
workflow). It must **not** be used to choose granularity. Every downstream
analysis in this document instead screens candidate clusterings by temporal
differentiation (next section) before using them.

## 1. Channel clustering hyperparameter optimization

Three families, channel counts confirmed from `n_assigned` in each
`selected_clusterings.csv` (noise-free rows): peav 1024, nemotron_layer18_mp
2048, topoomni_layer18_sheet_mp 2048. Winning (max `selection_score`)
configuration per `selection_role`, computed as the argmax over all reducer x
clusterer rows in each family's CSV:

| family | role | reducer | clusterer | k | score | silhouette |
|---|---|---|---|---|---|---|
| peav | display_2d | mds(2d) | kmeans | 8 | 0.7375 | 0.3854 |
| peav | display_3d | mds(3d) | kmeans | 6 | 0.6779 | 0.3011 |
| peav | latent_best | mds(5d) | kmeans | 16 | 0.6203 | 0.1996 |
| nemotron_layer18_mp | display_2d | umap(2d) | kmeans | 6 | 0.7630 | 0.4153 |
| nemotron_layer18_mp | display_3d | umap(3d) | kmeans | 2 | 0.7240 | 0.3666 |
| nemotron_layer18_mp | latent_best | umap(5d) | kmeans | 2 | 0.6831 | 0.2830 |
| topoomni_layer18_sheet_mp | display_2d | umap(2d) | kmeans | 6 | 0.7767 | 0.4539 |
| topoomni_layer18_sheet_mp | display_3d | umap(3d) | kmeans | 10 | 0.7062 | 0.3062 |
| topoomni_layer18_sheet_mp | latent_best | umap(4d) | kmeans | 2 | 0.7095 | 0.3393 |

Source (one row per family): `outputs/cluster/{peav,nemotron_layer18_mp,topoomni_layer18_sheet_mp}/_channel_timeseries_model_selection/norm-zscore_prepca50/selected_clusterings.csv`.

## 2. Temporal-differentiation screen

`outputs/cluster/_channel_vertex_alignment_screen.csv`, produced by
`cluster/screen_temporal_differentiation.py`: 216 rows confirmed (54 vertex
rows, all tagged `family=group_average`, shared across model families; 162
channel rows, 54 per family). Metric: mean |off-diagonal Pearson r| among a
candidate clustering's own cluster-mean profiles on the 626-bin movie
timeline — low mean = temporally distinct clusters, high mean = the "solution"
is one signal split into near-duplicate pieces (script docstring,
`cluster/screen_temporal_differentiation.py:1-6`).

- **Vertex side, k=2 solutions** (24 rows at k=2): mean |off-diag r| 0.938–0.974,
  a single degenerate outlier at 0.058 (isomap-derived umap(4d)+hdbscan,
  n_clusters=2 but effectively near-empty). Confirms near-total redundancy at
  k=2.
- **Vertex side, higher k**: improves monotonically with k but plateaus —
  k=24: 0.406–0.521, k=25: 0.537, k=28: 0.518, k=94 (max k tested): 0.406.
  Matches the claimed 0.4–0.55 plateau.
- **Channel side, peav**: best-differentiated configurations (k=12–16) reach
  0.101–0.177 mean |off-diag r|.
- **Channel side, nemotron_layer18_mp**: best configurations (k=10–44) reach
  0.131–0.150.
- **Channel side, topoomni_layer18_sheet_mp**: minimum mean |off-diag r|
  across **every** k tested (2 through 100) is 0.267, at k=100
  (mds(6d)+birch). No swept topoomni channel configuration differentiates
  below ~0.27.

## 3. Channel-vertex functional alignment

Method (why the analysis takes this form, `cluster/channel_vertex_alignment.py:1-11`):
vertex clusters live on 108,441 grayordinates/subcortical units (confirmed
from `group_average_raw_cortex_59k.dtseries.nii` shape `(3655, 108441)`) and
channel clusters live on 1024/2048 embedding channels — disjoint index sets,
so label overlap is meaningless. The only comparable quantity is each
cluster's mean profile on the shared 626-bin movie timeline; every pair
(vertex cluster x channel cluster) is tested against a circular-shift null
(5000 shifts) because the profiles are strongly autocorrelated and an i.i.d.
permutation null would be invalid.

Pairs tested, raw BH-FDR (q<0.05) significant, and significant after
regressing the cortex-wide mean bin time series out of both sides
("global-signal control"), from each `alignment_manifest.json`:

| family | vertex config (k) | channel config (k, role) | n_pairs | sig raw | sig, global ctrl |
|---|---|---|---|---|---|
| peav | v-pca5-kmeans6 (6) | c-mds3-kmeans6 (6, display_3d) | 36 | 15 | 3 |
| peav | v-tsne4-birch24 (24) | c-mds5-kmeans16 (16, latent_best) | 384 | 67 | 65 |
| peav | v-tsne4-hdbscan11 (11) | c-tsne6-kmeans12 (12, latent_best) | 132 | 56 | 43 |
| peav | v-mds4-hdbscan2 (2, latent_best) | c-mds5-kmeans16 (16, latent_best) | 32 | 13 | not run |
| nemotron_layer18_mp | v-pca5-kmeans6 (6) | c-mds3-kmeans6 (6, display_3d) | 36 | 21 | 17 |
| nemotron_layer18_mp | v-tsne4-birch24 (24) | c-mds2-kmeans16 (16, display_2d) | 384 | 115 | 116 |
| topoomni_layer18_sheet_mp | v-pca5-kmeans6 (6) | c-fastica3-kmeans6 (6, display_3d) | 36 | 34 | 22 |
| topoomni_layer18_sheet_mp | v-tsne4-birch24 (24) | c-mds3-kmeans16 (16, display_3d) | 384 | 211 | 194 |

All four requested spot-checks confirmed exactly: peav k6x6 15->3, peav
k24x16 67->65, nemotron k24x16 115->116, topoomni k24x16 211->194.

Effect sizes (|r|, BH-FDR-significant pairs after global-signal control) for
the four **credible** configurations — peav and nemotron k6x6 and k24x16,
the ones that are also temporally well-differentiated per section 2 — pooled
across 201 significant pairs: min 0.106, max 0.369, mean 0.178. Source:
`outputs/cluster/{peav,nemotron_layer18_mp}/_channel_vertex_alignment/{v-pca5-kmeans6_c-mds3-kmeans6,v-tsne4-birch24_c-mds5-kmeans16,v-tsne4-birch24_c-mds2-kmeans16}/alignment_long_globalctrl.csv`.
(topoomni and the hdbscan-based peav configuration were excluded from this
pooled range — see interpretation below; their raw |r| spans wider, up to
0.58, and are not treated as evidence of anything.)

**Interpretation.** The peav k6x6 configuration collapses from 15 to 3
significant pairs under the global-signal control, while the well-
differentiated k24x16 configurations (peav 67->65, nemotron 115->116) are
essentially unchanged. That contrast is the internal control that makes the
surviving alignment credible: a coarse, poorly-differentiated clustering
(peav channel k=6 mean |off-diag r| ~0.15–0.38 depending on config, see
manifest `channel_temporal_differentiation`) produces "significant" pairs
that are mostly explained by a shared global signal; a well-differentiated
one does not.

topoomni's raw counts (34 and 211) are the highest of any family but must
**not** be read as stronger evidence of alignment. Per section 2, no
topoomni channel-side configuration screened (any k, up to 100) differentiates
below mean |off-diag r| ~0.27 — its clusters are highly non-independent, so
many "significant pairs" are the same underlying redundant signal counted
multiple times. Both topoomni counts are flagged uninterpretable.

The vertex side never gets past partial differentiation either: the best
vertex configurations plateau at mean |off-diag r| ~0.4-0.55 (section 2), never
approaching the well-separated regime the score alone would suggest at k=2.
Every alignment result in this section rides on partial, not full, vertex
differentiation — a standing limitation, not a solved problem.

## Older pipeline (not superseded)

`cluster/run_cluster.py` and its helpers (`reduce.py`, `cluster_temporal.py`,
`cluster_spatial.py`, `interaction.py`, `state_content.py`,
`av_integration.py`) answer a different question — HMM stimulus-state x
HDBSCAN brain-network interaction, plus AV-integration cartography against
scrambled/dummy controls. Per `cluster/README.md:13-14`, this is "a complete,
frozen scientific result — not deprecated, just not where new work happens."
Nothing above touches or revises it.

## Open questions

- **Per-subject version worth running?** Evidence favors it for peav and
  nemotron: both have well-differentiated channel-side configurations
  (section 2) whose alignment survives the global-signal control essentially
  intact (section 3). topoomni has no such configuration at any swept k —
  running per-subject on channel clusters that don't temporally differentiate
  at the group level would not obviously produce an interpretable result.
- **Does topoomni's channel clustering need re-sweeping?** The current sweep
  (`outputs/cluster/topoomni_layer18_sheet_mp/_channel_timeseries_model_selection/`)
  never gets below mean |off-diag r| ~0.27 at any reducer/dimensionality/k
  tested up to 100. Unclear whether that is a ceiling of the representation
  itself or of the swept reduction parameters (landmark count, `n_neighbors`,
  `min_dist`, dimension grid) — would need a wider or different sweep to
  distinguish the two before drawing a conclusion either way.
