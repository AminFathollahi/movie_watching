# Project Overview — Audiovisual Representation in Movie-Watching Cortex

Scientific status of this repository: what is claimed, what supports each claim, how the
modules feed each other, and what changed in the most recent work session. `README.md`
covers structure and how to run each pipeline; this document does not repeat that — it
covers the science and where each number in it lives on disk. Module-level detail lives in
`rsa/README.md`, `cluster/README.md`, `cluster/RESULTS.md`, `cf_modeling/README.md`,
`encoding/README.md`, `subcortical/README.md`, and `outputs/results_report.tex`
(the full write-up).

## 1. What this project claims

HCP 7T movie-watching fMRI (175 subjects, 4 runs, cortex + subcortex) is compared against
audiovisual transformer embeddings (PE-AV, CAV-MAE-sync, Omni3B, TopoOmni,
Omni-Embed-Nemotron-3B) on a shared 626-bin, 5-second-resolution movie timeline. Four
analysis families ask overlapping questions from that shared timeline: **RSA** (does the
brain's representational geometry match the model's?), **encoding** (can vertex activity be
predicted from the model?), **CF modeling** (do somatosensory and auditory/visual cortical
fields spatially converge, per Hedger et al. 2025?), and **clustering** (do temporal states
in the model and in the brain organize into comparable structure?). RSA is the primary
engine; encoding and CF modeling are supporting analyses run for triangulation, not as
independent headline results.

## 2. Core claims and evidence status

### 2.1 The primary result

Group-average searchlight RSA of PE-AV joint (`av`) embeddings against 108,441 cortical
grayordinates, k=100 neighborhoods, 5 s bins, 5 s skip, 5 s hemodynamic delay, Spearman,
175 subjects.
Source: `outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy`.

Verified directly from that array: peak ρ = 0.1830, mean ρ (>0) = 0.0424, 99.98% of
grayordinates positive.

### 2.2 The inference story — read this before citing any percentage from this project

**Four procedures were run against the same primary map and give four different answers.
They are not interchangeable, and the disagreement is now understood rather than
unresolved.**

| Procedure | Script | What it models | Result on the primary map |
|---|---|---|---|
| 1-factor t-test | `rsa/group_stats.py` (no `--n-blocks`) | Between-subject variance only, movie content held fixed | 108,389/108,441 (100.0%) FDR-sig, max t=43.67 |
| **2-factor block bootstrap (primary)** | `rsa/group_stats.py --n-blocks 16` → `rsa/shared/rsa_utils.py::corrected_2factor_bootstrap` | Between-subject **and** between-stimulus-block variance (Schütt et al. 2023 Eq. 5) | n_blocks=16: **6,963/108,441 (6.4%)**; n_blocks=8: 0/108,441; n_blocks=4: 0/108,441 |
| Circular-shift permutation | `rsa/perm_searchlight.py` | None modeled; null from run-aware temporal shifts of the model timeseries | 764/108,441 (0.70%) FDR-sig |
| Spin test | `rsa/run_spin_permutations.py` | Spatial autocorrelation only, single map vs. spatially-rotated copy of itself | 187/108,441 (0.17%) uncorrected, **0/108,441 after FDR** |

All four numbers reloaded and matched exactly against their source arrays/JSON
(`outputs/statistical_reconciliation.md`; the perm_searchlight FDR mask count was
independently re-verified here by summing
`outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_fdr_mask.dscalar.nii` = 764).

**Why the 16-block bootstrap is primary, not the 1-factor test.** The 1-factor test treats
the movie stimulus as a fixed condition set identical for every subject; with N=175
subjects, essentially any small, population-consistent bias — including one caused by a
handful of well-tracked movie segments that have nothing to do with genuine audiovisual
binding — reaches significance almost everywhere. The 2-factor bootstrap additionally
resamples movie blocks, asking whether the effect would survive under a different sample of
naturalistic content — the claim actually being made whenever this project says "where
integration lives." That is why 100.0% collapses to 6.4%: most of the cortex that looked
significant was significant only relative to between-subject noise, not between-stimulus
noise.

**The spin test trap.** `rsa/run_spin_permutations.py`'s CLI takes exactly one map — there
is no second/reference map anywhere in the script. Its null is
`null[s, v] = empirical[spin_indices[s, v]]`: each vertex's value under a spatial rotation
of the *same* map. This is a single-map, self-referential, local-hot-spot test — "does this
vertex stand out from a spatially shuffled copy of the same map" — not a test of the map
against zero, and not a two-map spatial-correspondence test in the classical
Alexander-Bloch (2018) sense. Because the primary map is diffusely, near-uniformly positive
(no sharp punctate peaks), a hot-spot-vs-shuffled-self test finds almost nothing **by
construction**, independent of whether the map's overall level is real. **The near-zero
spin-test result must never be cited as evidence against the effect or as a failed
spatial-autocorrelation control.** This has been misread twice already; see
`outputs/statistical_reconciliation.md` §2 for the full argument.

**Synthesis.** Procedures that model stimulus variance (2-factor bootstrap) or temporal
autocorrelation (circular-shift permutation) agree the effect is real but **spatially
concentrated**, not cortex-wide. Restricting the primary map to the 6,963 grayordinates
(6.4% of cortex) that survive the 2-factor bootstrap FDR gate at n_blocks=16 *sharpens* the
result rather than weakening it: mean ρ rises from 0.0424 (unmasked) to 0.0888 (masked,
2.1×), and the surviving territory concentrates in bilateral **TA2, A4, A5, TPOJ1, STGa**
(auditory-belt/parabelt and temporo-parietal association cortex) — anatomically consistent
with the independent five-model cross-architecture convergence analysis in
`outputs/results_report.tex` (auditory-association/parabelt: TA2, MBelt, PI, TPOJ1,
PBelt, STGa, area 52). That cross-check — restricting to the statistically defensible 6.4%
of cortex lands in the same territory implicated by a completely different method — is
stronger evidence than the raw "99.98% of vertices positive" descriptive number.

*One separate, unrelated t-test also produces a t≈34.7/99.97%-cortex figure that is easy to
confuse with the primary map's own 1-factor test (t=43.67/100.0%, above): it is the
per-subject paired intact-vs-scrambled random-effects test described in §2.4, not a second
inference procedure on the primary map itself.*

### 2.3 The 16-block power ceiling

Every 2-factor-corrected claim's effective N is **16 movie blocks**, not 175 subjects: df =
`min(n_subjects-1, n_blocks-1)` = 15. At the primary map's peak vertex, block-to-block
variance dominates subject-to-subject variance by roughly two orders of magnitude — this is
exactly the regime `tests/test_paired_bootstrap_resampling.py`'s synthetic self-check is
tuned to (`sigma_common=0.045` vs. `sigma_noise=0.02`, docstring: "matching the real PE-AV
data, where per-vertex block-to-block variance is ~2 orders of magnitude larger than
subject-to-subject variance"). Recruiting more subjects cannot raise power under this
regime. The two remedies are (a) restrict the comparison family to an *anatomically*
defined a priori ROI — never one derived from the map under test, which would be circular —
or (b) increase block count via finer temporal blocking. Neither has been run yet (§5).

### 2.4 Temporal-scramble binding: underpowered, not falsified

Two complementary consolidations of intact-vs-scrambled AV pairing exist, in different
currencies, both over the same 18 model/readout combinations (PE-AV, CAV-MAE-sync, Omni3B,
TopoOmni ×3 depths ×{mean-pool, last-token}, Nemotron ×4 layers):

- **Plain (non-partialled) RSA diff** (`rsa/scramble_diff_maps.py`): all **18/18**
  configurations show a positive mean diff — scrambling reduces raw RSA everywhere, not
  only its integration-contrast residual.
- **Partial-RSA integration-contrast binding map** (`rsa/temporal_scramble_binding.py`):
  **17/18** positive; CAV-MAE-sync is the one exception (mean −0.0024, 28.7% positive),
  plausibly because its own integration signal was already the weakest of the five
  architectures.

The direct per-subject test of this claim
(`rsa/scramble_paired_stats.py`, PE-AV av, N=175, paired diff = ρ_intact − ρ_scrambled per
subject per grayordinate) gives a **1-factor** paired t-test of 108,413/108,441 (99.97%)
FDR-significant, peak t=34.7 — this is the t≈34.7/99.97% figure, and it belongs to this
scramble test, not to the primary map's own 1-factor test (§2.2). Under the same **2-factor**
block-bootstrap correction used for the primary map (n_blocks=16), this paired diff drops
to **0/108,441 FDR-significant** (verified: `n_sig_c2f: 0` in
`scramble_paired_stats_175subs_nblocks16_summary.json`; the corresponding
BH-corrected p-value map's minimum, computed here directly from the stored
`sigmap_c2f`/`t_c2f` maps, is **p_fdr ≈ 0.085**, and the minimum one-sided uncorrected p at
that same df=15 is ≈0.0005 — the report's own framing of "near-miss," not the specific
"≈0.0028" uncorrected figure sometimes quoted for this result; see the discrepancy note at
the end of this document).

This is read as **underpowered, not falsified**: it is the same 16-block ceiling as §2.3,
now applied to a paired-difference statistic that is even more sensitive to shared
temporal autocorrelation between the two RSA time series being subtracted. The descriptive
18/18-positive result, the partial-RSA binding control, and the per-subject random-effects
t-test all converge; only this one specific block-resampling correction does not, and it is
plausibly over-conservative for a paired design rather than evidence of no effect
(`outputs/results_report.tex`, "Conservative-correction caveat," line ~801).

A suspected bug in the paired block bootstrap was investigated and **there is none**:
production code computes `diff_block = intact_block - scrambled_block` *before* calling
`corrected_2factor_bootstrap`, so resampling draws one joint subject/block sample that
applies to both conditions at once — the shared subject+block noise cancels by
construction. `tests/test_paired_bootstrap_resampling.py` demonstrates on synthetic data
that this joint scheme has near-nominal false-positive rate and usable power, while the
alternative (bootstrapping each condition independently, then summing variances) silently
and severely loses power by double-counting noise that should cancel.

### 2.5 Partial RSA — regressing out unimodal nuisance models

`rsa/partial_rsa.py` regresses nuisance model RDMs out of the PE-AV `av` target RDM before
brain comparison, configured via `rsa/shared/model_registry.py`. Three variants exist for
the primary map, all verified against
`outputs/rsa/raw/group_average/pe-av-small-16-frame_av_partial_corr*/k100_delay5s_bin5s_skip5s_spearman/partial_corr_r_searchlight.dscalar.nii`:

| Nuisance regressed out | Mean ρ | Peak ρ |
|---|---|---|
| AudioMAE only | 0.0413 | 0.1765 |
| VideoMAEv2-Large only | 0.0354 | 0.1766 |
| Both (AudioMAE + VideoMAEv2-Large) | 0.0344 | 0.1684 |

Partialling reduces peak ρ modestly (≈8%) and still leaves the map positive at nearly every
vertex — indicating PE-AV `av` encodes multimodal information not fully explained by either
unimodal model alone. Whether that residual is genuinely whole-cortex or concentrated in
the 6.4% bootstrap-significant core (§2.2) is not separately tested by partial RSA itself.

### 2.6 Channel-level modality vs. CCA preference — retracted and replaced

**Superseded 2026-09-02.** The earlier result in this section — an audio-only-minus-video-only
channel contrast of +0.077, 95% CI [0.029, 0.119], q=0.0024, with 155/166 subjects positive —
depended on thresholding two continuous quantities into a 4-class channel scheme
(`audio_only`/`video_only`/`both`/`neither`). That scheme has been deleted. The contrast does not
survive without it and should not be cited.

**What replaces it.** Two continuous per-channel axes, both signed, computed on run-wise z-scored
626-bin timeseries:

- modality axis = corr(intact AV, audio-only pass) − corr(intact AV, video-only pass)
- CCA axis = corr(intact AV, CCA-A ROI mean) − corr(intact AV, CCA-P ROI mean)

plus a partial variant of each (audio given video minus video given audio; anterior given
posterior minus posterior given anterior). Positive = audio-preferring and anterior-preferring.
Correlating the two axes across channels, with a moving-block bootstrap CI (block length 5 bins
≈ 25 s, 2000 resamples):

| Model | Variant | Spearman | Pearson |
|---|---|---|---|
| PE-AV (1024 ch) | zero-order | −0.037 [−0.064, +0.009] | −0.035 [−0.064, +0.008] |
| PE-AV | partial | −0.029 [−0.058, +0.014] | −0.024 [−0.056, +0.016] |
| Nemotron L18 mp (2048 ch) | zero-order | −0.025 [−0.045, −0.000] | −0.023 [−0.041, +0.000] |
| Nemotron L18 mp | partial | −0.025 [−0.043, −0.003] | −0.035 [−0.054, −0.008] |
| Topo-Omni L18 sheet mp (2048 ch) | zero-order | −0.000 [−0.049, +0.051] | +0.002 [−0.049, +0.049] |
| Topo-Omni L18 sheet mp | partial | +0.009 [−0.039, +0.056] | +0.012 [−0.036, +0.055] |

**Interpretation.** All six correlations are within ±0.04 of zero and five of six CIs include
zero; the one exclusion (Nemotron, partial) is small and not replicated in the other two
architectures. A channel's modality preference carries essentially no information about whether
it tracks anterior or posterior CCA territory. This is a null, and it is consistent across three
independent model families — the previous positive contrast was an artifact of the discarded
thresholding, not a weaker version of the same effect.

`cf_modeling/persubject_cca_channel_1pct.py` has been rewritten against these two axes (fixed
group-level modality axis; per-subject CCA axis from each subject's own top-1% mask) but has
**not been re-run**; the per-subject numbers previously quoted here are withdrawn, and
`outputs/cf_modeling/persubject_cca_1pct/README.md` still describes the retired class scheme.

Artifacts (table above verified directly against these JSONs, values unchanged):
`outputs/cf_modeling/channel_cca_preference/results/` —
`channel_modality_cca_correlation_peav_top1pct_peav.json`,
`channel_modality_cca_correlation_nemotron_layer18_mp_top1pct_nemotron_layer_18_mp.json`,
`channel_modality_cca_correlation_topoomni_layer18_sheet_mp_top1pct_topoomni_layer_18_mp.json`,
and six scatter PNGs (zero-order + partial per family):
`peav_zero_order_scatter.png`, `peav_partial_scatter.png`,
`nemotron_layer18_mp_zero_order_scatter.png`, `nemotron_layer18_mp_partial_scatter.png`,
`topoomni_layer18_sheet_mp_zero_order_scatter.png`,
`topoomni_layer18_sheet_mp_partial_scatter.png`. Scripts: `cf_modeling/channel_cca_analysis.py`,
runner `cf_modeling/run_channel_cca_analysis.sh`. Bootstrap settings recorded in the JSONs:
2000 resamples, block length 5 bins, seed 20260826, run_bins [151, 161, 156, 158], 626 total
bins.

### 2.7 Clustering: the k=2 bias, and how it was screened out

The clustering selection score shared by `cluster/voxel_timeseries_model_selection.py` and
`cluster/channel_timeseries_model_selection.py`
(`voxel_timeseries_model_selection.py:556-558`, verified) is:

```
0.35·silhouette_unit + 0.20·(1/(1+davies_bouldin)) + 0.15·calinski_harabasz_unit
+ 0.15·cluster_size_balance + 0.15·assigned_fraction
```

Silhouette dominates (35%) and silhouette rewards few, well-separated clusters, so the
score is structurally biased toward k=2. On the **vertex/voxel side**, k=2 wins every
selection role (display_2d 0.933, display_3d 0.922, latent_best 0.861 — all confirmed on
disk), and its two cluster-mean time series correlate at r≈0.97 — near-total temporal
redundancy, i.e. one shared stimulus-locked global signal split in two, not two
functionally distinct networks. On the **channel side**, the bias appears for nemotron and
topoomni but not for peav (peav's `latent_best` winner is k=16). Full tables:
`cluster/RESULTS.md` §1.

**The fix applied:** `cluster/screen_temporal_differentiation.py` computes, for every
candidate clustering, the mean |off-diagonal Pearson r| among that solution's own
cluster-mean profiles on the 626-bin timeline — low = temporally distinct clusters, high =
one signal split into near-duplicates. Vertex-side k=2 solutions score 0.938–0.974
(confirms redundancy); the vertex side never gets past *partial* differentiation even at
its best (plateaus at 0.406–0.55, no k tested reaches full separation). Channel-side peav
and nemotron reach 0.10–0.18 at k=12–44 (well-differentiated); **topoomni never drops below
≈0.27 at any k from 2 to 100** — no swept topoomni channel configuration is usable as a
differentiated clustering (`outputs/cluster/_channel_vertex_alignment_screen.csv`, 216
rows). **The top-`selection_score` row must never be taken as "the" clustering for a
downstream analysis without this screen.**

Visualized in `cluster/voxel_timeseries_cluster_scatterplots.ipynb` (vertex side) and
`cluster/channel_timeseries_cluster_scatterplots.ipynb` (channel side, same six
reducers/three clusterers, run per family).

### 2.8 Channel-vertex functional alignment

`cluster/channel_vertex_alignment.py` correlates each vertex cluster's mean profile
against each channel cluster's mean profile on the shared 626-bin timeline (the only
comparable quantity between disjoint index sets — 108,441 grayordinates vs. 1,024/2,048
embedding channels), tested against a circular-shift null (5,000 shifts) and again after
regressing the cortex-wide mean bin timeseries out of both sides ("global-signal control").
Pooled across the four temporally well-differentiated configurations (peav and nemotron, at
their k6×6 and k24×16 pairings; 201 significant pairs after global-signal control): |r|
range 0.106–0.369, mean 0.178 (`cluster/RESULTS.md` §3).

The internal control that makes this credible: the *poorly*-differentiated peav k6×6
config collapses from 15→3 significant pairs under global-signal control, while the
*well*-differentiated k24×16 configs are essentially unchanged (peav 67→65, nemotron
115→116). topoomni's raw counts are the highest of any family (34, 211) but are explicitly
**not** treated as evidence — per §2.7, no topoomni channel clustering differentiates at any
swept k, so its "significant pairs" are plausibly the same redundant signal counted
repeatedly.

### 2.9 Cross-model independence convention

Omni3B, TopoOmni, and Omni-Embed-Nemotron-3B all descend from Qwen2.5-Omni-3B and count as
one family for validation purposes — one member of this family must never drive another
member's clustering, localizer, or ROI definition. PE-AV is the architecturally independent
model used whenever an analysis needs a model that cannot validate itself (e.g. clustering
stimuli to test a TopoOmni readout).

### 2.10 A deliberate non-decision

A full repository restructure was considered and explicitly rejected: the audit found no
dead code and no over-abstraction, and the "legacy" `cluster/run_cluster.py` interaction
pipeline (HMM stimulus-state × HDBSCAN brain-network) is a complete, frozen scientific
result that shares `cluster/io_cluster.py` with the current clustering-selection pipeline —
not a candidate for deletion (`cluster/README.md`). Recorded here so it is not relitigated.

Two minor, unresolved naming issues, confirmed on disk, left as-is pending a decision:

- `rsa/channel_class_rsa.py` writes its output CIFTI into
  `outputs/cf_modeling/channel_cca_preference/results/rsa/`, not under `outputs/rsa/`.
- `encoding/av_derived_maps.py` and `rsa/av_derived_maps.py` are two different scripts
  (different derived-map families — encoding r vs. RSA rho, and different reference-model
  sets) that happen to share a filename.

One genuinely orphaned file: root-level `verify_subjects.py` (a standalone subject-list
sanity check, not wired into any `analysis.sh` or module).

### 2.11 Full-sheet RSA — the whole Topo-Omni sheet, not one layer

Every previous Topo-Omni result in this project used `topoomni_layer18_sheet_mp` — one
decoder layer, 4 of 304 rows of the sheet. This analysis covers all 304×512 = 155,648 units.

**Sheet geometry.** The model loaded is `Qwen2_5OmniThinkerForConditionalGeneration` — the
Thinker only; the Talker (speech generation) is never instantiated, so **the whole sheet is
the Thinker**. Three sub-modules inside it each carry their own `CorticalAdaptor` list:
vision encoder (`vision_config.depth=32`) at rows 0–159/cols 0–255 (40,960 units), audio
encoder (`audio_config.encoder_layers=32`) at rows 0–159/cols 256–511 (40,960 units), and the
Thinker text model (`text_config.num_hidden_layers=36`) at rows 160–303/all 512 cols (73,728
units, 47.4% of the sheet — larger than both perceptual towers combined). That third block is
what the extraction script calls the "decoder" and what upstream `Model
Repos/topo-omni/src/models/qwen2_5_omni.py` calls `multimodal_cortical_sheet` (line 197;
adaptor list at line 597): the autoregressive language backbone consuming fused audio+video
tokens, **not** a Whisper-style audio decoder and **not** the Talker. The old layer-18
reference is rows 232–235 of this block — 4 of 304 rows.

**Scripts:** `notebooks/feature_extraction/topo_omni_extract_full_sheet.py` (extraction,
intact joint-AV pass only), `rsa/full_sheet_rsa.py` + `rsa/run_full_sheet_rsa.sh` (searchlight
RSA). Full parameter and artifact listing: `rsa/README.md`, "Full-sheet RSA".

**Validation that this sheet is the same object as the earlier layer-18 embeddings:** the
extraction script's gate correlates all 36 decoder layers against the pre-existing
`topoomni_layer18_sheet_mp_av.npy` and asserts layer 18 is the unique argmax. Passed: layer 18
= 0.7089, runner-up layer 19 = 0.3041, margin 0.4049.

**The result.** 155,648 units × 626 bins, k=100, n_perm=1000, spearman, seed 42, 5 s bins / 5 s
skip / 5 s delay, true coordinates (`permute_coordinates`, seed 42). cca_a (anterior seed) rho
range [0.00535, 0.46769], mean 0.1157, 154,044/155,648 FDR-significant. cca_p (posterior)
range [0.00875, 0.23909], mean 0.0953, 155,648/155,648 FDR-significant. Per-block mean rho —
vision: cca_a 0.0395 / cca_p 0.0506; audio: 0.2732 / 0.1439; decoder: 0.0706 / 0.0931.

**The finding.** The anterior>posterior contrast is entirely an audio-encoder-tower effect: in
the vision tower and in the language backbone, posterior edges anterior instead. Both seeds
peak in audio (cca_a highest = audio layer 29, 0.3110; cca_p highest = audio layer 28, 0.1686)
and both trough at vision layer 0 (0.0115, 0.0176). diff (a minus p) mean +0.0205, sd 0.0710,
positive in only 31.16% of units — the audio tower is 26.3% of the sheet, which is what
reconciles a positive mean with a minority of positive units.

**Caveats — limits, not findings:**

- Cross-seed spatial Pearson r = 0.8639, top-decile hotspot Jaccard 0.7940. The two seed maps
  are largely the same map; the A-P contrast is a residual on a large shared signal, not two
  independent topographies.
- The hotspot contiguity test is saturated and must not be cited as evidence:
  `observed_mean_nn_dist` 1.00232 (cca_a) / 1.00343 (cca_p) against an integer-grid floor of
  exactly 1.0, versus a null of ~1.7059, p = 0.0005 — exactly 1/2001, the smallest value 2000
  permutations can produce. It also tests spatial clustering on coordinates generated by a
  topographic training objective, so clustering is true by construction. This is a second
  instance of the same failure mode already recorded for the spin test (§2.2) — a test that is
  null-by-construction on the map it is applied to.
- The `significance_caveat` string inside the delivered `metadata.json` is wrong: an
  over-strict 0.999 "at ceiling" gate fired on cca_a's 98.97%, so it reads "meaningfully below
  ceiling" for a map where only 1,604 of 155,648 units missed. The gate is now fixed at 0.98 in
  `rsa/full_sheet_rsa.py` and both branches interpolate the actual minimum fraction, but the
  delivered `metadata.json` predates the fix and was deliberately not retroactively edited.

**Artifacts:** `outputs/rsa/cca_seed_sheet_rsa/fullsheet_k100_truecoords/` — two `_rho.npy`,
two `.csv`, three PNG sheet maps (`cca_a_av_rho_sheet_map.png`, `cca_p_av_rho_sheet_map.png`,
`diff_rho_sheet_map.png`), `metadata.json`. See `rsa/README.md` for the full parameter and file
listing.

## 3. Analysis graph

The shared 626-bin, 5-second-resolution movie timeline is the **only valid comparison
space** between the fMRI grayordinate axis (108,441 cortical + subcortical units) and each
model's channel axis (1,024 for PE-AV, 2,048 for the Omni-family models) — these index sets
are otherwise disjoint and never compared directly.

```mermaid
flowchart TD
    STIM["Movie stimulus\n(HCP 7T, 4 runs)"]

    STIM --> FMRI["fMRI preprocessing\npreprocess_individual.py\nSG high-pass -> PSC -> GSR"]
    STIM --> EMB["Model embedding extraction\nnotebooks/feature_extraction/\nPE-AV, CAV-MAE-sync, Omni3B,\nTopoOmni, Nemotron"]

    FMRI --> TL["Shared 626-bin / 5s timeline\n(only valid comparison space:\n108,441 grayordinates <-> 1024/2048 channels)"]
    EMB --> TL

    TL --> RSA["rsa/\nsearchlight + Glasser RSA\nPRIMARY ENGINE"]
    TL --> ENC["encoding/\nhimalaya ridge encoding\nbanded-ridge variance partition"]
    TL --> CF["cf_modeling/\nconnective-field modeling\n(Hedger et al. 2025)"]
    TL --> CLU["cluster/\ntemporal-state / vertex / channel\nclustering"]

    RSA --> RSAOUT["Primary map\nPE-AV av searchlight rho\n108,441 grayordinates"]
    RSAOUT --> INF["Inference layer\n1-factor t-test | 2-factor bootstrap (PRIMARY)\ncircular-shift permutation | spin test\nsee sec 2.2"]

    RSA --> PARTIAL["Partial RSA / temporal-scramble\nbinding / cross-model convergence"]
    CF --> CCA["CCA-A / CCA-P ROI definition\n(from RSA top-1%)"]
    CCA --> CHCCA["cf_modeling channel CCA\npreference analysis"]
    CLU --> CVA["channel<->vertex alignment\n(screened by temporal\ndifferentiation first)"]
    PARTIAL --> CVA

    INF --> REPORT["outputs/results_report.tex"]
    PARTIAL --> REPORT
    ENC --> REPORT
    CCA --> REPORT
    CHCCA --> REPORT
    CVA --> REPORT
    CLU --> REPORT
```

## 4. Recent-session inventory

Everything below is new or untracked on disk (`git status`), read directly (docstring +
CLI/argparse) rather than inferred from filenames. Grouped by module; each line names the
output location where the script writes.

### `cf_modeling/`

- `LBOE_200_validation.md` — sensitivity check: 50/100/200 requested LBOEs give full-map
  held-out R² correlating ≥0.996 with the 200-LBOE fit; concludes no evidence of overfitting
  at 200 LBOEs, documents the `min(requested, n_L-2, n_R-2)` per-hemisphere cap.
- `bivariate_cifti.py` — shared pycortex-style 2-D bivariate colormap quantizer and CIFTI
  dlabel/legend writer; imported by every `export_*_bivariate_cifti.py` script below, not
  run standalone.
- `cf_naming.py` — small naming helper: parses/appends/strips the `_lboeN` suffix on
  analysis-ROI names without touching the underlying mask name.
- `channel_cca_analysis.py` — relates AV-sensitive embedding channels (audio/video winner
  and 4-class significance, block-bootstrap tests) to CCA-A/CCA-P ROI-mean timecourses for
  PE-AV, Nemotron-18-MP, and TopoOmni-18-sheet-MP; writes into
  `outputs/cf_modeling/channel_cca_preference/results/analyses/`.
- `channel_cca_notebook.py` — table/figure-building helpers (CIs, FDR, display labels) used
  by `channel_cca_preference.ipynb`; not run standalone.
- `channel_cca_preference.ipynb` — the executable report/visualization surface for the
  channel-CCA analysis; the only place these results are plotted. Writes
  `outputs/cf_modeling/channel_cca_preference/results/{analyses,rsa,tables,figures}/`.
- `compare_lboe_sensitivity.py` — summarizes held-out PE-AV CCA fits across requested LBOE
  counts into a JSON + markdown comparison table (feeds `LBOE_200_validation.md`).
- `export_bivariate_cifti.py` — exports one Hedger-style 2-D CF map (ROI-A × ROI-B R²_nc) as
  a quantized wb_view dlabel + legend, into the ROI pair's `cifti_maps/` directory.
- `export_partial_bivariate_cifti.py` — exports the 8-map partial-correlation CCA CIFTI as
  four 2-D dlabels (bilateral/within-hemisphere/L/R).
- `export_raw_corr_bivariate_cifti.py` — same, for the ordinary (non-partial) ROI-mean
  correlation CIFTI.
- `migrate_cca_artifacts.py` — dry-run-by-default renamer that migrates legacy `mca_v`/
  `mca_d` artifact names to the canonical `cca_a`/`cca_p` names, additive or in-place, with a
  JSON rollback manifest.
- `migrate_lboe_artifacts.py` — dry-run-by-default renamer that appends `_lboeN` qualifiers
  to historical unqualified fitted-CF-tree artifacts; writes
  `lboe_naming_migration_manifest.json`.
- `persubject_cca_channel_1pct.py` — per-subject top-1% PE-AV mask fitting and audio/video
  channel-preference contrast (§2.6 above); writes
  `outputs/cf_modeling/persubject_cca_1pct/{per_subject_contrast.csv, mask_dice_summary.json,
  group_summary.json}`.
- `roi_definitions.json` — central config for CCA-island ROI construction (thresholds,
  candidate-model ranking) consumed by `run_cca_islands.py`.
- `roi_mean_partial_connectivity.py` — computes bilateral/within-hemisphere/L/R ROI-mean
  partial-correlation maps (exact three-correlation identity, no ridge) between two CCA ROI
  timecourses and every grayordinate.
- `roi_mean_raw_connectivity.py` — same, ordinary (non-partial) Pearson correlation.
- `run_cca_islands.py` — thresholds the group-average PE-AV RSA map at 0.12 (or other
  configured cutoff/percent), selects the two largest bilateral temporal-lobe surface
  components per hemisphere, and writes reusable CCA-A/CCA-P masks + labeled dlabel.
- `run_channel_cca_analysis.sh` — driver for the channel-CCA analysis modes (`analyze`,
  `topoomni`, `nemotron-own-rois`).
- `status.md` — running lab-notebook status log for the CF-modeling/channel-CCA work
  session. Its block-bootstrap channel counts and CCA-A−CCA-P contrast (+0.077) belong to the
  retired 4-class scheme and are superseded — see §2.6. Also records that the per-subject CF
  run (`02_fit_cf_model.py --mode per_subject`, ROI pair `cca_a_peav_1pct_lboe100` ×
  `cca_p_peav_1pct_lboe100`) was deliberately halted 2026-09-03 to free the GPU, not a crash:
  26 subjects complete (8.6 GB) at `/media/amin/ADATA HD710 PRO/cf_modeling/per_subject/cca_a_peav_1pct_lboe100_cca_p_peav_1pct_lboe100/subjects/`;
  subject 140117 was killed mid-fit holding only its `.yml` config and `pipeline.log`, no
  `.npy`, so it is re-fit rather than skipped on resume. See `status.md` for the full entry
  and log path.

### `cluster/`

- `README.md` — new module documentation (the two independent pipelines, script reference,
  hyperparameter-selection workflow); see there for full usage.
- `RESULTS.md` — the write-up of the clustering-selection pipeline's results, including the
  k=2 bias, the temporal-differentiation screen, and the channel-vertex alignment findings
  quoted throughout §2.7–2.8 above; also flags its own internal cross-check that one
  upstream brief mislabeled channel-side scores as vertex-side (worth knowing if that
  labeling resurfaces elsewhere).
- `channel_timeseries_clustering.py` — channel analogue of `voxel_timeseries_clustering.py`:
  reduces/clusters embedding channels (rows) across the 626 movie bins (columns); writes
  CSV labels (no grayordinate axis).
- `channel_timeseries_model_selection.py` — channel analogue of
  `voxel_timeseries_model_selection.py`; same sweep/selection-score math, CSV output.
- `channel_vertex_alignment.py` — the channel↔vertex alignment test (§2.8); writes
  `outputs/cluster/{family}/_channel_vertex_alignment/{config pair}/alignment_manifest.json`
  + `alignment_long_globalctrl.csv`.
- `consolidate_voxel_timeseries_outputs.py` — merges selected voxel-timeseries cluster maps
  into five authoritative review CIFTIs under `review_ciftis/`; `--delete-duplicates` removes
  superseded per-run dlabels after round-trip validation.
- `run_voxel_timeseries_clustering.sh` — driver for the baseline voxel-timeseries
  reduction×clustering sweep.
- `run_voxel_timeseries_model_selection.sh` — driver for the hyperparameter-selection sweep.
- `screen_temporal_differentiation.py` — the k=2-bias screen (§2.7); writes
  `outputs/cluster/_channel_vertex_alignment_screen.csv` (216 rows: 54 vertex + 162 channel).
- `voxel_timeseries_cluster_scatterplots.ipynb` — executed notebook: plain-English summary
  of what each reducer/clusterer selected and why, plus 2-D/3-D scatterplots (45 figures)
  under the model-selection output's `figures/`.
- `voxel_timeseries_clustering.py` — baseline grayordinate reduction (PCA/MDS/Isomap/t-SNE/
  FastICA/UMAP) × clustering (k-means/HDBSCAN/BIRCH) sweep; writes
  `outputs/cluster/group_average/_voxel_timeseries/`.
- `voxel_timeseries_model_selection.py` — the two-stage hyperparameter-selection pipeline
  (reducer sweep via trustworthiness/continuity/rank agreement, then clustering sweep via
  the silhouette-weighted score in §2.7); writes
  `outputs/cluster/group_average/_voxel_timeseries_model_selection/`.

### `rsa/`

- `av_derived_maps.py` — appends nine descriptive AV comparison maps (conjunction,
  superadditivity, max-uni, each against `own`/`unimodal`/`text_aligned` reference
  families) to a group-average `_av` searchlight CIFTI.
- `channel_class_rsa.py` — **DEPRECATED** (2026-09-02). Searchlight RSA restricted to
  AV-sensitivity-classed embedding channels (audio_only/video_only/both/neither); the class
  scheme it consumes was deleted with §2.6. May be revived against the top-right / bottom-left
  quadrants of the two-axis preference scatterplots. Wrote into
  `outputs/cf_modeling/channel_cca_preference/results/rsa/` (naming note, §2.10).
- `max_uni.py` — per-subject AV-minus-max-unimodal RSA contrast with proper group inference
  (per-subject contrast computed first, then one-sample t-test + BH-FDR across subjects).
- `residualized_maps.py` — consolidates one AV model's `partial_correlation`,
  `linear_resid`, and `projection_resid` scalars into a single CIFTI, resumable per scalar.
- `run_channel_class_rsa.sh` — **DEPRECATED**, hard-exits. Was the driver for the
  12-analysis (3 models × 4 significance classes) channel-class RSA run.
- `run_spin_permutations.py` — restored verbatim from git history (was renamed/deleted, only
  a stale `.pyc` survived) — see §2.2 for the critical scope caveat on what this test
  actually measures.
- `full_sheet_rsa.py`, `run_full_sheet_rsa.sh` — the full-Topo-Omni-sheet searchlight RSA
  (§2.11).
- `cca_seed_sheet_rsa_truecoords.py`'s `characterize()` was fixed this session: its O(n²)
  `cdist` nearest-neighbour null did not scale from the 12,288-unit subset sheet to the full
  155,648-unit sheet (at a 15,565-point top decile, each of 2000 iterations allocated a ~1.94
  GB distance matrix, twice, once per seed) and was replaced by an exact `cKDTree`
  equivalent — numerically identical on a 4,000-point check (both 3.208852121618058,
  difference 0.0), 57.8x faster per iteration (0.534 s → 0.009 s), turning a >22 min hang into
  51 s. RNG draw order and iteration counts were left untouched, so the permutation results
  are the ones the original code would have produced had it finished.

### `encoding/`

- `av_derived_maps.py` — encoding-currency analogue of `rsa/av_derived_maps.py`: appends
  `av_superadditivity`/`av_conjunction` (+ stimulus-regressor-masked variants) to a
  group-average `encoding_pearson_r_audiovisual.dscalar.nii`. Same filename as, but
  functionally distinct from, `rsa/av_derived_maps.py` (§2.10).
- `migrate_encoding_output_names.py` — collision-safe renamer mapping legacy encoding
  artifact stems (`encoding_r2_unique_av`, `encoding_r_av`, …) to the current explicit
  naming schema, in both filenames and CIFTI map labels.
- `run_native_av_group_average.sh` — driver that reruns the canonical 5 s group-average
  encoding analysis for every native-AV model with matched A/V/AV embeddings, then appends
  the derived maps above.

### `tests/`

19 new test files, one per new module above plus targeted regression checks. The one
substantively worth knowing about independent of its module:
`test_paired_bootstrap_resampling.py` — synthetic-data proof that the paired scramble
block-bootstrap resamples the already-paired difference correctly (§2.4); run directly
(`python tests/test_paired_bootstrap_resampling.py`) or via pytest. The rest are
conventional unit tests for their same-named source script (naming/CLI/output-shape checks)
and are not separately itemized here.

### New artifacts under `outputs/`

- `outputs/statistical_reconciliation.md` — the source document for §2.2 and §2.4's
  inference numbers; independently re-verifies every headline figure quoted in
  `results_report.tex` against its underlying array/JSON and finds nothing wrong, plus
  surfaces the previously-unlisted circular-shift-permutation result.
- `outputs/_report_scripts/`, `outputs/_report_extraction/` — the extraction scripts and
  CSV/JSON outputs (masked-vs-unmasked descriptives, top parcels) that
  `statistical_reconciliation.md` and `results_report.tex` are built from.
- `outputs/cf_modeling/persubject_cca_1pct/` — §2.6's output tree.
- `outputs/cf_modeling/channel_cca_preference/` — §2.6/channel-CCA notebook's full
  `results/{analyses,rsa,tables,figures}/` tree.
- `outputs/cluster/{peav,nemotron_layer18_mp,topoomni_layer18_sheet_mp}/_channel_timeseries_model_selection/`
  and `.../_channel_vertex_alignment/` — §2.7/2.8's per-family channel-side outputs.
- `outputs/cluster/_channel_vertex_alignment_screen.csv` — the temporal-differentiation
  screen output (§2.7).
- New partial-RSA maps under `outputs/rsa/raw/group_average/pe-av-small-16-frame_av_partial_corr{,_audiomae_only,_videomae_only}/` — §2.5's three variants, all confirmed present with the
  quoted mean/peak ρ values.
- `outputs/model_embeddings/topoomni_fullsheet/bin5s_skip5s/topoomni_fullsheet_av.npy` —
  (626, 155648) float32, 389.7 MB, the full-sheet embedding behind §2.11.
- `outputs/rsa/cca_seed_sheet_rsa/fullsheet_k100_truecoords/` — §2.11's RSA output directory.

## 5. Open items

- **ROI-restricted binding test.** The cheap fix for the 16-block power ceiling (§2.3) —
  test the scramble binding contrast within an anatomically defined a priori ROI instead of
  whole-cortex — has not been run.
- **Per-subject clustering.** Evidence supports running it for peav and nemotron (both have
  temporally well-differentiated channel-side configurations whose alignment survives
  global-signal control, §2.7–2.8); topoomni has no such configuration at any swept k, so a
  per-subject run on its channel clusters would not obviously be interpretable.
- **TopoOmni channel re-sweep.** Unresolved whether topoomni's channel clustering's ≈0.27
  differentiation floor (§2.7) is a ceiling of the representation itself or of the swept
  reduction parameters (landmark count, `n_neighbors`, `min_dist`, dimension grid) — in
  flight, not yet run.
- **CCA-seed-to-TopoOmni-sheet RSA.** Done — see §2.11 (full 155,648-unit sheet, both seeds
  FDR-significant almost everywhere; the anterior>posterior contrast is an audio-tower
  effect, and the hotspot-contiguity control is saturated/uninformative).
- **`cf_modeling/persubject_cca_channel_1pct.py`** has still never been run under the
  two-axis specification (§2.6).
- **`rsa/topography_control.py`'s `random_neighbors`** draws its null across the whole unit
  pool, so it is confounded by per-layer rho differences (0.219 at layer 1 vs. 0.386 at layer
  18); a within-layer control is the correct version and has not been run.
- **`rsa/cca_seed_sheet_rsa.py` and `rsa/cca_seed_sheet_rsa_truecoords.py`** are
  near-duplicates (~875 diff lines) and want consolidating; both are untracked, so any
  consolidation must copy first and verify with `cmp`.

## Discrepancy notes (numbers that did not check out as originally stated, or needed correction)

- The commonly-quoted "t=34.7, ~99.97% of cortex significant" figure belongs to the
  **per-subject paired intact-vs-scrambled random-effects t-test** (§2.4;
  `scramble_paired_stats_175subs_summary.json`: `max_t_stat=34.69`, `n_sig_fdr=108413`), not
  to the primary map's own 1-factor t-test, which is a different, larger number:
  max t=43.67, 108,389/108,441 (99.976%, rounds to 100.0%)
  (`rsa/raw/groupstats/pe-av-small-16-frame_av/.../summary.json`).
- The plain (non-partialled) scramble-diff table is 18/18 positive; the partial-RSA
  (integration-contrast) binding map is 17/18 positive (CAV-MAE-sync negative). Both exist;
  "18/18" only describes the former.
- The scramble binding contrast's "best uncorrected p" was computed here directly from
  `scramble_paired_stats_175subs_nblocks16.dscalar.nii`'s stored `t_c2f` map (df=15): the
  minimum one-sided p is ≈0.00055 (two-sided ≈0.0011), not ≈0.0028. The **BH-FDR minimum
  p_fdr ≈ 0.085** and **0/108,441 vertices surviving FDR** were both independently confirmed
  (the latter directly from the JSON's `n_sig_c2f: 0`; the former by recomputing BH-FDR from
  the stored per-vertex t-statistics).
- `outputs/results_report.tex` currently contains an internally inconsistent value for the
  circular-shift permutation result: a table row states 108,181/108,441 (~99.8%) for
  `rsa/perm_searchlight.py`, while the same script's actual saved FDR mask
  (`..._fdr_mask.dscalar.nii`, re-summed directly here) contains exactly 764 significant
  vertices (0.70%), matching `outputs/statistical_reconciliation.md`. `results_report.tex`
  is being concurrently edited elsewhere in this work session; this document uses the
  disk-verified 764/0.70% figure throughout and flags the live `.tex` inconsistency rather
  than silently adopting either number.
