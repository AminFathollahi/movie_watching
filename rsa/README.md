# RSA — Representational Similarity Analysis

Vertex-wise searchlight RSA and Glasser parcel RSA comparing cortical fMRI representations to model embeddings from audiovisual transformer models.

## Structure

```
rsa/
├── analysis.sh                  # master runner — all config lives here
├── environment.yml                  # conda env: movie
├── searchlight.py               # geodesic neighbourhood RSA → .dscalar.nii
├── glasser.py                   # Glasser parcel RSA → .dscalar.nii + ranked_report.csv
├── group_stats.py               # aggregate per-subject maps → group stats CIFTI
├── run_spin_permutations.py     # spin-test (Alexander-Bloch 2018) significance for one group-average map; single-map self-referential null, see note below
├── partial_rsa.py               # partial RSA controlling for individual modalities
├── av_derived_maps.py           # conjunction, superadditivity, and max-unimodal maps
├── max_uni.py                   # per-subject AV-minus-max-unimodal RSA contrast with group inference (t-test + BH-FDR)
├── residualized_maps.py         # consolidate residualized group-average maps
├── channel_class_rsa.py         # AV-channel-class restricted group-average RSA
├── run_channel_class_rsa.sh     # three-model, four-significance-class RSA runner
├── run_extended_analyses.sh     # reproducible residualized AV analysis suite
├── multimodal_decomposition.py  # variance decomposition across RSA modalities
├── rdm_diagonal.py              # off-diagonal control for fMRI autocorrelation
├── precompute_neighbors.py          # pre-build geodesic neighbourhood lookup
├── crossnobis_searchlight.py    # inter-subject crossvalidated Mahalanobis searchlight RSA (N=175 subjects, no repeated-clip design)
├── noise_ceiling.py             # vertex-wise inter-subject noise ceiling (NC_upper / NC_lower) for searchlight RSA
├── tfce_groupstats.py           # group stats with TFCE + sign-flipping permutation FWE correction (Fisher-z applied) alongside BH-FDR
├── kreilability.py              # split-half reliability of searchlight RSA maps across neighbourhood sizes (k)
├── topoomni_sheet_localizer.py            # Ward's-linkage stimulus-cluster localizer on Topo-Omni's cortical sheet (Algorithm 1); speech positive control
├── topoomni_av_separability_localizer.py  # Ward's-linkage localizer testing modality-presence / temporal-binding selectivity via Fisher's-exact cluster enrichment
├── label_av_separability_maps.py          # merge per-cluster + combined AV-separability searchlight runs into one labeled CIFTI
├── label_sheet_localizer_maps.py          # merge per-cluster + combined sheet-localizer searchlight runs into one labeled CIFTI
├── localizer_naming.py          # shared driver/sheet/kind/design name construction for the localizer family
├── localizer_readout_comparison.py        # peak rho, %FDR-significant, top Glasser parcels, and pairwise Dice overlap across localizer readouts
├── integration_convergence.py   # cross-architecture convergence of best-additive integration maps (mean z-score + sign-agreement count)
├── spatial_stats.py             # 2D spatial-compactness metrics for TopoOmni cortical-sheet unit selections (Island Moran's I inputs)
├── draw_rsa_borders.py          # trace Workbench surface borders around high-RSA islands
├── av_scramble_permutation_inference.py   # aggregate AV-scramble reruns into an empirical permutation null
├── scramble_diff_maps.py        # consolidate intact vs. temporally-scrambled plain/partial RSA into one CIFTI ("binding")
├── scramble_paired_stats.py     # per-subject paired t-test (intact − scrambled rho) with FDR / optional sign-flip permutation test
├── temporal_scramble_binding.py # per-vertex + cross-model binding maps (integration_intact − integration_scrambled)
├── dummy_diff_maps.py           # consolidate intact vs. single-real-modality (dummy) plain/partial RSA into one CIFTI (modality-presence diff)
├── shared/
│   ├── __init__.py
│   ├── residuals.py                 # shared linear/projection residual-embedding computations (used by RSA + encoding generation scripts)
│   ├── rsa_utils.py                 # fMRI extraction, embedding processing, RDM utilities
│   └── model_registry.py            # model name / path registry
```

## run_spin_permutations.py — scope note

Produces the spin-test numbers reported in `outputs/results_report.tex`
(`rsa/spin_tests/*_spin1000_summary.json`). Its CLI takes exactly one map
(`--combined-cifti`/`--map-name`); the null is
`null[s, v] = empirical[spin_indices[s, v]]` — each vertex's value under
1000 random spatial rotations of the *same* map. This is a single-map,
spatial-autocorrelation-preserving local hot-spot test, not a two-map
spatial-correspondence test and not a map-vs-zero test. On a diffusely
positive map (no sharp punctate peaks) it will find almost nothing
significant by construction; a near-zero result here does not mean the
underlying effect failed a spatial-autocorrelation control. The script was
briefly absent from this tree (`git log` shows it was renamed to
`groupstats.py` at `5bb63b7` and then deleted at `04ce871`, with only a
stale `__pycache__/run_spin_permutations.cpython-310.pyc` surviving) and has
been restored verbatim from git history.

## group_stats.py vs tfce_groupstats.py

Two separate group-level significance scripts exist and are NOT interchangeable:

- `group_stats.py` — omits the Fisher-z transform per Schutt et al. 2023 §5.1.4 (see its docstring).
- `tfce_groupstats.py` — applies Fisher-z alongside TFCE + sign-flipping permutation FWE correction, plus BH-FDR.

This is a deliberate fork, not an inconsistency — but it silently changes
p-values depending on which script produced a given map. Use `tfce_groupstats.py`
when you need FWE-corrected significance; use `group_stats.py` when you need
the Schutt-convention FDR result. Check which one wrote a given output before
comparing p-values across analyses.

## Usage

```bash
conda activate movie
cd movie_watching   # run from repo root

bash rsa/analysis.sh avg                         # group-average, all methods
bash rsa/analysis.sh avg searchlight             # group-average, searchlight only
bash rsa/analysis.sh avg glasser                 # group-average, Glasser only
bash rsa/analysis.sh persubject                  # per-subject, streaming (default)
bash rsa/analysis.sh persubject all 4            # 4 parallel jobs
bash rsa/analysis.sh persubject all 8 100610     # resume from subject 100610
bash rsa/analysis.sh groupstats                  # re-run group stats on existing maps
bash rsa/analysis.sh all                         # avg + persubject + groupstats
bash rsa/run_channel_class_rsa.sh                # 12 restricted RSA analyses in one CIFTI
```

## Configuration

All parameters are set in `analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `STREAM` | `true` | Streaming mode — preprocess raw CIFTIs on-the-fly (no pre-saved CIFTI needed) |
| `SG_FILTER` | `true` | Savitzky-Golay high-pass filter (removes scanner drift) |
| `PSC` | `true` | Percent signal change normalization |
| `GSR` | `true` | Global signal regression |
| `BIN_SEC` | 2.0 | Temporal bin size (seconds) |
| `DELAY_SEC` | 5.0 | Hemodynamic delay — boxcar shift applied at analysis time |
| `HRF` | `false` | Convolve embeddings with SPM HRF instead of boxcar delay |
| `METHOD` | `spearman` | RDM correlation method |
| `K` | 150 | Searchlight neighbourhood size (vertices) |
| `MODELS` | (array) | Model registry — add new models here |
| `TIMING_CSV` | `data/movie_timing.csv` | Authoritative timing file |

## Streaming vs Disk Mode

**Streaming (default, `STREAM=true`):** Raw 7T CIFTIs are loaded on-the-fly with no signal preprocessing by default (SG/PSC/GSR all off). Movie segment extraction, hemodynamic delay, binning, and z-scoring happen at analysis time using `movie_timing.csv`.

**Disk mode (`STREAM=false`):** Reads pre-saved CIFTIs from `PREPROCESSED_DIR`. Use after running `preprocess_individual.py --save-individual`. The preprocessing suffix (`FMRI_SUFFIX`) must match what `preprocess_individual.py` used.

**Template CIFTI:** `TEMPLATE_CIFTI` points to the `sg_psc` group-average CIFTI. It is used only for its BrainModelAxis (grayordinate structure) — it does not need to match per-subject preprocessing.

## fMRI Preprocessing Convention

- **Continuous full-run CIFTIs** — `preprocess_individual.py` concatenates all 4 runs. Default is raw (no SG/PSC/GSR). No timing filtering is done during preprocessing.
- **Timing applied at analysis time** — `searchlight.py` and `glasser.py` load `movie_timing.csv`, compute within-run onset TRs from global `onset_sec`, apply hemodynamic delay, bin to `BIN_SEC` resolution, z-score per run (training stats applied to test), and split train/test.
- **Z-scoring** — per run on training time bins; same mean/std applied to test bins from the same run (no leakage).

## Model Registry

```bash
MODELS=(
    "pe-av-small-16-frame:av"
    # "pe-av-small-16-frame:v,a,av"
    # "audiomae:a"
    # "videomaev2-large:v"
)
```

Any model whose embedding file is absent is silently skipped. Add a new model by appending a line.

## Outputs

`run_channel_class_rsa.sh` writes one combined CIFTI under
`outputs/cf_modeling/channel_cca_preference/results/rsa`. It covers PE-AV
full CLS-AV plus Nemotron-18 MP and Topo-Omni-18 sheet MP full embeddings.
Each has `audio_only`, `video_only`, `both`, and `neither` analyses; winner
classes are intentionally excluded. Every class includes rho, empirical raw p,
BH-FDR q, max-T FWER p, signed significance maps, and raw/BH/max-T .05 masks.
The shared null consists of 500 synchronized, nonzero within-run circular
shifts. Temporary restart caches are removed after the combined CIFTI and exact
map/parameter inventory are written.

```
{OUTPUT_DIR}/{subject}/{model}_{modality}/{config_label}/
    rsa_59k_{prep}_k{K}_delay{D}s_bin{B}_{method}_searchlight.dscalar.nii   # searchlight
    rsa_59k_{prep}_delay{D}s_bin{B}_{method}_glasser.dscalar.nii            # Glasser
    ranked_report.csv                                                          # Glasser only

{OUTPUT_DIR}/groupstats/{model}_{modality}/{config_label}/
    group_stats_{N}subs.dscalar.nii    # 8-map CIFTI: mean_rho, cohens_d, sigmaps, clusters
    summary.json
```

For supported natural joint-AV models, a group-average `_av` searchlight run
also adds nine descriptive maps to its normal combined
`rsa_59k_*_maps.dscalar.nii`:

- `av_conjunction_{own,unimodal,text_aligned}`: binary masks where
  `(AV > 0) & (AV > A) & (AV > V)`.
- `av_superadditivity_{own,unimodal,text_aligned}`: signed
  `AV - (A + V)` contrasts.
- `av_max_uni_{own,unimodal,text_aligned}`: signed
  `AV - max(A, V)` contrasts.

`own` uses cls-a/cls-v for PE-AV and CAV-MAE, and the matching pre-fusion
encoder A/V RSA maps for Omni3B, Topo-Omni, and Nemotron. `unimodal` uses
AudioMAE+VideoMAEv2; `text_aligned` uses WavLM+PE-Core. Missing group-average
A/V RSA dependencies run automatically with `_av`, provided
their embeddings exist for the selected bin/stride.

The group-average residualized AV suite is reproducible with:

```bash
bash rsa/run_extended_analyses.sh all
```

The runner generates linear- and projection-residual embeddings, runs their
searchlight RSA, runs partial-correlation RSA, and consolidates the results.
Each of the 42 natural AV models receives one
`rsa_59k_*_residualized_maps.dscalar.nii` in its normal `_av` result directory
with `partial_correlation`, `linear_resid`, and `projection_resid` scalars.
The `embed`, `rsa`, `partial`, and `consolidate` stages can also be run
individually. Consolidation resumes per scalar.

CAV-MAE's AV representation is the explicit concatenation of its own A/V
streams, so its projection residual has no representational variance. Its
`projection_resid` scalar is therefore an exact zero map.

`config_label` for searchlight: `k{K}_delay{D}s_bin{B}s_{method}` (e.g. `k150_delay5s_bin2s_spearman`)  
`config_label` for Glasser: `delay{D}s_bin{B}s_{method}`

## Subjects List

`data/subjects.txt` — one subject ID per line. Supports:
- Full-line comments: `# comment`
- Inline comments: `100610  # note`
- Blank lines

The parser strips all comments and blank lines before feeding subject IDs to GNU parallel.

## Resume / Skip

Re-running skips any subject/model whose output CIFTI already exists.  
Delete the output CIFTI to force a rerun for that subject.

## Silence GNU parallel citation notice (one-time)

```bash
parallel --citation
```

## Full-sheet RSA

- `notebooks/feature_extraction/topo_omni_extract_full_sheet.py` — extracts
  the complete 304×512 = 155,648-unit Topo-Omni cortical sheet, intact joint
  audiovisual pass only. `main()` ends with a self-validation gate that
  correlates all 36 decoder layers against the pre-existing
  `topoomni_layer18_sheet_mp_av.npy` reference and asserts layer 18 is the
  unique argmax; on the delivered sheet it passed with layer 18 = 0.7089 vs.
  runner-up layer 19 = 0.3041 (margin 0.4049).
- `rsa/full_sheet_rsa.py` — the searchlight RSA driver. Imports
  `_plot_sheet_map`/`_sanity_corr` from `rsa/cca_seed_sheet_rsa.py` and
  `load_true_coords`/`characterize` from `rsa/cca_seed_sheet_rsa_truecoords.py`.
- `rsa/run_full_sheet_rsa.sh` — the runner.

**Sheet geometry** (304 rows × 512 cols = 155,648 units, row-major
flattened):

- Rows 0–159, cols 0–255: VISION encoder tower, 32 layers × 5 rows (40,960
  units).
- Rows 0–159, cols 256–511: AUDIO encoder tower, 32 layers × 5 rows (40,960
  units).
- Rows 160–303, all 512 cols: DECODER block, the Thinker language-model
  backbone, 36 layers × 4 rows (73,728 units). Layer L occupies rows
  `160+4L .. 163+4L`; the layer-18 reference used by every earlier Topo-Omni
  embedding is rows 232–235.
- `region_layer_id` scheme: vision layer `l` → `l` (0–31); audio layer `l` →
  `100+l` (100–131); decoder layer `l` → `200+l` (200–235). 100 layers total.

**Run parameters:** 155,648 units × 626 bins, k=100 neighbours, n_perm=1000,
seed 42, spearman, bin 5s / skip 5s, delay 5s, FDR alpha 0.05, true
coordinates (`permute_coordinates`, seed 42) cached at
`outputs/rsa/cca_seed_sheet_rsa/topoomni_true_coords_seed42.npy`. Wall time
17m03s.

**Artifacts:** `outputs/rsa/cca_seed_sheet_rsa/fullsheet_k100_truecoords/`:
`cca_a_av.csv`, `cca_a_av_rho.npy`, `cca_a_av_rho_sheet_map.png`,
`cca_p_av.csv`, `cca_p_av_rho.npy`, `cca_p_av_rho_sheet_map.png`,
`diff_rho_sheet_map.png`, `metadata.json`.

**Results** (all from `metadata.json`):

- cca_a (anterior seed): rho range [0.00535, 0.46769], mean 0.1157,
  FDR-significant 154,044/155,648.
- cca_p (posterior seed): rho range [0.00875, 0.23909], mean 0.0953,
  FDR-significant 155,648/155,648.
- Per-block mean rho — vision (n=32 layers): cca_a 0.0395 / cca_p 0.0506;
  audio (n=32): cca_a 0.2732 / cca_p 0.1439; decoder (n=36): cca_a 0.0706 /
  cca_p 0.0931. The anterior>posterior contrast is entirely an audio-block
  effect — in the vision and decoder blocks posterior actually edges
  anterior. Both seeds peak in the audio tower (cca_a highest = audio layer
  29 at 0.3110; cca_p highest = audio layer 28 at 0.1686) and both trough at
  vision layer 0 (cca_a 0.0115, cca_p 0.0176).
- diff (a minus p): mean +0.0205, sd 0.0710, positive in 31.16% of units.
  The audio tower is 26.3% of the sheet, close to that 31% positive
  fraction — that reconciles a positive mean difference with only a
  minority of units being positive.
- diff_max: unit 48936 = audio layer 19, row 95, col 470, delta +0.2465.
  diff_min: unit 127804 = decoder layer 22, row 249, col 135, delta −0.0909.
- Cross-seed spatial Pearson r = 0.8639; top-decile hotspot Jaccard 0.7940
  (overlap 13,778 of union 17,352, n_top_decile 15,565 per seed).

**Caveats — limits, not findings:**

1. Cross-seed spatial r = 0.864 with hotspot Jaccard 0.794 means the two
   seed maps are largely the same map. The A-P contrast is a residual on
   top of a large shared signal, not two independent topographies.
2. The hotspot contiguity test is saturated and should not be cited as
   evidence. `observed_mean_nn_dist` was 1.00232 (cca_a) and 1.00343
   (cca_p) against an integer-grid floor of exactly 1.0, versus a null of
   ~1.7059. p = 0.0005 is exactly 1/2001, the smallest value 2000
   permutations can produce. It also tests spatial clustering on
   coordinates that were themselves generated by a topographic training
   objective, so clustering is expected by construction.
3. The `significance_caveat` field inside the delivered `metadata.json`
   (written 2026-09-03 11:08) is wrong: it reads "Significance was
   meaningfully below ceiling for at least one seed" because of the
   over-strict 0.999 gate in the pre-fix version of `full_sheet_rsa.py`.
   The true minimum is 98.97% (cca_a). The gate is fixed in
   `rsa/full_sheet_rsa.py` as of this commit, but the existing
   `metadata.json` was not retroactively edited — it predates the fix.
