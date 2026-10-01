# Measure comparison: RSA, CKA and encoding

Compares, on the group-average cortical axis (108,441 grayordinates), the searchlight maps of representational similarity analysis (RSA), centered kernel alignment (CKA) and encoding for the same models. The script `compare_measures.py` reads existing maps and writes one markdown report; it runs no searchlight. Scope: the cross-validated CKA, its log-likelihoods and model differences are read from `cka/` outputs. The recomputation of non-cross-validated CKA from the subject-partition mean (plumbing check) and the `cv-energy` map (and the table that conditions on it) are not part of this tool.

## Inputs

All maps must exist before the script runs. `{cfg(m)}` = `k{k}_delay{D}s_bin{B}s_skip{S}s_{m}_{scaling}`.

| Name in the report | File |
|---|---|
| `corr-spearman` | `{rsa-dir}/raw/group_average/{model}_av/rsa_59k_raw_{cfg(spearman)}_maps.dscalar.nii` |
| `euclid-spearman` | `.../{model}_av/diagnostics/rsa_59k_raw_{cfg(euclid-spearman)}_maps.dscalar.nii` |
| `corr-spearman_nr` | `.../{model}_av/diagnostics/rsa_59k_raw_{cfg(corr-spearman)}_norepeats_maps.dscalar.nii` |
| `cka` | `{cka-dir}/raw/group_average/{model}_av/diagnostics/cka_59k_raw_{cfg(noncv)}_maps.dscalar.nii` |
| `cka_nr` | `.../diagnostics/cka_59k_raw_{cfg(noncv)}_norepeats_maps.dscalar.nii` |
| `cka-cv`, `cka-cv-ar`, `cka-cv-loglik`, `cka-cv-ar-loglik` | `.../{model}_av/cka_59k_raw_{cfg(cv)}_maps.dscalar.nii` (and `cv-ar`, `cv-loglik`, `cv-ar-loglik`) |
| model difference | `{cka-dir}/raw/group_average/cka_59k_raw_{cfg(cv-loglik-diff_{A}_av_minus_{B}_av)}_maps.dscalar.nii` (and `cv-ar-loglik-diff_...`), for each pair of `--models` in the order given; skipped if absent |
| encoding | `{encoding-dir}/group_average/{model}/delay{D}s_bin{B}s_skip{S}s/encoding_r2_{split}_{scaling}_{encoding-tag}_{models,partition}.dscalar.nii`, `split` in `fixed`, `runwise` |
| Glasser parcels, time courses, surfaces | `{data-dir}`: `HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210...59k_fs_LR.dlabel.nii`, `preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii` and `..._run_trs.npy`, `movie_timing.csv`, the 59k midthickness surfaces and the geodesic neighbor cache `{geodesic-cache-dir}/group_average_{hem}_neighbors_k{k}.npy` |

How the maps are made: `rsa/README.md` (`corr-spearman`; the two diagnostic variants `euclid-spearman` and `corr-spearman_nr` come from `rsa/searchlight.py --distance euclidean` and `--drop-repeated-clips`) and `cka/README.md` (`cka` is the non-cross-validated CKA; `_nr` = the four repeated clips dropped). Encoding maps are the R-squared maps of `encoding/` (`r2_a`, `r2_v`, `r2_j`, `r2_av`, `r2_avj` from the models file; `unique_j` from the partition file). The encoding columns are `split:map`.

## Run

```bash
python reports/measure_comparison/compare_measures.py \
    --models pe-av-small-16-frame nemotron_layer18_mp \
    --bin-sec 5 --k 100 --scaling center --encoding-tag unimodal_own \
    --rsa-dir outputs/rsa --cka-dir outputs/cka --encoding-dir outputs/encoding \
    --output reports/measure_comparison/measure_comparison.md
```

Defaults: `--skip-sec` = `--bin-sec`, `--delay-sec 5`, `--tr 1`, `--encoding-tag unimodal_own`, `--fmri-suffix raw`, `--subject group_average`, `--core-percent 5`, `--data-dir` and the three output directories under the project root, `--geodesic-cache-dir {rsa-dir}/_geodesic_cache`. CPU only, about one minute for two models.

## Tables (one block per model)

Let K be the number of windows (626 at 5 s) and "map" any RSA, CKA or encoding map; all use the values at the 108,441 grayordinates.

1. **Summary.** Mean, 95th percentile (`numpy.percentile`) and maximum of each RSA/CKA map.
2. **Log-likelihood.** For the `cka-cv` and `cka-cv-ar` values r, Lambda = -(J/2) log(1 - r^2), J = K(K-1)/2 (the maps are read from disk); mean, 95th percentile, maximum.
3. **Spatial correlation.** 3a: Pearson correlation across grayordinates of each RSA/CKA map (row) with every RSA/CKA and encoding map (column). 3b: the same on rank-transformed values (average ranks).
4. **Top parcels.** For each map, the ten Glasser parcels with the highest mean of the map over the parcel's grayordinates, with that mean.
5. **Ring diagnostic**, for three cores: top `--core-percent` percent of `fixed:unique_j`, of `runwise:unique_j` and of `fixed:r2_avj`. For each grayordinate, f = (number of its k searchlight neighbors in the core) / k, medial-wall neighbors counting as not in the core. Grayordinates are binned by f (f = 0, then tenths, upper edge included); the table gives the number of grayordinates and the mean of each RSA/CKA map and of `unique_j` (of the same split as the core, `fixed` for the `r2_avj` core) per bin.
6. **Neighborhood homogeneity** h: for each grayordinate, the mean pairwise Pearson correlation, across the K windows, of the (re-z-scored) binned responses of its neighbors (computed here from the group-average responses, no searchlight). 6a: Pearson and Spearman correlation of each map with h. 6b: mean of h and of each map per decile of the rank of h (decile 1 = lowest).
7. **Repeated-clip effect.** For `corr-spearman` and `cka`: Pearson correlation between the all-windows map and the map with the repeated clips dropped, and the change (dropped minus all) in mean, 95th percentile and maximum.
8. **Model difference** (one block per pair A, B with maps on disk): dl = Lambda_A - Lambda_B for `cv` and `cv-ar`, positive favoring A; mean, 5th and 95th percentile, and counts of grayordinates with dl > 0 and dl < 0.
9. **Model-difference parcels.** Ten parcels with the highest (favoring A) and lowest (favoring B) parcel mean of dl.

No inferential statistics are computed; the tables are descriptive effect-map comparisons (grayordinates are spatially autocorrelated).

## Tests

```bash
conda run -n movie python -m pytest tests/test_measure_comparison.py
```
