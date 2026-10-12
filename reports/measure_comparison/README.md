# Measure comparison: RSA, CKA and encoding

Compares the searchlight maps that representational similarity analysis (RSA), centered kernel alignment (CKA) and encoding produce for the same models on the 108,441 cortical grayordinates of the group average. `compare_measures.py` reads existing maps, runs no searchlight, and writes one markdown report. All tables are descriptive; no inferential statistics are computed, since neighboring grayordinates are spatially dependent.

## Inputs

| Name in the report | Source |
|---|---|
| `pearson_spearman` | `rho_j` of the RSA with correlation-distance (1 - Pearson r) RDMs and Spearman comparison (`rsa/`, `..._pearson_spearman_center_joint_only_models.dscalar.nii`) |
| `euclidean_spearman` | `rho_j` of the Euclidean RSA of `rsa/rdm_searchlight.py` (`..._euclidean_spearman_center_unimodal_own_models.dscalar.nii`) |
| `pearson_spearman_nr` | RSA diagnostic with the repeated clips dropped (`{model}/diagnostics/`) |
| `cka`, `cka_nr` | `cka_j` of the non-cross-validated CKA (`cka/`, tag `--cka-tag`) and its diagnostic with the repeated clips dropped |
| `loro:{map}` | leave-one-run-out encoding maps `r2_a`, `r2_v`, `r2_j`, `r2_av`, `r2_avj` and `unique_j` (`encoding/`, tag `--encoding-tag`) |

The script also reads the group-average responses, the Glasser parcellation and the geodesic neighbor cache of `rsa/`.

## Method

For each model the report gives:

- the mean, 95th percentile and maximum of each RSA and CKA map;
- the Pearson and Spearman correlation across grayordinates of each RSA or CKA map with every other map, including the encoding maps;
- the ten Glasser parcels with the highest parcel mean of each map;
- a ring diagnostic: with the core defined as the top `--core-percent` percent (default 5) of `loro:unique_j` or of `loro:r2_avj`, grayordinates are binned by f, the fraction of their k searchlight neighbors inside the core, and the mean of each map is given per bin;
- neighborhood homogeneity h, the mean pairwise Pearson correlation over windows of the binned responses of a grayordinate's k neighbors, and each map's correlation with h and mean per decile of h;
- the effect of dropping the repeated clips on `pearson_spearman` and `cka`.

For each pair of models it gives the distribution of the difference of their `cka` maps and the ten parcels most favoring each model.

## Usage

```bash
python reports/measure_comparison/compare_measures.py \
    --models pe-av-small-16-frame nemotron_layer18_mp --bin-sec 5 --k 100 \
    --scaling center --encoding-scaling demean \
    --output reports/measure_comparison/measure_comparison.md
```

`--scaling` selects the RSA and CKA embedding normalization (`center` or `zscore`) and `--encoding-scaling` the encoding feature scaling (`demean` or `zscore`). Input directories default to `outputs/{rsa,cka,encoding}` and `data/` of the project; see `--help` for all options.

## Outputs

One markdown file (default `reports/measure_comparison/measure_comparison.md`) with one block of tables per model and one per model pair.

## Tests

`tests/test_measure_comparison.py`.
