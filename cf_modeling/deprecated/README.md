# cf_modeling/deprecated

Archived, unmaintained code for connective-field analyses of the "CCA" ROIs (`cca_a`, `cca_p`), anterior and posterior temporal regions obtained by thresholding group representational-similarity searchlight maps, together with the scripts that read those maps and their run script (`run_analysis.sh`). The current pipeline is in `../`. Output paths inside these scripts may not match the current layout, so pass explicit output directories.

`roi_mean_raw_connectivity.py` and `export_raw_corr_bivariate_cifti.py` (zero-order ROI-mean correlation maps and their four-view exports) were retired when the ROI-mean analysis kept only the leave-one-run-out partial correlation; their tests are in `tests/` beside them.
