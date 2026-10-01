# cf_modeling/lib

Wrappers that let vicsompy's `MssCf` (multi-source spectral connective-field model) run on this project's 59,412-grayordinate cortical CIFTI data for arbitrary ROI pairs, without modifying vicsompy. Imported by `02_fit_cf_model.py` and `03_functional_masks.py`; `vicsompy` must already be on `sys.path` (those scripts add `--vicsompy-repo`).

| Module | Contents |
|--------|----------|
| `cf_model.py` | `CfModel(MssCf)`. `inject_subsurfaces(subsurfaces, roi_names, n_lboes)` loads prebuilt subsurfaces instead of building them; `make_dm_grayord`, `fit_grayord`, `test_xval_grayord` build the design matrix, fit and score with grayordinate targets; `compute_null_r2` fits `y ~ intercept + b * (ROI mean time series)` on the test segment and returns R^2 per grayordinate; `save_all_maps` writes the `cf_model_*.npy` arrays and `band_sizes.npy` (names from `cf_naming.py`; meaning in `../README.md`). `get_params` first moves the fitted himalaya pipeline from GPU to CPU to avoid memory errors in the split prediction. |
| `data_adapter.py` | `grayord_to_sphere_space`, `sphere_to_grayord_space`, `build_sphere_lut`: convert between cortical grayordinates (59,412 rows, medial wall excluded) and the full bilateral sphere layout (118,584 rows = 2 x 59,292; left first, medial wall zero), using the CIFTI `BrainModelAxis`. |
| `config_builder.py` | `build_temp_yaml(roi_a, roi_b, n_lboe_a, n_lboe_b, masks_dir, surfaces_dir, out_dir, ...)` writes the temporary YAML configuration `MssCf` expects (modalities, per-ROI LBOE counts, himalaya solver settings: `random_search`, `n_iter`, batch sizes, regularization grid `logspace(alpha_min, alpha_max, alpha_vals)` with defaults 1, 20, 20) and returns its path. |
| `subject_adapter.py` | `CfSubjectAdapter(subject_id, out_csv)`: the two attributes `MssCf` reads from a subject object (`subject`, `out_csv`; also `out_flat`, `analysis_name`), so no HCP directory layout is needed. |

Import the modules directly, e.g. `from lib.cf_model import CfModel` (the scripts put `cf_modeling/` on `sys.path`).
