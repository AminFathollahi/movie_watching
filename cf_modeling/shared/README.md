# cf_modeling/shared

`ridge_utils.py`: standalone NumPy/himalaya implementations of the steps that `lib/cf_model.py` delegates to vicsompy's `MssCf`. They are not imported by the pipeline scripts; `tests/test_cf_math.py` uses them to check the math on synthetic data.

| Function | What it computes |
|----------|------------------|
| `build_pipeline(n_samples_train, run_onsets, band_sizes, roi_names, ...)` | scikit-learn pipeline `ColumnKernelizer` (per band: `StandardScaler` then linear kernel) followed by himalaya `MultipleKernelRidgeCV` (solver `random_search`; defaults `n_iter=20`, regularization grid `logspace(1, 20, 20)`, `n_targets_batch=20000`) with leave-one-run-out cross-validation. Returns `(pipeline, backend)`. |
| `build_sphere_to_grayord_lut(bm_axis, n_verts_per_hem=59292)` | Integer array of length 2 x 59,292 mapping a bilateral sphere vertex index to its grayordinate row, `-1` for medial wall. |
| `project_onto_lboes(data, subsurfaces, lut=None)` | Design matrix: for each ROI and hemisphere, `data[ROI vertices].T @ eigenvectors.real`, stacked as `[ROI A left, ROI A right, ROI B left, ROI B right]`; also returns the band sizes `2 * n_lboe`. |
| `fit_null_r2(regressor, Y)` | R^2 of the least-squares fit `Y ~ intercept + b * regressor` for every column of `Y`; 0 where the target has zero variance. |

`generate_leave_one_run_out` is re-exported from `vicsompy.utils` (needs `VICSOMPY_REPO` or the default path to the vicsompy checkout).

Test: `pytest tests/test_cf_math.py -v` from the repository root.
