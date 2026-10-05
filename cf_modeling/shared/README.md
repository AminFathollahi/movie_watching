# cf_modeling/shared

`ridge_utils.py` holds standalone NumPy and himalaya versions of the steps that `lib/cf_model.py` delegates to vicsompy's `MssCf`. The pipeline scripts do not import it; `tests/test_cf_math.py` uses it to check the model math on synthetic data.

| Function | Computes |
|---|---|
| `build_pipeline` | Per band, `StandardScaler` followed by a linear kernel (`ColumnKernelizer`), then himalaya `MultipleKernelRidgeCV` (solver `random_search`, regularization grid `logspace(1, 20, 20)`) with leave-one-run-out cross-validation. |
| `build_sphere_to_grayord_lut` | Map from bilateral surface vertex index (2 x 59,292) to grayordinate row, `-1` for the medial wall. |
| `project_onto_lboes` | Design matrix `data[ROI vertices].T @ eigenvectors`, stacked as `[A left, A right, B left, B right]`, with the band sizes. |
| `fit_null_r2` | R^2 of the least-squares fit `Y ~ intercept + b * regressor` for every column of `Y`. |

The module imports `generate_leave_one_run_out` from `vicsompy.utils`, so it needs `VICSOMPY_REPO` (see the top-level README).

Test: `pytest tests/test_cf_math.py -v`
