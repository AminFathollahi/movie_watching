# cf_modeling/lib

Adapters that run vicsompy's `MssCf` (multi-source connective-field model) on the 108,441-grayordinate cortical CIFTI data for arbitrary ROI pairs, without modifying vicsompy. They are imported by `02_fit_cf_model.py` and `03_functional_masks.py`, which put `cf_modeling/` and the vicsompy checkout (`--vicsompy-repo`) on `sys.path`.

| Module | Content |
|---|---|
| `cf_model.py` | `CfModel(MssCf)`: loads prebuilt subsurfaces, builds the design matrix and fits and scores the model with grayordinate targets, computes the ROI-mean null R^2 (`y ~ intercept + b * ROI mean time series` on the test segment), and writes the `cf_model_*.npy` arrays and `band_sizes.npy` (names from `cf_naming.py`, meaning in `../README.md`). |
| `data_adapter.py` | Conversion between cortical grayordinates (108,441 rows, medial wall excluded) and the bilateral surface layout (2 x 59,292 rows, left hemisphere first, medial wall zero), using the CIFTI `BrainModelAxis`. |
| `config_builder.py` | `build_temp_yaml`: writes the temporary YAML configuration `MssCf` reads (ROI names, LBOE counts, himalaya settings; regularization grid `logspace(alpha_min, alpha_max, alpha_vals)`, defaults 1, 20, 20). |
| `subject_adapter.py` | `CfSubjectAdapter`: the minimal subject object `MssCf` needs, so no HCP directory layout is required. |
