# cf_modeling/deprecated

Archived, not maintained. Connective-field analyses built on the "CCA" ROIs (`cca_a`, `cca_p`: anterior and posterior temporal regions obtained by thresholding group representational-similarity searchlight maps of an earlier stimulus segmentation), plus code that reads those searchlight maps. The current pipeline (Glasser ROI pairs fit on the full movie) is in `../`.

Imports resolve as `cf_modeling.deprecated.*`, but output paths inside the scripts and `run_analysis.sh` still name the pre-move locations; pass explicit output directories (or prefix them with `deprecated/`) to reproduce.

| File | Content |
|------|---------|
| `run_cca_islands.py`, `roi_definitions.json` | Threshold a cortical searchlight map (fixed value, global top 1-2%, adaptive hemisphere saddle, peak fractions) and keep the two largest temporal components per hemisphere as `cca_a*`/`cca_p*` masks; JSON holds the variants |
| `persubject_cca_channel_1pct.py`, `persubject_cca_maps.py` | Per-subject top-1% CCA masks and seed connectivity maps |
| `channel_cca_analysis.py`, `run_channel_cca_analysis.sh` | Per-channel modality axis versus CCA axis |
| `overlap.py` | Group-average CF integration map versus searchlight maps (Spearman correlation with bootstrap interval, overlap maps); the per-subject statistics part is `../overlap.py` |
| `run_analysis.sh` | The `cca*` and `persubject_cca_peav_1pct` modes of the CF runner (`bash run_analysis.sh MODE`) |
| `compare_lboe_sensitivity.py`, `LBOE_200_validation.md` | LBOE-count comparison for one CCA pair |
| `migrate_cca_artifacts.py` | Rename `mca_*` artifacts to `cca_*` |
| `status.md` | Log of the CCA fits |

Tests that still import from here: `tests/test_cca_island_variants.py`, `tests/test_channel_cca_analysis.py`, `tests/test_migrate_cca_artifacts.py`.
