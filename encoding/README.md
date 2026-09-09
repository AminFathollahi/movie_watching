# Encoding — Ridge Regression Models

Ridge regression models predict cortical fMRI responses from embeddings of video, audio, and audiovisual transformer models. Alpha is selected per vertex via leave-one-run-out cross-validation on training data (videos 1–14), evaluated on held-out test videos (5, 9, 14, 18).

## Scripts

| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `encoding.py` | Per-subject/group-average ridge regression for video, audio, audiovisual | `outputs/encoding/{subject}/{model}/{config}` |
| `incremental_av.py` | Run-wise comparison of A+V+J (joint) vs. A+V; compression-efficiency variants (PCA, random projection, clustering) | `outputs/encoding/incremental_av/{subject}/{model}/{config}` |
| `pairing_control.py` | Audiovisual pairing-advantage control: compares intact pairings against fold-confined mismatched A/V pairings (3 seeds), with block-level inference over 14 unique movie blocks | `outputs/encoding/pairing_control/group_average/{model}/inference/` |
| `compression_summary.py` | Aggregates compression-efficiency R² and fitted parameters across subjects | `outputs/encoding/incremental_av` |
| `roi_av_profile.py` | Per-ROI audiovisual encoding profiles (unimodal and interactive effects) | `outputs/encoding/roi_av_profile` |
| `av_derived_maps.py` | Group-average audiovisual conjunction, superadditivity, max-unimodal contrast maps | `outputs/encoding/group_average` |
| `diff_maps.py` | Differential encoding maps (A, V, AV contrasts) from group-average results | `outputs/encoding/group_average` |
| `variance_partition.py` | Partitions encoding R² into baseline, extended, and delta (A, V, AV) components | `outputs/encoding/group_average` |
| `group_stats.py` | Aggregates per-subject encoding R² maps to group mean, Cohen's d, significance | `outputs/encoding/group_average` |
| `migrate_encoding_output_names.py` | Rename legacy output files to match current naming convention | utility/migration |

## Runners

`analysis.sh` modes:
- `bash analysis.sh` — Group-average encoding, all models
- `bash analysis.sh persubject [N_JOBS] [RESUME_SUBJECT]` — Per-subject ridge regression (GNU parallel)
- `bash analysis.sh incremental_av` — Run-wise A+V+J vs. A+V and compression analyses
- `bash analysis.sh pairing_control` — Fold-confined mismatch controls (seed configurable via `PAIRING_SEEDS`)
- `bash analysis.sh factorial_interaction` — Crossed-pair AV interaction representation (seed configurable)

`run_diff_study.sh`:
- `bash run_diff_study.sh plain` — Plain encoding for native AV, scrambled AV, and dummy-modality conditions
- `bash run_diff_study.sh incremental` — Incremental variance (A+V+J − A+V) for all conditions
- `bash run_diff_study.sh all` — Both plain and incremental

`run_extended_analyses.sh`:
- `bash run_extended_analyses.sh` — Group-average encoding on residualized embeddings (linear/projection residual variants)

`run_native_av_group_average.sh`:
- `bash run_native_av_group_average.sh` — Canonical 5-second group-average encoding for all native AV models

`run_roi_av_profile.sh`:
- `bash run_roi_av_profile.sh av` — ROI-level audio/video variance decomposition (CCA and auditory/visual anchors)
- `bash run_roi_av_profile.sh text` — ROI-level transcript/caption variance decomposition
- `bash run_roi_av_profile.sh all` — Both av and text

## Configuration

All parameters in `analysis.sh`:
- `BIN_SECS`: temporal binning (default 2.0 seconds)
- `SKIP_SEC`: window stride (default equals BIN_SEC)
- `DELAY_SEC`: hemodynamic delay (default 5.0 seconds)
- `HRF`: if true, convolve embeddings with SPM HRF (default false)
- `BACKEND`: himalaya solver (`torch_cuda` / `torch` / `numpy`)
- `MODELS`: array of model names from shared registry
