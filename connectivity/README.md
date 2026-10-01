# Connectivity: seed-based functional connectivity

`seed_connectivity.py` computes whole-cortex functional connectivity from a seed region defined by Workbench border files (for example the top-RSA islands written by `rsa/draw_rsa_borders.py`).

## What is computed

1. **Seed mask.** The left and right `.border` files are converted to surface region-of-interest masks with `wb_command -border-to-rois ... -include-border` on the 59k midthickness surfaces (one mask per border, combined with a logical OR), then restricted to the cortical grayordinates of the template CIFTI's brain-model axis (59,412 grayordinates; left hemisphere first). The run stops if the mask is empty.
2. **Seed time series.** With data `X (n_grayordinates, T)` and seed set `S`, the seed time series is the mean over seed grayordinates: `s(t) = (1 / |S|) * sum_{g in S} X[g, t]`.
3. **Connectivity map.** For every grayordinate `g`, the Pearson correlation over the `T` selected time points:
   `r(g) = sum_t (X[g, t] - mean_t X[g]) (s(t) - mean_t s) / ( sqrt(sum_t (X[g, t] - mean_t X[g])^2) * sqrt(sum_t (s(t) - mean_t s)^2) )`, set to 0 where the denominator is 0. Seed grayordinates are included in `s(t)` and in the map, so `r` is near 1 inside the seed; all summary statistics below exclude them.
4. **Per-subject mode.** The map is computed for every subject, transformed with `z = arctanh(clip(r, -0.999999, 0.999999))`, averaged over the `n` successfully processed subjects, and transformed back with `tanh(mean z)`.

**Time window (`--window`)** selects which time points enter `X` and `s`. Time is in seconds with TR = 1 s, global (cumulative across runs), so seconds index the concatenated series directly:

| Window | Time points |
|---|---|
| `full` (default) | All time points |
| `rest` | Official inter-clip rest blocks (block label `0` in `data/HCP_7T_Movie_Clip_Timing.csv`, converted from run-local to global time by `official_timing.py`), dropping the first `--rest-trim-sec` seconds of each block |
| `stim` | The clip windows in `data/movie_timing.csv` (`onset_sec` to `end_sec`), the same windows used by RSA and encoding |

Time points between a clip's official end and its `movie_timing.csv` end belong to neither `rest` nor `stim`.

## Usage

Run from the repository root in the `movie` conda environment.

```bash
python connectivity/seed_connectivity.py \
    --border-lh outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman_center/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_center_top5pct_lh.border \
    --border-rh outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman_center/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_center_top5pct_rh.border \
    --mode group_average --window full \
    --roi-name pe-av-small-16-frame_av_top5pct \
    --out-dir outputs/connectivity/pe-av-small-16-frame_av_top5pct/group_average
```

The border files above come from:

```bash
python rsa/draw_rsa_borders.py --rsa-npy <.../rsa_59k_..._searchlight.npy> \
    --threshold-mode percentile --top-pct 5 --min-verts 10
```

which keeps the top 5 percent of cortical vertices by RSA correlation (jointly over both hemispheres), removes connected islands smaller than 10 vertices, and writes `{stem}_top5pct_{lh,rh}.border` next to the input array.

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--border-lh`, `--border-rh` | required | Seed border files |
| `--roi-name` | required | Seed name used in output file names |
| `--mode` | required | `group_average` or `per_subject` |
| `--window` | `full` | `full`, `rest`, or `stim` |
| `--out-dir` | required | Output directory |
| `--timing-csv` | `data/movie_timing.csv` | Clip windows for `--window stim` |
| `--official-timing-csv` | `data/HCP_7T_Movie_Clip_Timing.csv` | Rest-block table for `--window rest` |
| `--run-trs-npy` | `data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy` | TRs per run, used for the rest-block time conversion in `group_average` mode |
| `--rest-trim-sec` | `6.0` | Seconds dropped from the start of each rest block (hemodynamic carry-over from the preceding clip) |
| `--group-average-cifti` | `data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii` | `group_average` input; also the template for the seed mask |
| `--raw-dir` | `data/individual-59k` | `per_subject` input: raw 7T CIFTIs, 4 runs per subject |
| `--subjects-list` | `data/subjects.txt` | One subject ID per line (`#` comments allowed) |
| `--excluded-list` | `data/excluded.txt` | Subject IDs removed from the list, if the file exists |
| `--template-cifti-per-subject` | the group-average CIFTI above | Supplies the brain-model axis for the seed mask and the output header in `per_subject` mode |
| `--left-surface`, `--right-surface` | `data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.{L,R}.midthickness_MSMAll.59k_fs_LR.surf.gii` | Surfaces for `-border-to-rois` |
| `--workbench` | `/opt/workbench/bin_linux64/wb_command` | Connectome Workbench executable |

`per_subject` mode loads each subject's four raw runs through `preprocess_individual.preprocess_subject` with Savitzky-Golay filtering, percent signal change and global signal regression all off, and concatenates the runs. It processes every subject in the list and therefore takes a long time; subjects that fail to load are skipped with a warning. The Glasser summary uses `data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii` (59k mesh); it is skipped with a warning if loading fails.

## Outputs

With `tag = {roi-name}_{window}`, in `--out-dir`:

| File | Contents |
|---|---|
| `connectivity_{tag}.npy` | `(59412,)` float32 `r` (group average, or the Fisher-averaged map in per-subject mode) |
| `connectivity_{tag}.dscalar.nii` | The same map as a CIFTI scalar, map name `connectivity_{tag}` |
| `connectivity_{tag}_ranked_parcels.csv` | Glasser parcels ranked by mean `r` over the parcel's non-seed vertices; columns `parcel, mean_r, n_vertices, seed_overlap_frac, rank` |
| `connectivity_{tag}_roi_membership_parcels.csv` | Glasser parcels that contain seed vertices, ranked by seed vertex count; columns `parcel, n_roi_vertices, pct_of_parcel, pct_of_roi, rank` |
| `connectivity_{tag}_summary.txt` | Seed name, window, number of subjects averaged, seed size, peak `r` including and excluding the seed, mean of the positive `r` outside the seed, percentage of non-seed grayordinates with `r > 0` |
| `per_subject_maps/{subject}_connectivity_{roi-name}_{window}.npy` | Per-subject `r` maps (`per_subject` mode only) |

No group-level significance test is computed by this script.

## Tests

`seed_connectivity.py` has no tests. The related ROI-mean correlation code in `cf_modeling/` is covered by `tests/test_roi_mean_raw_connectivity.py` and `tests/test_roi_mean_partial_connectivity.py`:

```bash
pytest tests/test_roi_mean_raw_connectivity.py tests/test_roi_mean_partial_connectivity.py
```
