# Connectivity — Seed-Based Functional Connectivity

Computes seed-based functional connectivity maps using vertices/regions extracted from searchlight RSA hotspots or ROI borders. Seeds are extracted from high-RSA regions and their correlation with the rest of cortex is computed per subject and at group average.

## Scripts

| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `seed_connectivity.py` | Seed-based functional connectivity: loads seed border/vertices from RSA output, computes Pearson correlation with whole-brain fMRI per subject/group-average | `outputs/connectivity/{seed_name}/group_average` or `per_subject` |

## Usage

```bash
python seed_connectivity.py \
    --border-lh {path}/rsa_top5pct_lh.border \
    --border-rh {path}/rsa_top5pct_rh.border \
    --mode group_average \
    --roi-name pe-av-small-16-frame_av_top5pct \
    --out-dir outputs/connectivity/pe-av-small-16-frame_av_top5pct/group_average
```

Seed sources:
- RSA searchlight border files (exported via `draw_rsa_borders.py`)
- ROI mask CIFTIs from CF modeling or Glasser atlas
- Manually defined vertex sets

Output includes:
- Per-subject and group-average connectivity correlation maps (.dscalar.nii)
- Group statistics (mean, std, t-test, BH-FDR)
- Connectivity strength summaries by Glasser parcel
