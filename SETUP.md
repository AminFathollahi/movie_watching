# Environment Setup

This repository requires Python 3.10+, a Conda environment, and the **vicsompy** package
(for CF modeling only). Follow these steps exactly.

## 1. Clone vicsompy (CF modeling only)

vicsompy **cannot be pip-installed** with modern PyTorch (≥ 2.6) due to a hard dependency
on an older torch version. All CF modeling scripts import it directly from source.

```bash
cd /path/to/your/workspace
git clone https://github.com/nicholashedger/vicsompy.git
# Note the full path — you'll set VICSOMPY_REPO below
```

If a specific vicsompy commit is needed for exact reproducibility, pin it:
```bash
cd vicsompy
git checkout <commit_sha>   # see NOTICE in cf_modeling/ for the version used
```

## 2. Set up the Conda environment

```bash
# Clone from an existing vicsompy_av environment (recommended — preserves the
# CUDA-enabled torch build that conda/pip cannot recreate correctly):
conda create --clone vicsompy_av --name movie -y

# Or create from the reference spec (manual torch CUDA build required):
conda env create -f cf_modeling/environment.yml
```

The `movie` environment includes:
- `torch 2.11.0+cu128` (CUDA 12.8, RTX 5070Ti compatible)
- `himalaya 0.3.5`
- `pycortex` (for flatmap visualisation)
- `nibabel`, `scipy`, `scikit-learn`, `pandas`, `tqdm`, `joblib`, `statsmodels`
- `mne ≥ 1.9` (Savitzky-Golay filter in preprocessing)
- GNU `parallel` (per-subject batch processing)

## 3. Configure paths

Two environment variables control key paths. Set them in your shell profile
(`~/.bashrc`, `~/.zshrc`, or an `env.sh` file you source before running analyses):

```bash
# Path to the cloned vicsompy repository (CF modeling only)
export VICSOMPY_REPO=/path/to/vicsompy

# Path to the pycortex filestore (flat maps; CF modeling only)
export PYCORTEX_FILESTORE=/path/to/hedger2026
```

All shell scripts in `cf_modeling/`, `encoding/`, and `rsa/` respect these variables.
You can also override them per-run:

```bash
VICSOMPY_REPO=/alt/path bash cf_modeling/run_analysis.sh geometry
```

## 4. Verify the setup

```bash
conda activate movie

# Verify torch + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Verify vicsompy importable (CF modeling only)
python -c "import sys; sys.path.insert(0, '$VICSOMPY_REPO'); from vicsompy.modeling import MssCf; print('vicsompy OK')"

# Verify pycortex (CF modeling visualization only)
python -c "import cortex; print('pycortex OK')"
```

## 5. Data paths

Edit the `CONFIG` section at the top of each `run_analysis.sh` to match your local
data directories before running any analyses. Key variables:

| Variable | Description |
|---|---|
| `DATA_BASE` | Root data directory |
| `OUTPUTS_BASE` | Root outputs directory |
| `CIFTI_DIR` | Raw 7T HCP CIFTI files (streaming mode) |
| `VICSOMPY_REPO` | vicsompy source repo (CF modeling) |
| `PYCORTEX_STORE` | Pycortex filestore (CF modeling visualization) |
| `GLASSER_DLABEL` | HCP-MMP1 59k_fs_LR dlabel parcellation |
| `SUBJECTS_LIST` | Text file listing subject IDs (one per line) |

## 6. Run order

```bash
conda activate movie

# Preprocessing (one-time; skip if using streaming mode)
bash cf_modeling/run_analysis.sh preprocess

# CF modeling (geometry → group-average → per-subject)
bash cf_modeling/run_analysis.sh all

# RSA searchlight
bash rsa/run_analysis.sh avg

# Encoding models
bash encoding/run_analysis.sh avg
```

See `cf_modeling/README.md`, `rsa/README.md`, and `encoding/README.md` for full details.
