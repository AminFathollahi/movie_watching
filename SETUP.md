# Environment Setup

This repository requires Python 3.10+, a Conda environment, and the **vicsompy** package
(for CF modeling only). Follow these steps exactly.

## 1. Clone vicsompy (CF modeling only)

vicsompy **cannot be pip-installed** with modern PyTorch (≥ 2.6) due to a hard dependency
on an older torch version. All CF modeling scripts import it directly from source.

```bash
git clone https://github.com/nicholashedger/vicsompy.git /path/to/Vicarious_somatotopy
# Note the full path — you'll set VICSOMPY_REPO below
```

## 2. Set up the Conda environment

```bash
conda env create -f cf_modeling/environment.yml
conda activate movie

# Install CUDA-enabled torch manually (required for GPU acceleration):
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install remaining pip packages:
pip install himalaya==0.4.11 pycortex "mne>=1.9"
```

The `movie` environment includes:
- `torch 2.11.0+cu128` (CUDA 12.8, RTX 5070Ti compatible)
- `himalaya 0.4.11`
- `pycortex 1.3.0` (for flatmap visualisation)
- `nibabel 5.4.0`, `scipy 1.15.2`, `scikit-learn 1.3.2`, `pandas 2.3.3`
- `numpy 1.26.2` (< 2.0 required by himalaya)
- `mne 1.9.0` (Savitzky-Golay filter in preprocessing)
- `joblib`, `tqdm`, `h5py`, `matplotlib`
- GNU `parallel` (per-subject batch processing)

## 3. Configure paths

Two environment variables control key paths. Set them in your shell profile
(`~/.bashrc`, `~/.zshrc`, or an `env.sh` file you source before running analyses):

```bash
# Path to the cloned vicsompy repository (CF modeling only)
export VICSOMPY_REPO=/path/to/Vicarious_somatotopy

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

# Verify himalaya
python -c "import himalaya; print('himalaya', himalaya.__version__)"

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
# Default preprocessing: SG high-pass + PSC, no GSR (suffix = sg_psc)
bash cf_modeling/run_analysis.sh all

# RSA searchlight
bash rsa/run_analysis.sh avg

# Encoding models
bash encoding/run_analysis.sh avg
```

See `cf_modeling/README.md`, `rsa/README.md`, and `encoding/README.md` for full details.
