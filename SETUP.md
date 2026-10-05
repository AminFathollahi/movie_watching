# Setup

Python 3.10 and conda are required. The connective-field code additionally needs the `vicsompy` source repository.

## vicsompy

`vicsompy` is imported from source because it cannot be pip-installed together with PyTorch 2.6 or later (it pins an older torch).

```bash
git clone https://github.com/nicholashedger/vicsompy.git /path/to/vicsompy
export VICSOMPY_REPO=/path/to/vicsompy
```

`cf_modeling/01_extract_geometry.py`, `02_fit_cf_model.py` and `shared/ridge_utils.py` read `VICSOMPY_REPO`; both scripts also accept `--vicsompy-repo`. `cf_modeling/run_analysis.sh` sets its own `VICSOMPY_REPO` in its CONFIG block.

## Environment

```bash
conda env create -f cf_modeling/environment.yml
conda activate movie
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"
```

The `movie` environment covers CF modeling, RSA, encoding and clustering. Package versions are listed in `cf_modeling/environment.yml` (numpy below 2.0 is required by himalaya; GNU `parallel` is used for per-subject batches). The environments for embedding extraction (`avtransformer`, `topo_omni`, `cav-mae-sync`, `audiocaption`) are described in the root `README.md`.

Check the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import himalaya; print(himalaya.__version__)"
python -c "import cortex"
python -c "import sys; sys.path.insert(0, '$VICSOMPY_REPO'); from vicsompy.modeling import MssCf"
```

## Paths

Data and output locations are set in the CONFIG block at the top of `cf_modeling/run_analysis.sh`, `rsa/analysis.sh` and `encoding/analysis.sh`. Edit them before running.

| Variable | Meaning |
|---|---|
| `DATA_BASE` | Root data directory |
| `OUTPUTS_BASE` | Root output directory |
| `CIFTI_DIR` | Raw 7T HCP CIFTI files (used when `STREAM=true`) |
| `SUBJECTS_LIST` | Text file with one subject ID per line |
| `GLASSER_DLABEL` | HCP-MMP1 parcellation, `59k_fs_LR` dlabel (CF modeling and RSA) |
| `VICSOMPY_REPO` | vicsompy checkout (CF modeling) |
| `PYCORTEX_STORE` | Pycortex filestore, exported as `PYCORTEX_FILESTORE` (CF modeling) |

`cf_modeling/run_analysis.sh` also reads `MOVIE_HCP_DIR` and `MOVIE_RAW_CIFTI_DIR` from the environment, if set, for the HCP group-average directory and `CIFTI_DIR`.

## Run order

```bash
conda activate movie
bash cf_modeling/run_analysis.sh preprocess     # one time; not needed with STREAM=true
bash cf_modeling/run_analysis.sh all            # geometry, group average, per subject
bash rsa/analysis.sh avg                        # group-average RSA
bash encoding/analysis.sh variance_partition    # encoding models
```

The default preprocessing is Savitzky-Golay high-pass filtering and percent signal change without global signal regression (suffix `sg_psc`). See `cf_modeling/README.md`, `rsa/README.md` and `encoding/README.md` for the individual analyses.
