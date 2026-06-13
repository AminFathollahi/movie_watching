# Movie-Watching fMRI — Representation Analysis

HCP 7T movie-watching fMRI analyses examining how cortical regions represent audiovisual content. Three analysis pillars, each self-contained with its own `run_analysis.sh`.

## Repository Structure

```
movie_watching/
├── preprocess_individual.py    # Per-run signal cleaning → full-run CIFTI
├── make_average.sh             # Build group-average CIFTI from per-subject CIFTIs
├── rsa/                        # Representational Similarity Analysis
├── encoding/                   # Ridge encoding models (himalaya)
├── cf_modeling/                # Audiovisual cortical field modeling (Hedger et al. 2025)
└── notebooks/
    ├── visualization/          # RSA significance maps, surface plots, pycortex flatmaps
    └── feature_extraction/     # Model embedding extraction (avtransformer env)
```

## Analysis Pillars

All three pillars use the **same `movie` conda environment**.

| Folder | Method | Script | Environment |
|---|---|---|---|
| `rsa/` | Searchlight + Glasser parcel RSA | `run_analysis.sh` | `movie` |
| `encoding/` | Ridge encoding (himalaya RidgeCV) | `run_analysis.sh` | `movie` |
| `cf_modeling/` | Banded ridge connective field modeling (Hedger 2025) | `run_analysis.sh` | `movie` |

## Quick Start

### 1. Set up the environment

```bash
conda env create -f cf_modeling/environment.yml
conda activate movie
# Install CUDA torch manually (needed for GPU acceleration):
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"
```

See `SETUP.md` for full setup instructions including path configuration.

### 2. Preprocess fMRI (optional — streaming mode skips this step)

`preprocess_individual.py` applies per-run signal cleaning (SG high-pass → PSC → GSR)
and concatenates all 4 runs into a single full-run CIFTI.

```bash
conda activate movie

python preprocess_individual.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --out-dir /path/to/data/preprocessed \
    --subjects-list /path/to/data/subjects.txt \
    --sg-filter --psc --gsr \
    --save-individual --save-average
```

Output per subject: `{sub}_sg_psc_cortex_59k.dtseries.nii` + `{sub}_sg_psc_run_trs.npy`
(suffix is `sg_psc` with the default `GSR=false`; becomes `sg_psc_gsr` when `GSR=true`)

> **Streaming mode** (default for RSA/encoding per-subject): Set `STREAM=true` in each
> `run_analysis.sh`. Raw CIFTIs are preprocessed on-the-fly — no saved CIFTI is needed.

### 3. Extract model embeddings

```bash
# PE-AV embeddings
conda activate avtransformer
jupyter notebook notebooks/feature_extraction/pe_av_embeddings.ipynb

# Whisper transcripts + Gemma 4 audio captions + InternVL2.5 video captions
# (three models, sequential — each clears GPU before load and after unload)
conda activate avtransformer
jupyter notebook notebooks/feature_extraction/text.ipynb

# LLM rewrite (fuses captions + audio captions + transcripts via Claude Haiku)
conda activate audiocaption
export ANTHROPIC_API_KEY="sk-ant-..."
jupyter notebook notebooks/feature_extraction/audiocaption.ipynb
```

> **Note:** `audiocaption.ipynb` is **deprecated** for audio captioning (previously used CLAP-Cap).
> It is kept only for the LLM rewrite step (Claude Haiku fusing the three text sources).
> Audio captioning now runs in `text.ipynb` via `google/gemma-4-E4B-it`.

Output: `{EMBEDDINGS_DIR}/{model_name}/bin{B}s_skip{S}s/{model_name}_{v,a,av}.npy`  
Text outputs: `outputs/model_embeddings/text/{seg}s/{captions,audio_captions,transcripts,rewritten_text_inputs}.json`

### 4. Run RSA

```bash
conda activate movie
bash rsa/run_analysis.sh avg                      # group-average (disk mode)
bash rsa/run_analysis.sh persubject               # all subjects, streaming (default)
bash rsa/run_analysis.sh persubject all 4 100610  # 4 parallel jobs, resume from 100610
bash rsa/run_analysis.sh groupstats               # re-aggregate per-subject maps
```

### 5. Run encoding models

```bash
conda activate movie
bash encoding/run_analysis.sh avg          # group-average
bash encoding/run_analysis.sh persubject   # all subjects
```

### 6. Run CF modeling

```bash
conda activate movie
bash cf_modeling/run_analysis.sh all           # group-average + per-subject
bash cf_modeling/run_analysis.sh persubject    # per-subject only
```

## Data Conventions

- **fMRI space**: HCP 7T, 4 runs, TR = 1 s, 59k grayordinate surface (cortex only, 59412 vertices)
- **Preprocessing**: SG high-pass → PSC (using pre-SG mean for normalisation) → GSR, applied per run on the continuous run. No timing filtering at preprocessing time.
- **Timing**: `data/movie_timing.csv` — authoritative timing file (18 videos, 4 runs, global `onset_sec`). All analysis scripts apply hemodynamic delay at analysis time.
- **Hemodynamic delay**: Applied at analysis time via `--delay-sec 5.0` (boxcar shift) or `--hrf` (SPM HRF convolution).
- **Embeddings**: `{EMBEDDINGS_DIR}/{model_name}/bin{B}s_skip{S}s/{model_name}_{v,a,av}.npy`
- **Outputs**: `{OUTPUT_DIR}/{subject_or_group_average}/{model}/{config_label}/`

## Environments

```bash
# All analyses (RSA, encoding, CF modeling):
conda env create -f cf_modeling/environment.yml  # movie env
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"

# PE-AV embeddings + video captions + transcripts:
conda env create -f notebooks/feature_extraction/environment.yml  # avtransformer env

# Audio captions (CLAP-Cap) + LLM rewrite:
conda env create -f notebooks/feature_extraction/audiocaption_environment.yml  # audiocaption env
```

| Environment | Used for |
|---|---|
| `movie` | RSA, encoding, CF modeling, preprocessing |
| `avtransformer` | PE-AV embeddings, InternVL2.5 video captions, Whisper transcripts |
| `audiocaption` | CLAP-Cap audio captions, Claude Haiku LLM rewrite |
