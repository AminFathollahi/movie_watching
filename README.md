# Movie-Watching fMRI — Representation Analysis

HCP 7T movie-watching fMRI analyses examining how cortical regions represent audiovisual content. `rsa/`, `encoding/`, and `cf_modeling/` are each self-contained with their own `analysis.sh`; `connectivity/` runs directly via `seed_connectivity.py`.

## Repository Structure

```
movie_watching/
├── preprocess_individual.py    # Per-run signal cleaning → full-run CIFTI
├── make_average.sh             # Build group-average CIFTI from per-subject CIFTIs
├── official_timing.py          # Reconciles HCP official rest-block timing with movie_timing.csv
├── rsa/                        # Representational Similarity Analysis (primary engine)
├── encoding/                   # Ridge encoding, incremental AV, and compression analyses
├── cf_modeling/                # Audiovisual cortical field modeling (Hedger et al. 2025)
├── cluster/                    # Temporal-state/network and vertex-timeseries clustering analyses
├── connectivity/               # Seed-based whole-cortex functional connectivity from an RSA-derived ROI
├── viz/                        # Standalone cortex-map plotting (nilearn, no wb_view dependency)
└── notebooks/
    ├── visualization/          # RSA significance maps, surface plots, pycortex flatmaps
    ├── feature_extraction/     # Model embedding extraction (avtransformer / topo_omni / cav-mae-sync envs)
    └── RidgeEncodingForSpeechEnvMusic_BrainModel/  # early exploratory ridge-encoding work predating the PE-AV pipeline
```

## Analysis Pillars

`rsa/`, `encoding/`, `cf_modeling/`, and `connectivity/` all use the **same `movie` conda environment**.

| Folder | Method | Script | Environment |
|---|---|---|---|
| `rsa/` | Searchlight + Glasser parcel RSA, plus validation (noise ceiling, crossnobis, 2-factor bootstrap, permutation/spin tests) and the multimodal-integration analyses (partial RSA, temporal-scramble binding, cross-architecture convergence, Topo-Omni stimulus-clustering localizer) | `analysis.sh` | `movie` |
| `encoding/` | Ridge encoding plus leakage-safe incremental AV and matched-compression comparisons | `analysis.sh` | `movie` |
| `cf_modeling/` | Banded ridge connective field modeling (Hedger 2025) | `analysis.sh` | `movie` |
| `cluster/` | Temporal-state/network clustering plus vertex-timeseries reduction × clustering maps | `cluster.sh` / `run_vertex_clustering.sh` | `movie` |
| `connectivity/` | Seed-based whole-cortex functional connectivity from an RSA top-5% ROI, 3 time windows (full/rest/stim) | `seed_connectivity.py` | `movie` |

**For detailed script documentation, analysis methods, and output structure, see the README.md in each folder**: [`rsa/README.md`](rsa/README.md), [`encoding/README.md`](encoding/README.md), [`cf_modeling/README.md`](cf_modeling/README.md), [`cluster/README.md`](cluster/README.md), [`connectivity/README.md`](connectivity/README.md).

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
    --raw-dir /media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/individual-59k \
    --out-dir /path/to/data/preprocessed \
    --subjects-list /path/to/data/subjects.txt \
    --sg-filter --psc --gsr \
    --save-individual --save-average
```

Output per subject: `{sub}_sg_psc_cortex_59k.dtseries.nii` + `{sub}_sg_psc_run_trs.npy`
(suffix is `sg_psc` with the default `GSR=false`; becomes `sg_psc_gsr` when `GSR=true`)

> **Streaming mode** (default for RSA/encoding per-subject): Set `STREAM=true` in each
> `analysis.sh`. Raw CIFTIs are preprocessed on-the-fly — no saved CIFTI is needed.

### 3. Extract model embeddings

```bash
# PE-AV embeddings (primary native-AV model)
conda activate avtransformer
jupyter notebook notebooks/feature_extraction/pe_av_embeddings.ipynb

# PE-AV auxiliary extractions (headless scripts, same env):
python notebooks/feature_extraction/pe_av_extract_scramble.py         # temporal-scramble binding control
python notebooks/feature_extraction/pe_av_extract_dummy_modality.py --dummy-modality {a,v}
    # extracts PE-AV's native joint audio_video_embeds ("cls-av") from a single real
    # modality by feeding a fixed synthetic blank/silent placeholder for the other one
    # (the model's forward() only computes true joint embeds when both tensors are real)

# Whisper transcripts + Gemma 4 audio captions + InternVL2.5 video captions
# (three models, sequential — each clears GPU before load and after unload)
conda activate avtransformer
jupyter notebook notebooks/feature_extraction/text.ipynb

# LLM rewrite (fuses captions + audio captions + transcripts via Claude Haiku)
conda activate audiocaption
export ANTHROPIC_API_KEY="sk-ant-..."
jupyter notebook notebooks/feature_extraction/audiocaption.ipynb

# Other native-AV models used for cross-architecture comparison (rsa/analysis.sh Move 1/3/4):
conda activate cav-mae-sync
python notebooks/feature_extraction/extract_cav_mae_sync.py                # + CAV_MAE_SCRAMBLE_AV=1 for the scramble control

conda activate topo_omni
python notebooks/feature_extraction/topo_omni_extract.py                   # Omni3B/TopoOmni thinker hidden states + cortical sheet
python notebooks/feature_extraction/topo_omni_extract_intact.py          # genuine joint av (fixes the (a+v)/2 placeholder)
python notebooks/feature_extraction/topo_omni_extract_scramble.py          # temporal-scramble binding control
python notebooks/feature_extraction/omni3b_extract_intact.py             # same fix, Omni3B side
python notebooks/feature_extraction/omni3b_extract_scramble.py

conda activate avtransformer
python notebooks/feature_extraction/nemotron_extract_intact.py           # Omni-Embed-Nemotron-3B (layers 9/18/27/36)
python notebooks/feature_extraction/nemotron_extract_scramble.py
python notebooks/feature_extraction/build_scramble_unimodal_copies.py      # reindexes real a/v as scramble-run nuisance regressors
```

> **Note:** `audiocaption.ipynb` is **deprecated** for audio captioning (previously used CLAP-Cap).
> It is kept only for the LLM rewrite step (Claude Haiku fusing the three text sources).
> Audio captioning now runs in `text.ipynb` via `google/gemma-4-E4B-it`.

Output: `{EMBEDDINGS_DIR}/{model_name}/bin{B}s_skip{S}s/{model_name}_{v,a,av}.npy`  
Text outputs: `outputs/model_embeddings/text/{seg}s/{captions,audio_captions,transcripts,rewritten_text_inputs}.json`

#### Penultimate thinker/encoder layers (omni-family)

Beyond the depth sweep (`rsa/analysis.sh`'s `LAYERS=(35 34 27 18 9 1)`), each
omni-family model (Omni3B, TopoOmni, Omni-Embed-Nemotron-3B) has two more
targeted extractions:

```bash
conda activate topo_omni   # omni3b / topoomni
python notebooks/feature_extraction/omni3b_extract_thinker_penultimate.py     # thinker layer index 34 (penultimate of 35), mean-pool + last-token
python notebooks/feature_extraction/topo_omni_extract_thinker_penultimate.py  # same, + cortical-sheet variant
python notebooks/feature_extraction/omni3b_extract_encoder_penultimate.py     # audio_tower/visual pre-fusion hidden state (d=1280), a/v only
python notebooks/feature_extraction/topo_omni_extract_encoder_penultimate.py

conda activate avtransformer  # nemotron
python notebooks/feature_extraction/nemotron_extract_thinker_penultimate.py   # thinker layer index 35 (penultimate of 36)
python notebooks/feature_extraction/nemotron_extract_encoder_penultimate.py
```

#### Generalized residualization (`rsa/shared/residuals.py`)

Two independent ways to strip a nuisance signal from a joint AV embedding
before RSA/encoding ever see it, run once per model in
`rsa.shared.model_registry.NATIVE_AV_MODELS` (35 entries):

```bash
conda activate movie
# "_av_linear_resid_unimodal" (nuisance = AudioMAE_a + VideoMAEv2_v, same pair as
#  rsa/partial_rsa.py's partial_corr_* runs) and "_av_linear_resid_encoder"
#  (omni-family only, nuisance = own encoder-penultimate a/v towers):
python notebooks/feature_extraction/compute_linear_residual_embeddings.py \
    --embeddings-dir <embeddings_dir> --timing-csv data/movie_timing.csv \
    --run-trs <run_trs.npy> --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0

# "_av_projection_resid_own" (per-timepoint orthogonal projection vs. the
#  model's own unimodal a/v streams -- no cross-sample regression):
python notebooks/feature_extraction/compute_projection_residual_embeddings.py \
    --embeddings-dir <embeddings_dir> --timing-csv data/movie_timing.csv \
    --run-trs <run_trs.npy> --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0
```

Each residual is saved as a normal `{model}_av_{...}` embedding, so RSA and
encoding consume it unchanged. `rsa/run_extended_analyses.sh` and
`encoding/run_extended_analyses.sh` drive the full group-average sweep over
every generated pseudo-model (`embed`/`plain`/`partial` stages for RSA; see
`results_report.tex` §8.4 "Generalized Partial RSA and Residualization
Across All Native-AV Models" for the write-up).

### 4. Run RSA

```bash
conda activate movie
bash rsa/analysis.sh avg                      # group-average (disk mode)
bash rsa/analysis.sh persubject               # all subjects, streaming (default)
bash rsa/analysis.sh persubject all 4 100610  # 4 parallel jobs, resume from 100610
bash rsa/analysis.sh groupstats               # re-aggregate per-subject maps
```

### 5. Run encoding models

```bash
conda activate movie
bash encoding/analysis.sh avg          # group-average
bash encoding/analysis.sh persubject   # all subjects
```

### 6. Run CF modeling

```bash
conda activate movie
bash cf_modeling/analysis.sh all           # group-average + per-subject
bash cf_modeling/analysis.sh persubject    # per-subject only
```

### 7. Visualize a result map

```bash
conda activate movie
python viz/plot_cortex_map.py \
    --map-file <path.npy | path.dscalar.nii> \
    --template-cifti <path used to build the map, for BrainModelAxis> \
    --out-png <path.png>
```

`viz/` renders any grayordinate-space map directly with nilearn (no
Workbench/wb_view dependency). It also holds one-off Lorax-map conversion/
derivation scripts (`convert_lorax_to_fslr59k.py`, `add_canon_lorax_derived_maps.py`)
and `prepare_wb_view_cortical.py` for setting up a wb_view scene.

## Data Conventions

- **fMRI space**: HCP 7T, 4 runs, TR = 1 s, 59k grayordinate surface (cortex only, 59412 vertices)
- **Preprocessing**: SG high-pass → PSC (using pre-SG mean for normalisation) → GSR, applied per run on the continuous run. No timing filtering at preprocessing time.
- **Timing**: `data/movie_timing.csv` — authoritative timing file for RSA/encoding/CF (18 filtered "meaningful AV content" clips, 4 runs, global `onset_sec`). All analysis scripts apply hemodynamic delay at analysis time. `data/HCP_7T_Movie_Clip_Timing.csv` is the separate official HCP clip-timing source (run-local), used only by `official_timing.py` to derive inter-clip REST-block windows for `connectivity/`'s `rest` time window — reconciling the two is still an open item (see `results_report.tex` §3.2, "Timing reconciliation, still open").
- **Hemodynamic delay**: Applied at analysis time via `--delay-sec 5.0` (boxcar shift) or `--hrf` (SPM HRF convolution).
- **Embeddings**: `{EMBEDDINGS_DIR}/{model_name}/bin{B}s_skip{S}s/{model_name}_{v,a,av}.npy`
- **Outputs**: `{OUTPUT_DIR}/{subject_or_group_average}/{model}/{config_label}/`

## Environments

```bash
# All analyses (RSA, encoding, CF modeling, connectivity):
conda env create -f cf_modeling/environment.yml  # movie env
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install himalaya==0.4.11 pycortex "mne>=1.9"

# PE-AV / nemotron / Qwen-Omni-utils-based embeddings + video captions + transcripts:
conda env create -f notebooks/feature_extraction/environment.yml  # avtransformer env

# Audio captions (CLAP-Cap) + LLM rewrite:
conda env create -f notebooks/feature_extraction/audiocaption_environment.yml  # audiocaption env

# Omni3B / TopoOmni thinker + cortical-sheet extraction (own transformers/torch pins,
# newer than avtransformer's — kept isolated to avoid version conflicts):
conda env create -f notebooks/feature_extraction/topo_omni_environment.yml  # topo_omni env

# CAV-MAE-sync extraction (pins torch 1.13.1 / Python 3.7 — the official repo's
# requirement — MUST stay isolated from every other env in this repo):
conda env create -f notebooks/feature_extraction/cav_mae_sync_environment.yml  # cav-mae-sync env
```

| Environment | Used for |
|---|---|
| `movie` | RSA, encoding, CF modeling, connectivity, preprocessing |
| `avtransformer` | PE-AV embeddings (incl. scramble + dummy-modality cls-av extraction), Nemotron embeddings, InternVL2.5 video captions, Whisper transcripts |
| `audiocaption` | CLAP-Cap audio captions, Claude Haiku LLM rewrite |
| `topo_omni` | Omni3B / TopoOmni thinker hidden states + cortical sheet, unimodal-fix + scramble extraction |
| `cav-mae-sync` | CAV-MAE-sync embeddings + scramble control (requires a separate clone of the official repo — see `notebooks/feature_extraction/cav_mae_sync_environment.yml`) |
