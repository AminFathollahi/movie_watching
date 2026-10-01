# Visualization notebooks

Notebooks that read finished RSA outputs and embeddings and write figures. Run them with the `movie` environment from the repository root; paths are set in each notebook's first code cell.

| Notebook | Inputs | Outputs |
|---|---|---|
| `comparison.ipynb` | Group-average and per-subject PE-AV searchlight RSA maps (`outputs/rsa/raw/group_average/`, `outputs/rsa/raw/groupstats/`) for the audio, video and joint embeddings; maps of comparison models (Omni3B layers, CAV-MAE-Sync, WavLM + PE-Core, AudioMAE + VideoMAEv2, ImageBind at 2 s bins); the first ten cortical gradients (`data/margulies2016`); the S1200 myelin map | `outputs/figures/gradient_analysis/`: Spearman correlation of each RSA map with each gradient and with myelin (bar charts), a hexbin of the RSA map against gradient 1, scatter plots of PE-AV against each comparison map. Gradient and myelin maps are resampled from 32k to 59k fsLR with `wb_command` |
| `supp_embedding_plots.ipynb` | PE-AV embeddings (`outputs/model_embeddings/{model}/bin5s_skip5s/`), `data/movie_timing.csv` | `outputs/figures/supp_embedding/`: joint versus single-modality encoding similarity, chunked-segment versus full-run embeddings, audio-video cosine similarity matrices, audio, video and joint RDMs, matched versus unmatched segment-pair statistics |
| `peav_cca_rdm_visualization.ipynb` | Group-average raw dtseries and run lengths, `data/movie_timing.csv`, PE-AV and Nemotron layer-18 joint embeddings, the top-1% CCA-A and CCA-P masks of both models (`outputs/cf_modeling/masks/`) | `rdm_outputs/peav_cca_5s_run_normalized/` |

## peav_cca_rdm_visualization.ipynb

Builds one RDM per source on the same 5 s windows:

- Brain RDMs: for each of the four CCA masks, the multivertex pattern of the group-average fMRI (not the ROI mean) is binned into 5 s windows with a 5 s stride and a 5 s delay, z-scored within run, and compared across windows with correlation distance (1 minus Pearson r).
- Model RDMs: the joint audiovisual embedding binned and z-scored the same way (`rsa.shared.rsa_utils.process_model_embeddings`, no hemodynamic convolution), same distance.

No residualization, smoothing or other transform is applied. All RDM panels share one color scale from 0 to the largest off-diagonal distance.

Output directory `rdm_outputs/peav_cca_5s_run_normalized/`:

| File | Contents |
|---|---|
| `{name}.npy` | RDM, windows by windows, float32 |
| `{name}.png` | RDM image (matched color scale) |
| `rdm_correlations.csv` | Pearson correlation between the upper triangles of every pair of RDMs |
| `metadata.json` | Window definition, normalization, distance, color scale, windows per run, vertex count per mask, source paths |

`{name}` is one of `CCA-A_peav`, `CCA-P_peav`, `CCA-A_nemotron_l18`, `CCA-P_nemotron_l18`, `PEAV_CLS-AV`, `NEMOTRON_L18_AV`.
