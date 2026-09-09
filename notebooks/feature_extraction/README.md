# Feature Extraction — Model Embeddings

Extracts audiovisual, audio-only, and video-only embeddings from transformer models at various layers and conditions. Supports intact, temporally scrambled, and single-modality-only (dummy) variants. Embeddings are saved as per-frame or per-bin arrays for downstream RSA and encoding analyses.

## Extraction Scripts by Model Family

**PE-AV (PerceiverIO Audio-Visual)**
- `pe_av_extract_intact.py` — Intact joint AV embeddings
- `pe_av_extract_scramble.py` — Temporally scrambled AV embeddings
- `pe_av_extract_dummy_modality.py` — Single-modality (A-only, V-only) embeddings for modality-presence control

**Nemotron (Qwen2.5-Omni language model, layer-specific extraction)**
- `nemotron_extract_intact.py` — Intact layer-18 last-token AV embeddings
- `nemotron_extract_scramble.py` — Temporally scrambled layer-18 embeddings
- `nemotron_extract_dummy_modality.py` — Single-modality dummy embeddings
- `nemotron_extract_encoder_penultimate.py` — Penultimate encoder-tower layer (for residualized variants)
- `nemotron_extract_thinker_penultimate.py` — Penultimate language-model thinker layer
- `nemotron_extract_lasttoken_av_layer36_bin5s.py` — Alternate layer extraction (layer 36)
- `nemotron_extract_lasttoken_av_bin5s_backfill.py` — Bin-level extraction with temporal backfill
- `nemotron_extract_cumulative.py` — Cumulative/running embeddings (not standard)

**Omni3B (Qwen2.5-Omni 3B model)**
- `omni3b_extract_intact.py` — Intact layer-18 last-token AV embeddings
- `omni3b_extract_scramble.py` — Temporally scrambled embeddings
- `omni3b_extract_dummy_modality.py` — Single-modality embeddings
- `omni3b_extract_encoder_penultimate.py` — Penultimate encoder layer
- `omni3b_extract_thinker_penultimate.py` — Penultimate thinker layer

**Topo-Omni (Qwen2.5-Omni with cortical-sheet topographic constraint)**
- `topo_omni_extract.py` — Standard layer-18 last-token extraction
- `topo_omni_extract_intact.py` — Intact AV embeddings (canonical)
- `topo_omni_extract_scramble.py` — Temporally scrambled embeddings
- `topo_omni_extract_dummy_modality.py` — Single-modality embeddings
- `topo_omni_extract_encoder_penultimate.py` — Penultimate encoder layer
- `topo_omni_extract_thinker_penultimate.py` — Penultimate thinker layer
- `topo_omni_extract_full_sheet.py` — Complete 304×512 cortical sheet across all 36 decoder layers | `outputs/model_embeddings/topo-omni` |

**CAV-MAE (Contrastive Audio-Visual Masked Autoencoder)**
- `extract_cav_mae_sync.py` — Synchronized audio-visual embeddings

**Support & Utility Scripts**
- `av_pairing.py` | Extract and validate paired audio and video feature sequences at frame and bin levels; used by pairing_control.py and factorial_interaction | `outputs/model_embeddings` |
- `compute_linear_residual_embeddings.py` | Generate linear-residual embeddings (A/V residualized on the other) | `outputs/model_embeddings` |
- `compute_projection_residual_embeddings.py` | Generate projection-residual embeddings (remove best-rank-k projection component) | `outputs/model_embeddings` |
- `build_scramble_unimodal_copies.py` | Create dummy single-modality versions of scrambled embeddings | utility |
- `segment_bin5s_skip1s.py` | Bin embeddings into 5-second segments with 1-second stride | utility |
- `fix_bin.py` | Repair binning errors in existing embeddings | utility |
- `timing.py` | Movie timing utilities and TR-to-second conversions | utility |
- `rewrite_text_inputs.py` | Regenerate text (caption/transcript) embeddings | utility |
- `whisper_transcribe_bin2s.py` | ASR transcription at 2-second bin resolution | `outputs/model_embeddings` |
- `download_model.ipynb` | Download and cache model weights | notebook |

## Analysis Notebooks

- `pe_av_embeddings.ipynb` — PE-AV embedding exploration and validation
- `omni3b.ipynb` — Omni3B embedding properties and layer-wise analysis
- `omni7b.ipynb` — Omni7B embedding properties
- `topo_omni.ipynb` — Topo-Omni topographic-sheet analysis and layer selection
- `cav_mae_sync.ipynb` — CAV-MAE synchronization and embedding alignment
- `audiocaption.ipynb` — Audio caption extraction pipeline
- `text.ipynb` — Text (caption/transcript) embedding analysis
- `stimulus_regressor.ipynb` — Binary stimulus on/off regressor construction
- `segmentation.ipynb` — Segment timing and bin construction
- `single_modality_embedding_extraction.ipynb` — Dummy-modality embedding strategies
