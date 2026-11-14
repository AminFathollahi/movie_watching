# Feature extraction

Stimulus segmentation and embedding extraction. Every model is run on the same fixed windows of the 18 movie clips and writes one row per window, so the arrays line up row for row with the fMRI windows used by `rsa/`, `encoding/`, `cluster/` and `subcortical/`.

## Output layout

```
{embeddings-dir}/{model}/bin{B}s_skip{B}s/{model}_{a|v|av}.npy
```

`{embeddings-dir}` is `outputs/model_embeddings`. `B` is the window width in seconds and the stride equals `B` unless the directory name says otherwise (`bin5s_skip1s` is a 5 s window every 1 s). Rows are ordered by clip (1 to 18), then window start time. `a`, `v` and `av` are audio, video and joint audiovisual embeddings of the same windows.

Models run in their native precision: float32 for PE-AV, CAV-MAE-Sync and the unimodal encoders (TF32 disabled), bfloat16 for the Qwen2.5-Omni family (Omni3B, Topo-Omni, Nemotron). Pooled vectors are saved as float32.

Environments are listed in the root [`README.md`](../../README.md): `avtransformer` (PE-AV, Nemotron, Omni3B, `extract_embeddings.py`, Whisper, InternVL, Gemma), `topo_omni`, `cav-mae-sync`, `audiocaption` (CLAP-Cap and the caption rewrite) and `movie` (derived embeddings). Run commands from the repository root, for example `conda run --no-capture-output -n avtransformer python notebooks/feature_extraction/<script>`.

## Stimulus segmentation

| Script | Writes |
|---|---|
| `segment_official.py` | Cuts the 18 clips from the full run movies (`data/segmented_stimulus/full/`) at the official clip times and splits them into 1, 2, 5 and 10 s windows (stride = width; the last partial window is dropped; 3175, 1583, 626 and 310 windows). Writes `Video{N}/Video{N}_chunks_{B}s/` (muted video), `Audio{N}/Audio{N}_chunks_{B}s/` (wav) and, for 1, 2 and 5 s only, `Video{N}/Video{N}_av_chunks_{B}s/` (muxed) under `data/segmented_stimulus/filtered/`, and `data/movie_timing.csv`. Needs `data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy` for the run lengths. |
| `segment_bin5s_skip1s.py` | The same three layouts at 5 s width and 1 s stride (`..._chunks_5s_skip1s`, zero-padded part numbers), cut from `filtered/Video{N}/Video{N}.mp4` and `Audio{N}/Audio{N}.m4a`; each chunk directory gets a `timing.csv`. |
| `segmentation.ipynb` | Notebook version of the per-clip cutting into 5 s and 10 s chunks. |
| `timing.py` | Writes `data/movie_timing.csv` from the spreadsheet `data/TOI_NEW.xlsx` by adding run offsets. The analyses use the timing file written by `segment_official.py`. |
| `stimulus_regressor.ipynb` | Binary stimulus on/off regressor from the official timing, saved as CIFTI. |

## Unimodal encoders

```bash
python notebooks/feature_extraction/extract_embeddings.py --models dasheng-1.2b vjepa2-vitl --bins 5 2 1 [--overwrite] [--limit N]
python notebooks/feature_extraction/extract_embeddings.py --verify [--models ...] [--bins ...]
```

| Flag | Default |
|---|---|
| `--models` | all models in the registry |
| `--bins` | `1 2 5` (10 is available once `segment_official.py` has cut the 10 s windows) |
| `--stimulus-dir` | `external/data/segmented_stimulus/filtered` (locations as in the top-level README) |
| `--embeddings-dir` | `outputs/model_embeddings` |
| `--models-home` | `hf_models/`, set as `HF_HOME`; models load from local files |
| `--device` | `cuda` |
| `--limit` | all chunks; with `N`, only the first `N` chunks of each bin |
| `--all-layers` | write every hidden state as `{model}-l{k:02d}` (`k` = 0 for the embedding-layer output, then 1 to `n`) instead of `-d50`, `-d75` and the final output; models without hidden states are skipped, and models whose layers are captured by hooks (Dasheng, OpenBEATs, VideoMAE V2, V-JEPA 2) have no layer 0 |
| `--overwrite` | recompute; otherwise a model and bin whose arrays exist are skipped |
| `--verify` | check saved arrays (row count against the stimulus chunks, non-finite values, all-zero rows, mean cosine of adjacent rows raw and mean-centered); exit status 1 if any problem |

Registry (audio models read `Audio{N}_chunks_{B}s`, video models `Video{N}_chunks_{B}s`):

| Modality | Models |
|---|---|
| audio (`_a`) | `wavlm-large`, `whisper-large-v3` (encoder), `nemotron-audio-enc`, `dasheng-0.6b`, `dasheng-1.2b`, `openbeats-large-i2`, `w2v-bert-2.0`, `clap-larger`, `languagebind-audio` |
| video (`_v`) | `videomaev2-large`, `videomaev2-giant`, `pe-core-l14`, `vjepa2-vitl`, `vjepa2-vitg`, `languagebind-video`, `viclip-l14`, `qwen2.5-vl-3b-vit`, `nemotron-vision-enc` |

Audio is mixed to mono, resampled to the model's rate, truncated to the window length, and never zero-padded into the pooled vector (Whisper pools only the encoder frames that cover real audio). Video models share 16 frames sampled uniformly from each chunk and apply their own resizing and normalization.

Depth variants: for every model except `clap-larger`, `pe-core-l14`, `languagebind-video`, `languagebind-audio` and `viclip-l14` (which output a projected embedding), three arrays are written. With `n` encoder layers, the output of layer `round(fraction x n)` (1-indexed) is mean-pooled over tokens for `fraction` 0.5 and 0.75 and saved as `{model}-d50` and `{model}-d75`; the plain name `{model}` is the final output, mean-pooled over tokens. Each array has a `{model}_layers.json` beside it with `model` and `dtype` and, for models with depth variants, `layer`, `n_layers` and `pooling` (the plain name records `layer = n_layers`). Example: `dasheng-1.2b-d75/bin5s_skip5s/dasheng-1.2b-d75_a.npy`.

Five models differ from the rest:

- `languagebind-video` (`LanguageBind/LanguageBind_Video_FT`, fully fine-tuned ViT-L/14, float32) is the video tower of LanguageBind and takes frames only (only the video tensor is passed to the model; no audio tower is loaded). It expects 8 frames per clip, sampled uniformly from each window with the same index rule as the other video models; the frames are scaled to [0, 1], normalized with the CLIP mean and standard deviation, resized bilinearly so the short side is 224 and center-cropped to 224 (the library's random horizontal flip is not applied). The saved vector is the 768-dimensional visual projection of the pooled output, L2-normalized (before the learned temperature scaling). It needs the `languagebind` package (version 0.1.0, the pip packaging of the official code with compatibility shims; the video tower's model file differs from the official repository only in the import of `_expand_mask` and `clip_loss`) and `transformers` below version 5, so run it in a separate environment (`languagebind`: a copy of `avtransformer` with `transformers` 4.57, `einops`, `peft`, `pytorchvideo`, `opencv-python-headless` and `languagebind` installed, and a one-line `torchvision/transforms/functional_tensor.py` re-exporting `torchvision.transforms._functional_tensor` for `pytorchvideo`). The checkpoint (`config.json` and `pytorch_model.bin` of the Hugging Face repository) is in `manual/LanguageBind_Video_FT` under the models directory.
- `viclip-l14` (ViCLIP-L/14 trained on InternVid-10M-FLT, Hugging Face conversion `OpenGVLab/ViCLIP-L-14-hf`, float32) is a video-text contrastive model. It uses its own 8 frames per window, sampled uniformly with the same index rule as the other video models, resized to 224 x 224 (no crop) and normalized with the ImageNet mean and standard deviation, as in the model's demo. The saved vector is the 768-dimensional video embedding of the vision encoder, L2-normalized. It loads its remote code from `manual/ViCLIP-L-14-hf` under the models directory and runs in the `languagebind` environment with `setuptools` below version 81 (the model code imports `pkg_resources`).
- `languagebind-audio` (`LanguageBind/LanguageBind_Audio_FT`, fully fine-tuned, float32) is the audio tower of LanguageBind and uses the package's own audio preprocessing: the waveform is mixed to mono, resampled to 16 kHz, mean-subtracted, converted to a Kaldi filter bank (112 mel bins, 25 ms window, 10 ms shift, which is the number of bins in this checkpoint's configuration), cut or padded to 1036 frames (about 10.4 s) by repeating the clip from its start, scaled with the model's mean and standard deviation and stacked as three identical channels. Every window of 1 to 10 s is shorter than 1036 frames, so it is repeated to fill the input: a 1 s window is tiled about 10 times, a 2 s window about 5 times, a 5 s window about twice and a 10 s window (about 998 frames) is padded by its first 38 frames. The embedding layers of the tower are called through a small re-implementation of the patch, class and position embedding sum, because `transformers` 4.57 rejects the non-square 112 x 1036 input. The saved vector is the 768-dimensional audio projection, L2-normalized. It uses the same environment and package as `languagebind-video`; the checkpoint is in `manual/LanguageBind_Audio_FT`.
- `qwen2.5-vl-3b-vit` (the vision encoder of `Qwen/Qwen2.5-VL-3B-Instruct`, bfloat16; only the `visual.*` weights are loaded) is run on 16 frames sampled uniformly from each window, resized to at most 336 x 336 pixels, with the frame rate passed to the model as 16 frames divided by the window width, without a prompt or the language model. The hidden states are the output of the patch embedding (`-l00`), of each of the 32 vision blocks (`-l01` to `-l32`) and of the patch merger (`-merger`, 2,048 channels; the blocks have 1,280), each mean-pooled over the visual tokens. `-merger` is written only with `--all-layers`; the plain name is the merger output and `-d50`, `-d75` are blocks 16 and 24.
- `nemotron-audio-enc` and `nemotron-vision-enc` (the `audio_tower` and `visual` modules of `nvidia/omni-embed-nemotron-3b`, bfloat16; only their weights are loaded, not the language model) are the encoders of Nemotron's Qwen2.5-Omni lineage, 32 layers of 1,280 channels each. The vision encoder gets exactly the input of `qwen2.5-vl-3b-vit` (16 frames per window, at most 336 x 336 pixels, frame rate 16 / window width, Nemotron's own video processor, which has the same patch size, merge size and normalization). The audio encoder gets the window's 16 kHz mono waveform through Nemotron's own feature extractor (128 mel bins, padded to 2,048,000 samples as in the intact extraction and cut back to the real frames by the attention mask), with no text and no video. Hidden states: the input embedding (`-l00`: the patch embedding, or for audio the two convolutions plus the sinusoidal position embedding) and the output of each of the 32 blocks (`-l01` to `-l32`), mean-pooled over tokens; no merger or projection output is written.

## Joint audiovisual models

All joint models read the video and audio chunks of each window from `data/segmented_stimulus/filtered/`. Intact extraction produces `a`, `v` and `av`. Window width and stride are set by the environment variables `BIN_SEC` and `SKIP_SEC` (defaults per script below).

### PE-AV (`facebook/pe-av-small-16-frame`, float32)

`pe_av_extract_intact.py` writes `pe-av-small-16-frame_{v,a,av}.npy`: `v` is `video_embeds`, `a` is `audio_embeds` and `av` is `audio_video_embeds` (the joint fusion head), all from one forward pass per window. `BIN_SEC` defaults to 5.0 and `SKIP_SEC` to 1.0, so set `SKIP_SEC=5` for the non-overlapping grid; `STIMULUS_DIR`, `EMBEDDINGS_BASE` and `BATCH_SIZE` (default 8) are also read.

```bash
BIN_SEC=5 SKIP_SEC=5 python notebooks/feature_extraction/pe_av_extract_intact.py [--unimodal {own,dummy}]
```

`--unimodal own` (default) is the extraction above. The model returns `audio_video_embeds` only when both inputs are real tensors, so with `--unimodal dummy` the single-modality versions of the joint embedding use a placeholder for the missing modality: `a` is `audio_video_embeds` of (real audio, black video) and `v` of (silent audio, real video). Output: `pe-av-small-16-frame_dummy_av/.../pe-av-small-16-frame_dummy_av_{a,v}.npy`. `pe_av_embeddings.ipynb` is the notebook version that runs the PE-AV variants in `MODEL_IDS` on the `skip = bin` layout.

### Nemotron (`nvidia/omni-embed-nemotron-3b`, bfloat16)

A contrastively trained embedding model built on the Qwen2.5-Omni-3B Thinker with bidirectional attention, prompted with the document prefix `passage: `. Layers 9, 18, 27 and 36 (the final layer; `nemotron_extract_intact.py` also writes 35). Two readouts per layer, both L2-normalized: `mp` (mean over all non-padding tokens) and `lt` (last token). `a`, `v` and `av` come from three separate forward passes (audio only, video only, both), so the unimodal embeddings contain no tokens of the other modality. Names: `nemotron_layer{N}_mp`, `nemotron_layer{N}_lt`. `nemotron_extract_intact.py` reads `BIN_SEC` (default 2.0) and `SKIP_SEC` (default `BIN_SEC`).

`nemotron_extract_intact.py --pooling mpcat` writes a joint readout that gives audio and video equal weight regardless of their token counts (`mp` weights each by its share of the tokens), for layers 1, 9, 18, 27 and 36. Layer `L` is `hidden_states[L]` of the forward pass, the output of the `L`-th decoder layer (layer 36 after the final normalization), the same convention as the `mp` files. Audio and video token positions are the positions whose input id is the processor's `<|AUDIO|>` and `<|VIDEO|>` placeholder. With `ā` the mean hidden state over the audio positions and `v̄` the mean over the video positions of the joint pass of one window, `J = [ā / s_a | v̄ / s_v]` (4096 dimensions), where `s_a` and `s_v` are the root mean squared Euclidean norms of `ā` and `v̄` over all windows of the bin, so each half has unit mean squared norm. `J` is not L2-normalized. Output `nemotron_layer{L}_mpcat/bin{B}s_skip{B}s/`: `nemotron_layer{L}_mpcat_av.npy` (`J`), `_a.npy` and `_v.npy` (copies of the `nemotron_layer{L}_mp` audio-only and video-only vectors), `_tokens.npy` (audio and video token counts per window) and `_layers.json` (scales and token count summary). The same run computes the audio-only and video-only `mp` vectors of the five layers and writes a `nemotron_layer{L}_mp` `_a`/`_v` file only if it does not exist; for an existing file it prints the largest absolute difference and leaves it unchanged.

### Omni3B (Qwen2.5-Omni-3B Thinker, bfloat16)

`omni3b_extract_intact.py` (`BIN_SEC` default 5.0, `SKIP_SEC` default `BIN_SEC`) hooks decoder layers 1, 9, 18, 27, 34 and 35 (0-indexed, of 36). `a` and `v` are the mean over that modality's token positions from separate single-modality passes; `av` is the mean over the audio and video token positions of one joint pass; `lt` is the last position of that joint pass. Layers 9, 18, 27 and 34 carry the readout in the name (`omni3b_layer{N}_mp`, and `omni3b_layer{N}_lt` for `av` only); layers 1 and 35 keep the bare name `omni3b_layer{N}`.

### Topo-Omni (`epfl-neuroai/topo-omni`, bfloat16)

Qwen2.5-Omni-3B Thinker with a cortical adaptor after every layer that projects activations onto a 304 x 512 sheet. The scripts import the model class from a local clone of the topo-omni repository (path set at the top of each script). Each extraction writes the hidden state (`topoomni_layer{N}_mp`, `_lt`) and the sheet projection (`topoomni_layer{N}_sheet_mp`, `_sheet_lt`) from the same pass.

- `topo_omni_extract_intact.py` (`BIN_SEC` default 2.0, `SKIP_SEC` default `BIN_SEC`): layers 1, 9, 18, 27, 34, 35, pooled as for Omni3B.
- `topo_omni_extract.py`: 5 s windows over layers 1, 2, 4, 9, 18, 27, 35, with layer-selection diagnostics in `topoomni_diagnostics/`; `av` is `(a + v) / 2` of the two modality pools from one joint pass. `topo_omni.ipynb` is its notebook form.
- `topo_omni_extract_full_sheet.py` (`BIN_SEC` default 2.0): intact joint pass only. It hooks every decoder layer and both encoder towers and saves the full 304 x 512 sheet (155,648 units per window) as `topoomni_fullsheet/bin{B}s_skip{S}s/topoomni_fullsheet_av.npy`, shape (windows, 155648); decoder layers are pooled over audio and video token positions and encoder layers over all tokens of their own modality. It checks that the layer-18 slice best matches `topoomni_layer18_sheet_mp_av.npy`. Example: `BIN_SEC=5.0 SKIP_SEC=5.0 python notebooks/feature_extraction/topo_omni_extract_full_sheet.py`.

### CAV-MAE-Sync (float32)

`extract_cav_mae_sync.py` uses a local clone of the CAV-MAE-Sync repository and its `pretrained_models/cav_mae_sync.pth` (path set at the top of the script). It writes `cav-mae-sync_{a,v,av}.npy` for 5 s and 10 s windows (or the single width in `BIN_SEC`); `a` and `v` are class-token embeddings (video averaged over 16 frames) and `av` is their concatenation. `cav-mae-sync.ipynb` is the notebook form.

## Controls

Controls reuse the model's own extraction and change only the audio-video pairing. All use 5 s windows with stride 5 s.

| Control | Scripts | Output names |
|---|---|---|
| Temporal scramble: the audio of window `i` is replaced by the audio of window `perm(i)` for a seeded random permutation of all windows (seed 42) | `pe_av_extract_scramble.py --scramble-av`; `nemotron_extract_scramble.py`; `omni3b_extract_scramble.py`; `topo_omni_extract_scramble.py`; `CAV_MAE_SCRAMBLE_AV=1 extract_cav_mae_sync.py` | `{model}_avscramble` (`pe-av-small-16-frame`, `cav-mae-sync`); `nemotron_layer{N}_{mp\|lt}_avscramble`; `omni3b_layer{N}_{mp\|lt}_avscramble`; `topoomni_layer{N}_[sheet_]{mp\|lt}_avscramble`, `av` only |
| Dummy modality: one real modality with a black video or a silent audio (`data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4`, `dummy_silence_5s.wav`) in the other | `pe_av_extract_dummy_modality.py`, `nemotron_extract_dummy_modality.py`, `omni3b_extract_dummy_modality.py`, `topo_omni_extract_dummy_modality.py`, each with `--dummy-modality {a,v}` (required), `--force`, `--limit N` | `{model}_clsav_from_a` (dummy video, real audio) and `{model}_clsav_from_v` (dummy audio, real video); layer and readout naming as above |
| Fold-confined mismatch: audio taken from another clip within the same split (train, held-out run, excluded repeated clips) | `pe_av_extract_scramble.py --held-out-run {1-4} --seed S`; `nemotron_extract_scramble.py --held-out-run` | `pe-av-small-16-frame_avmismatch_run{R}_seed{S}`; `nemotron_layer{N}_{mp\|lt}_avmismatch_run{R}_seed{S}` (`av` and `pairing_manifest.csv`) |
| Factorial interaction: for window `i` with training reference `j`, `AV_ii + AV_jj - AV(video i, audio j) - AV(video j, audio i)`, with references from training windows only | `pe_av_extract_scramble.py --factorial-run {1-4} --seed S`; `nemotron_extract_scramble.py --factorial-run` | `pe-av-small-16-frame_interaction_run{R}_seed{S}`; `nemotron_layer{N}_{mp\|lt}_interaction_run{R}_seed{S}` (`av` and `pairing_manifest.csv`) |

`--seed` defaults to 42. For PE-AV a non-default seed is added to the scramble name (`pe-av-small-16-frame_avscramble_seed{S}`); the Nemotron scramble name has no seed. `--force` re-extracts. `build_scramble_unimodal_copies.py` (environment `movie`, no GPU) builds the `a` and `v` files of scrambled models from the intact separate-pass embeddings by indexing with the same permutation (`a_scrambled[i] = a[perm[i]]`, `v` unchanged), for `omni3b`, `topoomni` and `nemotron` layers 9, 18, 27 (and 36 for Nemotron), and writes them beside the scrambled `av`. `av_pairing.py` holds the pairing and contrast functions.

For Omni3B and Topo-Omni the scramble and dummy-modality scripts pool `av` as `(mean over audio tokens + mean over video tokens) / 2`, whereas `*_extract_intact.py` pools over the union of audio and video tokens.

## Penultimate layers and other readouts

| Script | Output |
|---|---|
| `{omni3b,topo_omni}_extract_thinker_penultimate.py` | Thinker layer 34 (`{omni3b,topoomni}_layer34_{mp,lt}`; Topo-Omni also the sheet). Pooling as in the intact script, except that Omni3B's `av` here is `(a + v) / 2` of the separate-pass vectors. |
| `nemotron_extract_thinker_penultimate.py` | Thinker layer 35 (`nemotron_layer35_{mp,lt}`). |
| `{omni3b,topo_omni}_extract_encoder_penultimate.py` | Audio-tower layer 30 and vision-tower block 30 (of 32), mean over tokens, 1280 dimensions, one joint pass: `{family}_encoder_penultimate/bin5s_skip5s/{family}_encoder_penultimate_{a,v}.npy`. The Nemotron version is in `deprecated/`; the Nemotron encoders are extracted layer by layer by `extract_embeddings.py` (`nemotron-audio-enc`, `nemotron-vision-enc`). |
| `omni3b_extract_talker.py` | Talker-stage hidden states of Omni3B (24 layers, 896 dimensions): thinker generation followed by talker generation without waveform synthesis; layers 0, 12 and 23 pooled over the generated codec positions, `omni3b_talker_layer{N}_{mp,lt}` (`av` only). Environment variables `THINKER_MAX_NEW_TOKENS` (128), `TALKER_MAX_NEW_TOKENS` (256), `OMNI3B_TALKER_LIMIT`. |
| `nemotron_extract_cumulative.py` | Nemotron on the video from time 0 through the end of each 5 s window (cumulative context): `nemotron_layer{N}_{mp,lt}_cumulative`. |
| `nemotron_extract_intact_bin5s_skip1s.py` | Fixed-configuration variant of `nemotron_extract_intact.py` (5 s windows with 1 s stride). Running `nemotron_extract_intact.py` with `BIN_SEC=5` produces the last-token layers 9, 18, 27, 35 and 36 at 5 s windows. |

## Derived embeddings (environment `movie`)

Both scripts take `--embeddings-dir`, `--timing-csv`, `--run-trs` (required), `--bin-sec` (5.0), `--skip-sec` (default `--bin-sec`), `--delay-sec` (5.0), `--tr` (1.0), `--models` (default `RESIDUALIZED_AV_MODELS` in `shared/model_registry.py`) and `--force`. Windows are first aligned to the fMRI bins (delay, per-run z-scoring), so use the same `--delay-sec` as the analyses that read the outputs. The result is saved as an `av` embedding.

```bash
python notebooks/feature_extraction/compute_linear_residual_embeddings.py \
    --embeddings-dir outputs/model_embeddings --timing-csv data/movie_timing.csv \
    --run-trs data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy \
    --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0
```

- `compute_linear_residual_embeddings.py`: residual `R = J - J_hat` of the joint embedding `J` after cross-validated ridge regression on nuisance embeddings; prints the multimodality score `||R||^2 / ||J||^2` and the chosen penalty. `--nuisance model:modality ...` (default `audiomae:a videomaev2-large:v`) and `--variant-suffix` (default `_unimodal`) set the nuisance and the name: `{model}_av_linear_resid_unimodal/.../{model}_av_linear_resid_unimodal_av.npy`.
- `compute_projection_residual_embeddings.py`: per window, the part of `av` orthogonal to span{`a_t`, `v_t`} (Gram-Schmidt on the model's own audio and video vectors, no regression across windows): `{model}_av_projection_resid_own`. For Omni3B, Topo-Omni and Nemotron it also writes `{target}_av_linear_resid_encoder` (ridge residual on the family's `encoder_penultimate` a and v, which live in a different space, so no projection is defined); `--skip-encoder` omits these.

## Text inputs

| Notebook or script | Environment | Output (`outputs/model_embeddings/text/{seg}s/`) |
|---|---|---|
| `text.ipynb` | `avtransformer` | `transcripts.json` (Whisper large-v3 with voice-activity filtering), `captions.json` (InternVL2.5-8B video captions), `audio_captions.json` (Gemma 4 E4B audio captions); segment widths 5 and 10 s |
| `audiocaption.ipynb` | `audiocaption` | CLAP-Cap `audio_captions.json` (same file name as in `text.ipynb`); `rewritten_text_inputs.json`, one fused description per segment from captions, audio captions and transcripts with Claude Haiku (`ANTHROPIC_API_KEY` required). Run `text.ipynb` first. |
| `rewrite_text_inputs.py` | any with `anthropic`, `pandas` | `text/bin5s_skip5s/event_descriptions.json` from the hand-labelled concept captions and `transcripts.json` |
| `whisper_transcribe_bin2s.py` | `avtransformer` | `text/bin2s_skip2s/transcripts.json` and per-window word counts `whisper_speech_proxy_a.npy` under `outputs/model_embeddings/whisper_speech_proxy/bin2s_skip2s/` |

## Other files

| File | Purpose |
|---|---|
| `omni3b.ipynb`, `omni7b.ipynb` | Exploratory: class names, token ids and layer-selection RDMs for Qwen2.5-Omni-3B and Qwen3-Omni (7B) |
| `download_model.ipynb` | Download and cache model weights |
| `convert_checkpoints_to_safetensors.py` | Convert `.pt` and `.bin` checkpoints (PE-Core, CLAP, OpenBEATs) to `model.safetensors` in the snapshot folder (paths set in the script) |
| `av_pairing.py` | Pairing and factorial-contrast functions used by the scramble scripts |
