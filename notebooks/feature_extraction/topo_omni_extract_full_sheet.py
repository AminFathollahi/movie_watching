"""
notebooks/feature_extraction/topo_omni_extract_full_sheet.py
==============================================================
Full 304x512 (155,648-unit) Topo-Omni cortical sheet extraction -- INTACT
joint audiovisual pass only (no unimodal a/v ablations: would triple GPU
cost for a deliverable that only needs the intact sheet).

Builds on notebooks/feature_extraction/topo_omni_extract_intact.py (READ
FULLY before touching that file -- it is untracked/irreplaceable and is
reused here only by reference, never edited). That script hooks a SUBSET of
decoder CorticalAdaptors (TARGET_LAYERS = [1,9,18,27,34,35]) and saves both
"hidden" (X_hat) and "sheet" (Z) variants. This script hooks EVERY decoder
layer plus both encoder towers' CorticalAdaptors and saves ONLY the
assembled full-sheet Z array.

Sheet geometry
--------------
Verified against Model Repos/topo-omni/src/models/qwen2_5_omni.py's
apply_spatial_loss reshape arithmetic (`visual_cortical_sheet.permute(1,0,2)
.reshape(-1,160,256)`, `audio_cortical_sheet...reshape(-1,160,256)`,
`multimodal_cortical_sheet.permute(1,0,2).reshape(-1,144,512)`) AND against
this checkpoint's own local config.json:
    vision_config: depth=32, hidden_size=1280 -> CorticalAdaptor.cortical_dim
        falls back to hidden_size (config.cortical_dim unset) = 1280
    audio_config:  encoder_layers=32, d_model=1280 -> cortical_dim=1280
        (hidden_size unset on this config, falls back to d_model)
    text_config:   num_hidden_layers=36, hidden_size=2048 -> cortical_dim=2048
1280 = 5*256 and 2048 = 4*512, so:
    rows   0-159 (encoder block, 32 layers x 5 rows each):
        vision layer l, unit d in [0,1280): abs_row = l*5 + d//256,
            col = d%256            (cols 0-255)
        audio  layer l, unit d in [0,1280): abs_row = l*5 + d//256,
            col = 256 + d%256      (cols 256-511)
    rows 160-303 (decoder block, 36 layers x 4 rows each):
        layer l, unit d in [0,2048): flat = l*2048 + d,
            abs_row = 160 + flat//512, col = flat%512
The saved (n_bins, 155648) array is this (304,512) grid, standard row-major
flatten (index = abs_row*512 + col) -- so a layer's 4 (or 5) rows, once
flattened, reproduce the exact same element order as the SAME layer's own
un-reshaped Z vector (reshape+reshape-back is a metadata-only no-op). This
is what the layer-18 validation at the bottom of main() checks against the
already-saved topoomni_layer18_sheet_mp_av.npy.

We get here by hooking each CorticalAdaptor's raw forward output (X_hat, Z)
directly with register_forward_hook -- exactly the principle
topo_omni_extract_intact.py already uses for its 6 decoder layers -- which
bypasses the model's own apply_spatial_loss unified_sheet assembly entirely
(that whole reshape pipeline never runs: model_config.apply_spatial_loss is
set False below, same as the existing script, purely to avoid the unrelated
spatial-loss compute cost -- our hooks do not depend on it).

Pooling
-------
Decoder layers: masked mean-pool over the joint sequence's audio+video token
positions -- bit-identical methodology to topo_omni_extract_intact.py's
sheet_av / the on-disk "_mp" suffix.
Encoder layers: plain mean over all output tokens of that pass. There is no
av_mask to apply here -- the vision tower's forward call only ever sees
video patches and the audio tower's forward call only ever sees audio
frames (both towers are invoked once each, on their own single-modality
input, inside the very same joint forward pass used for the decoder hooks).
Mean-over-all-own-modality-tokens is the natural single-modality analogue of
"_mp": both pool exhaustively over every token that belongs to the unit's
own input, with no cherry-picking.

DATA_BASE note: topo_omni_extract_intact.py's DATA_BASE symlink
(data/segmented_stimulus -> /media/amin/EXTERNAL_USB/...) is currently a
DEAD symlink on this machine (that drive is not attached). The identical
data is also mirrored on the ADATA HD710 PRO external drive, which IS
mounted -- DATA_BASE below tries the symlink first, then that mirror.

Run with:
    conda run --no-capture-output -n topo_omni \
        BIN_SEC=5.0 SKIP_SEC=5.0 \
        python "notebooks/feature_extraction/topo_omni_extract_full_sheet.py"
"""

import gc
import os

# Must run BEFORE any transformers/huggingface_hub import -- see
# topo_omni_extract_intact.py's module docstring for why.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from natsort import natsorted
from tqdm import tqdm

TOPO_OMNI_REPO = Path("/home/amin/Research/Representation/Movie/Model Repos/topo-omni")
assert TOPO_OMNI_REPO.is_dir(), f"topo-omni repo not found at {TOPO_OMNI_REPO}"
sys.path.insert(0, str(TOPO_OMNI_REPO))
sys.path.insert(0, str(TOPO_OMNI_REPO / "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, CorticalAdaptor
from qwen_vl_utils import process_vision_info

TOPO_MODEL_ID = "epfl-neuroai/topo-omni"

# ~/.cache/huggingface/hub/models--epfl-neuroai--topo-omni is ALSO a dead
# symlink to the unattached EXTERNAL_USB drive on this machine (same issue as
# DATA_BASE below) -- snapshot_download(local_files_only=True) fails through
# it even though a perfectly good mirror of the same snapshot sits on the
# ADATA HD710 PRO drive, which IS mounted. Resolve the snapshot dir directly
# instead of trusting HF_HOME resolution.
_HF_SNAPSHOT_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--epfl-neuroai--topo-omni",
    Path("/media/amin/ADATA HD710 PRO/hf_models/hub/models--epfl-neuroai--topo-omni"),
]


def _resolve_model_path() -> str:
    for base in _HF_SNAPSHOT_CANDIDATES:
        snap_dir = base / "snapshots"
        if snap_dir.is_dir():
            for child in sorted(snap_dir.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    return str(child)
    # Fall back to the normal resolution in case HF_HOME already points
    # somewhere with a valid, non-broken cache.
    return snapshot_download(TOPO_MODEL_ID, local_files_only=True)


_DATA_BASE_CANDIDATES = [
    Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered"),
    Path("/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus/filtered"),
]
DATA_BASE = next(
    (p for p in _DATA_BASE_CANDIDATES if p.is_dir() and any(p.iterdir())), None
)
assert DATA_BASE is not None, (
    f"No usable segmented_stimulus/filtered directory found. Tried: "
    f"{[str(p) for p in _DATA_BASE_CANDIDATES]}"
)

EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
BIN_SEC  = float(os.environ.get("BIN_SEC", "2.0"))
SKIP_SEC = float(os.environ.get("SKIP_SEC", str(BIN_SEC)))
MODEL_TAG = "topoomni_fullsheet"
AUDIO_SR = 16000

# Sheet geometry constants -- see module docstring.
SHEET_COLS             = 512
ENCODER_ROWS           = 160
DECODER_ROWS           = 144
N_SHEET_ROWS           = ENCODER_ROWS + DECODER_ROWS   # 304
N_SHEET_UNITS          = N_SHEET_ROWS * SHEET_COLS     # 155648
ENCODER_ROWS_PER_LAYER = 5     # 1280 / 256
DECODER_ROWS_PER_LAYER = 4     # 2048 / 512
ENCODER_COL_WIDTH      = 256
ENCODER_UNIT_DIM       = 1280
DECODER_UNIT_DIM       = 2048
N_VISION_LAYERS_EXPECTED = 32
N_AUDIO_LAYERS_EXPECTED  = 32
N_DECODER_LAYERS_EXPECTED = 36

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def main():
    MODEL_PATH = _resolve_model_path()
    print(f"Resolved local snapshot dir: {MODEL_PATH}")
    print(f"DATA_BASE: {DATA_BASE}")

    torch.cuda.empty_cache()
    gc.collect()

    print("Loading topo-omni Thinker config ...")
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    model_config.audio_config.is_training = False
    model_config.vision_config.is_training = False
    model_config.text_config.is_training = False
    model_config.apply_spatial_loss = False

    print("Loading topo-omni Thinker weights ...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, config=model_config, device_map={"": DEVICE},
        torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    model.lm_head = torch.nn.Identity()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    _MODEL_DEVICE = next(model.parameters()).device
    print(f"Thinker loaded on {_MODEL_DEVICE}.")

    N_DECODER_LAYERS = model.config.text_config.num_hidden_layers
    N_VISION_LAYERS  = model.config.vision_config.depth
    N_AUDIO_LAYERS   = model.config.audio_config.encoder_layers
    assert N_DECODER_LAYERS == N_DECODER_LAYERS_EXPECTED, N_DECODER_LAYERS
    assert N_VISION_LAYERS == N_VISION_LAYERS_EXPECTED, N_VISION_LAYERS
    assert N_AUDIO_LAYERS == N_AUDIO_LAYERS_EXPECTED, N_AUDIO_LAYERS
    print(f"N_DECODER_LAYERS={N_DECODER_LAYERS}  N_VISION_LAYERS={N_VISION_LAYERS}  "
          f"N_AUDIO_LAYERS={N_AUDIO_LAYERS}")

    AUDIO_TOKEN_ID = model.config.audio_token_id
    VIDEO_TOKEN_ID = model.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

    def _find_decoder_adaptors(n_expected):
        direct = getattr(getattr(model, "model", None), "cortical_adaptors", None)
        if direct is not None and len(direct) == n_expected and isinstance(direct[0], CorticalAdaptor):
            return direct
        for _, mod in model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and len(mod) == n_expected
                    and isinstance(mod[0], CorticalAdaptor)):
                return mod
        raise RuntimeError(f"Could not find CorticalAdaptor ModuleList of length {n_expected}")

    decoder_adaptors = _find_decoder_adaptors(N_DECODER_LAYERS)
    vision_adaptors = model.visual.cortical_adaptors
    audio_adaptors = model.audio_tower.cortical_adaptors
    assert len(vision_adaptors) == N_VISION_LAYERS and isinstance(vision_adaptors[0], CorticalAdaptor)
    assert len(audio_adaptors) == N_AUDIO_LAYERS and isinstance(audio_adaptors[0], CorticalAdaptor)
    print(f"Cortical adaptors found: decoder={len(decoder_adaptors)} "
          f"vision={len(vision_adaptors)} audio={len(audio_adaptors)}")
    print(f"ENCODER SHEET ADAPTORS FOUND: model.visual.cortical_adaptors "
          f"(len={len(vision_adaptors)}) and model.audio_tower.cortical_adaptors "
          f"(len={len(audio_adaptors)}) -- same CorticalAdaptor class as the decoder, "
          f"discovered directly as named submodules, no fallback search needed.")

    def _load_audio(path, target_sr):
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return np.zeros(target_sr, dtype=np.float32)
            native_sr = stream.sample_rate
            chunks = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray().astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                chunks.append(arr)
        if not chunks:
            return np.zeros(target_sr, dtype=np.float32)
        audio = np.concatenate(chunks)
        if native_sr != target_sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), native_sr, target_sr).numpy()
        return audio

    def _find_audio_path(video_path):
        # NOTE: the real chunked wavs live under Audio{N}/Audio{N}_chunks_5s/
        # (i.e. "_av_chunks_" -> "_chunks_" too), but topo_omni_extract_intact.py
        # (untracked, irreplaceable, never edited) only does Video->Audio and
        # never renames "_av_chunks_", so this candidate path never exists and
        # every extraction -- including every reference/sheet embedding already
        # on disk -- has always fallen back to decoding audio from the mp4's own
        # AAC track. Fixing the glob here would silently diverge this script's
        # audio from every other embedding in the project; that would require a
        # coordinated re-extraction of ALL of them. Left as-is deliberately for
        # consistency -- the warning below makes the fallback loud instead of
        # silent (the original script doesn't even print it).
        parts = [p.replace("Video", "Audio") for p in video_path.parts]
        wav_path = Path(*parts).with_suffix(".wav")
        if wav_path.exists():
            return wav_path
        _find_audio_path.miss_count = getattr(_find_audio_path, "miss_count", 0) + 1
        print(f"WARNING: audio wav not found, falling back to video audio track: {wav_path} (video={video_path})", file=sys.stderr)
        return video_path

    def _build_joint_inputs(video_path):
        audio_path = _find_audio_path(video_path)
        content = [
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text", "text": "Describe what you see and hear."},
        ]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, frames = process_vision_info(msgs)
        audio_array = _load_audio(audio_path, AUDIO_SR)
        return text, frames, audio_array

    def _install_hooks():
        dec_z, enc_v_z, enc_a_z = {}, {}, {}
        handles = []

        def _dec_hook(idx):
            def _hook(module, inp, out):
                dec_z[idx] = out[1].detach()
            return _hook

        def _enc_hook(store, idx):
            def _hook(module, inp, out):
                z = out[1].detach()
                store[idx] = z.mean(dim=0).float().cpu().numpy()
            return _hook

        for i, ad in enumerate(decoder_adaptors):
            handles.append(ad.register_forward_hook(_dec_hook(i)))
        for i, ad in enumerate(vision_adaptors):
            handles.append(ad.register_forward_hook(_enc_hook(enc_v_z, i)))
        for i, ad in enumerate(audio_adaptors):
            handles.append(ad.register_forward_hook(_enc_hook(enc_a_z, i)))
        return dec_z, enc_v_z, enc_a_z, handles

    def _assemble_sheet(dec_z, enc_v_z, enc_a_z, av_mask):
        sheet = np.zeros((N_SHEET_ROWS, SHEET_COLS), dtype=np.float32)
        for l in range(N_DECODER_LAYERS):
            zs = dec_z[l]  # (1, seq, 2048)
            pooled = (
                zs[0, av_mask, :].mean(dim=0).float().cpu().numpy()
                if av_mask.any() else np.zeros(DECODER_UNIT_DIM, dtype=np.float32)
            )
            row0 = ENCODER_ROWS + l * DECODER_ROWS_PER_LAYER
            sheet[row0:row0 + DECODER_ROWS_PER_LAYER, :] = pooled.reshape(
                DECODER_ROWS_PER_LAYER, SHEET_COLS)
        for l in range(N_VISION_LAYERS):
            row0 = l * ENCODER_ROWS_PER_LAYER
            sheet[row0:row0 + ENCODER_ROWS_PER_LAYER, 0:ENCODER_COL_WIDTH] = (
                enc_v_z[l].reshape(ENCODER_ROWS_PER_LAYER, ENCODER_COL_WIDTH))
        for l in range(N_AUDIO_LAYERS):
            row0 = l * ENCODER_ROWS_PER_LAYER
            sheet[row0:row0 + ENCODER_ROWS_PER_LAYER,
                  ENCODER_COL_WIDTH:2 * ENCODER_COL_WIDTH] = (
                enc_a_z[l].reshape(ENCODER_ROWS_PER_LAYER, ENCODER_COL_WIDTH))
        return sheet.reshape(-1)

    def extract_joint_full_sheet(video_path):
        text, frames, audio_array = _build_joint_inputs(video_path)
        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)
        dec_z, enc_v_z, enc_a_z, handles = _install_hooks()
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()
        ids = inputs["input_ids"].squeeze(0).cpu()
        av_mask = (ids == AUDIO_TOKEN_ID) | (ids == VIDEO_TOKEN_ID)
        del inputs
        torch.cuda.empty_cache()
        flat = _assemble_sheet(dec_z, enc_v_z, enc_a_z, av_mask)
        del dec_z, enc_v_z, enc_a_z
        return flat

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    if not all_segs:
        fallback_suffix = f"_chunks_{dur_int}s" if skip_int == dur_int else f"_chunks_{dur_int}s_skip{skip_int}s"
        all_segs = natsorted(list(DATA_BASE.rglob(f"*{fallback_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("TOPOOMNI_FULLSHEET_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"TOPOOMNI_FULLSHEET_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    results = []
    failed = []
    t_start = time.time()
    for i, vp in enumerate(tqdm(all_segs, desc="Full-sheet joint extraction")):
        try:
            results.append(extract_joint_full_sheet(vp))
        except Exception as e:
            failed.append((vp.name, repr(e)))
            results.append(np.zeros(N_SHEET_UNITS, dtype=np.float32))
        if i == 0:
            print(f"First segment took {time.time() - t_start:.1f}s "
                  f"(includes any lazy CUDA warmup).")

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    full_sheet = np.stack(results, axis=0).astype(np.float32)  # (n_bins, 155648)
    print(f"Full sheet array: {full_sheet.shape}")

    out_dir = EMBEDDINGS_BASE / MODEL_TAG / f"bin{dur_int}s_skip{skip_int}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL_TAG}_av.npy"
    np.save(out_path, full_sheet)
    print(f"Saved {out_path} ({full_sheet.nbytes / 1e6:.1f} MB)")

    # ── Validation: layer-18 slice must be the unique best-correlating decoder
    # layer against the reference (topoomni_layer18_sheet_mp_av.npy) ──
    #
    # NOT an absolute-correlation threshold (e.g. frac_units_corr>0.999) --
    # that reference was extracted from stimulus mp4s on the now-unmounted
    # EXTERNAL_USB drive. topo_omni_extract_intact.py (the very script that
    # made the reference), re-run today unmodified against today's
    # ADATA-mirror mp4s, reproduces layer 18's mean_corr=0.7076 exactly (4
    # decimal places) against the same reference -- so 0.7076 is the ceiling
    # this environment can achieve, not a bug in this script. What the gate
    # can actually test is layout: that layer 18's slice is still uniquely
    # the best match among all 36 decoder layers, i.e. the sheet's rows
    # weren't shuffled/mislaid.
    ref_path = (EMBEDDINGS_BASE / "topoomni_layer18_sheet_mp" /
               f"bin{dur_int}s_skip{skip_int}s" / "topoomni_layer18_sheet_mp_av.npy")
    if not ref_path.exists():
        print(f"VALIDATION SKIPPED: reference file not found at {ref_path}")
    else:
        ref = np.load(ref_path)  # (n_bins, 2048)
        reshaped = full_sheet.reshape(-1, N_SHEET_ROWS, SHEET_COLS)
        layer_mean_corr = np.full(N_DECODER_LAYERS_EXPECTED, np.nan, dtype=np.float64)
        for L in range(N_DECODER_LAYERS_EXPECTED):
            row0 = ENCODER_ROWS + L * DECODER_ROWS_PER_LAYER
            got = reshaped[:, row0:row0 + DECODER_ROWS_PER_LAYER, :].reshape(-1, DECODER_UNIT_DIM)
            if got.shape != ref.shape:
                print(f"VALIDATION FAILED: shape mismatch got={got.shape} ref={ref.shape}")
                sys.exit(1)
            a = got - got.mean(axis=0, keepdims=True)
            b = ref - ref.mean(axis=0, keepdims=True)
            num = (a * b).sum(axis=0)
            den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
            corr = np.divide(num, den, out=np.full(DECODER_UNIT_DIM, np.nan, dtype=np.float64),
                             where=den > 1e-12)
            layer_mean_corr[L] = float(np.nanmean(corr))

        order = np.argsort(layer_mean_corr)[::-1]
        top3 = [(int(L), float(layer_mean_corr[L])) for L in order[:3]]
        print("VALIDATION top-3 decoder layers by mean unit-correlation vs reference:")
        for L, c in top3:
            print(f"  layer {L}: mean_corr={c:.4f}")

        best_layer, best_corr = top3[0]
        runner_up_corr = top3[1][1]
        margin = best_corr - runner_up_corr
        ok = best_layer == 18 and margin >= 0.15
        if ok:
            print(f"VALIDATION PASSED: layer 18 is the unique argmax "
                  f"(margin over runner-up = {margin:.4f} >= 0.15)")
        else:
            print(f"VALIDATION FAILED: argmax layer={best_layer} (expected 18), "
                  f"margin={margin:.4f} -- full sheet is mislaid, refusing to hand off to RSA.")
            sys.exit(1)

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
