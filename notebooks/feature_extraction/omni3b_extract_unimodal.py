"""
notebooks/feature_extraction/omni3b_extract_unimodal.py
==========================================================
Fixes a methodological bug found while implementing Move 1 (best-additive AV
integration contrast) of the AV-integration extension: omni3b.ipynb's
existing "_a"/"_v" embeddings are mean-pooled from AUDIO-/VIDEO-token
positions within a SINGLE JOINT (audio+video) forward pass, and "_av" is
defined as the elementwise average `(_a + _v) / 2`. That means "_av" is a
deterministic function of "_a" and "_v" in raw feature space (verified
empirically: av == (a+v)/2 bit-for-bit), and "_a"/"_v" are themselves not
pure unimodal baselines (audio-token hidden states are still contextualized
by attention over the video tokens present in the same forward pass, and
vice versa). Both properties undermine a "does av carry unique fusion
information beyond [a,v]" contrast.

This script re-extracts "_a" and "_v" from GENUINELY SEPARATE forward passes
(audio-only input with no video tokens in the sequence at all; video-only
input with no audio tokens), for layers {9, 18, 27} only (the layers used by
Move 1/4/5 of the AV-integration extension). "_av" itself is left UNCHANGED
on disk (still the joint-pass (a+v)/2 value) -- only "_a"/"_v" are replaced,
so the integration residual (av regressed on [a_new, v_new]) is no longer
tautological.

ADDITIONALLY: saves a genuinely emergent joint-AV alternative,
"{tag}_layer{N}_lasttoken_av.npy" -- the hidden state at the LAST sequence
position of the SAME joint (audio+video) forward pass used for the original
"_av", rather than a masked-position average. The last position has
attended, via self-attention, over the entire preceding multimodal context,
so (unlike (a+v)/2) it is not a deterministic function of the "_a"/"_v"
masked-position pools -- closer to how the Topo-Omni paper itself reads out
a per-clip summary (last-layer activation of the final token) for its own
stimulus embeddings (Sec 4.7.2). This is a THIRD condition, not a
replacement -- both "_av" (masked-pool average) and "_av_lasttoken" are kept
for comparison.

Run with:
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/omni3b_extract_unimodal.py"
"""

import gc
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_vl_utils import process_vision_info
from natsort import natsorted

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH      = "Qwen/Qwen2.5-Omni-3B"
DATA_BASE       = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
TARGET_LAYERS   = [9, 18, 27]
MODEL_TAG       = "omni3b"
AUDIO_SR        = 16000

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading Qwen2.5-Omni-3B Thinker ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, device_map={"": DEVICE}, torch_dtype=DTYPE,
        attn_implementation="sdpa", local_files_only=True,
    )
    model.eval()
    _MODEL_DEVICE = next(model.parameters()).device
    print(f"Thinker loaded on {_MODEL_DEVICE}.")

    N_LAYERS = model.config.text_config.num_hidden_layers
    print(f"N_LAYERS={N_LAYERS}  TARGET_LAYERS={TARGET_LAYERS}")

    AUDIO_TOKEN_ID = model.config.audio_token_id
    VIDEO_TOKEN_ID = model.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

    def _find_decoder_layers(n_expected):
        for _, mod in model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and len(mod) == n_expected
                    and hasattr(mod[0], 'self_attn') and hasattr(mod[0], 'mlp')):
                return mod
        raise RuntimeError(f"Could not find decoder ModuleList of length {n_expected}")

    decoder_layers = _find_decoder_layers(N_LAYERS)

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
        parts = [p.replace("Video", "Audio") for p in video_path.parts]
        wav_path = Path(*parts).with_suffix(".wav")
        return wav_path if wav_path.exists() else video_path

    def _build_joint_inputs(video_path):
        """Joint audio+video message (same as the original omni3b.ipynb extraction)."""
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

    def extract_lasttoken(video_path, target_layers=TARGET_LAYERS):
        """Joint AV forward pass; return the LAST-TOKEN hidden state per layer
        (a genuinely emergent joint summary, not a fixed function of a/v pools)."""
        text, frames, audio_array = _build_joint_inputs(video_path)
        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)
        captured = {}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                captured[idx] = hs.detach().cpu()
            return _hook

        for i in target_layers:
            handles.append(decoder_layers[i].register_forward_hook(_make_hook(i)))
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()
        del inputs
        torch.cuda.empty_cache()

        return {idx: captured[idx][0, -1, :].float().numpy() for idx in target_layers}

    def _build_unimodal_inputs(video_path, modality):
        """modality: 'audio' or 'video'. Builds a message containing ONLY that
        modality -- no cross-modal tokens at all in the sequence."""
        if modality == "video":
            content = [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": "Describe what you see."},
            ]
        else:
            content = [
                {"type": "audio", "audio": str(_find_audio_path(video_path))},
                {"type": "text", "text": "Describe what you hear."},
            ]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if modality == "video":
            _, frames = process_vision_info(msgs)
            audio_array = None
        else:
            frames = None
            audio_array = _load_audio(_find_audio_path(video_path), AUDIO_SR)
        return text, frames, audio_array

    def extract_unimodal(video_path, modality, target_layers=TARGET_LAYERS):
        text, frames, audio_array = _build_unimodal_inputs(video_path, modality)
        kwargs = dict(text=[text], return_tensors="pt")
        if modality == "video":
            kwargs["videos"] = frames
        else:
            kwargs["audio"] = audio_array
            kwargs["sampling_rate"] = AUDIO_SR
        inputs = processor(**kwargs).to(_MODEL_DEVICE)

        captured = {}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                captured[idx] = hs.detach().cpu()
            return _hook

        for i in target_layers:
            handles.append(decoder_layers[i].register_forward_hook(_make_hook(i)))
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()

        ids = inputs["input_ids"].squeeze(0).cpu()
        mask = (ids == AUDIO_TOKEN_ID) if modality == "audio" else (ids == VIDEO_TOKEN_ID)
        del inputs
        torch.cuda.empty_cache()

        pooled = {}
        for idx in target_layers:
            hs = captured[idx]  # (1, seq_len, D)
            if mask.any():
                pooled[idx] = hs[0, mask, :].mean(dim=0).float().numpy()
            else:
                pooled[idx] = np.zeros(hs.shape[-1], dtype=np.float32)
        return pooled

    # ── Segment list (same convention as omni3b.ipynb) ───────────────────────
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("OMNI3B_UNIMODAL_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"OMNI3B_UNIMODAL_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    results_a  = {idx: [] for idx in TARGET_LAYERS}
    results_v  = {idx: [] for idx in TARGET_LAYERS}
    results_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp in tqdm(all_segs, desc="Unimodal + lasttoken extraction"):
        try:
            pooled_a = extract_unimodal(vp, "audio")
            pooled_v = extract_unimodal(vp, "video")
            pooled_lt = extract_lasttoken(vp)
            for idx in TARGET_LAYERS:
                results_a[idx].append(pooled_a[idx])
                results_v[idx].append(pooled_v[idx])
                results_lt[idx].append(pooled_lt[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_a[idx].append(np.zeros(2048, dtype=np.float32))
                results_v[idx].append(np.zeros(2048, dtype=np.float32))
                results_lt[idx].append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        model_name = f"{MODEL_TAG}_layer{idx}"
        out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr_a = np.array(results_a[idx], dtype=np.float32)
        arr_v = np.array(results_v[idx], dtype=np.float32)
        np.save(out_dir / f"{model_name}_a.npy", arr_a)
        np.save(out_dir / f"{model_name}_v.npy", arr_v)
        print(f"[{model_name}] saved unimodal _a={arr_a.shape} _v={arr_v.shape} -> {out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lasttoken"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_lt = np.array(results_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_lt)
        print(f"[{lt_model_name}] saved lasttoken _av={arr_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
