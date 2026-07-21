"""
notebooks/feature_extraction/topo_omni_extract_intact.py
=============================================================
Topo-Omni analogue of omni3b_extract_intact.py -- fixes the same
methodological bug for topoomni_layer{9,18,27} (hidden/X_hat) and
topoomni_layer{9,18,27}_sheet (cortical-sheet/Z): the existing "_a"/"_v"
are mean-pooled from audio-/video-token positions within a SINGLE JOINT
forward pass, and "_av" = (_a + _v) / 2 exactly (verified empirically,
bit-identical). See omni3b_extract_intact.py's docstring for the full
rationale -- identical reasoning applies here, just with Topo-Omni's custom
CorticalAdaptor-patched Thinker class instead of the stock one.

Re-extracts "_a"/"_v" (both hidden and sheet) from GENUINELY SEPARATE
audio-only / video-only forward passes, for layers {9, 18, 27} only.
Also saves "_lasttoken" hidden+sheet variants (last sequence position of the
joint pass) as a non-tautological alternative to the masked-pool "_av".
"_av" itself is left unchanged on disk.

Run with:
    conda run --no-capture-output -n topo_omni \
        python "notebooks/feature_extraction/topo_omni_extract_intact.py"
"""

import gc
import os

# Must run BEFORE any transformers/huggingface_hub import: the HTTP client's
# proxy config gets locked in at import time, so stripping these afterward
# has no effect and local_files_only lookups fail with a bogus "couldn't
# connect" error even though the model is fully cached locally.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
           "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import json
import sys
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

TOPO_MODEL_ID   = "epfl-neuroai/topo-omni"
DATA_BASE       = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 2.0, 2.0
TARGET_LAYERS   = [9, 18, 27]
MODEL_TAG       = "topoomni"
AUDIO_SR        = 16000

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def main():
    MODEL_PATH = snapshot_download(TOPO_MODEL_ID, local_files_only=True)
    print(f"Resolved local snapshot dir: {MODEL_PATH}")

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

    N_LAYERS = model.config.text_config.num_hidden_layers
    print(f"N_LAYERS={N_LAYERS}  TARGET_LAYERS={TARGET_LAYERS}")

    AUDIO_TOKEN_ID = model.config.audio_token_id
    VIDEO_TOKEN_ID = model.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

    def _find_cortical_adaptors(n_expected):
        direct = getattr(getattr(model, "model", None), "cortical_adaptors", None)
        if direct is not None and len(direct) == n_expected and isinstance(direct[0], CorticalAdaptor):
            return direct
        for _, mod in model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and len(mod) == n_expected
                    and isinstance(mod[0], CorticalAdaptor)):
                return mod
        raise RuntimeError(f"Could not find CorticalAdaptor ModuleList of length {n_expected}")

    cortical_adaptors = _find_cortical_adaptors(N_LAYERS)
    print(f"Cortical adaptors found: {len(cortical_adaptors)}")

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

    def _hook_capture(target_layers):
        captured_hidden, captured_sheet = {}, {}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                x_hat, z = out
                captured_hidden[idx] = x_hat.detach().cpu()
                captured_sheet[idx] = z.detach().cpu()
            return _hook

        for i in target_layers:
            handles.append(cortical_adaptors[i].register_forward_hook(_make_hook(i)))
        return captured_hidden, captured_sheet, handles

    def extract_lasttoken(video_path, target_layers=TARGET_LAYERS):
        text, frames, audio_array = _build_joint_inputs(video_path)
        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)
        captured_hidden, captured_sheet, handles = _hook_capture(target_layers)
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()
        del inputs
        torch.cuda.empty_cache()
        hidden_lt = {idx: captured_hidden[idx][0, -1, :].float().numpy() for idx in target_layers}
        sheet_lt  = {idx: captured_sheet[idx][0, -1, :].float().numpy() for idx in target_layers}
        return hidden_lt, sheet_lt

    def _build_unimodal_inputs(video_path, modality):
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

        captured_hidden, captured_sheet, handles = _hook_capture(target_layers)
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

        pooled_hidden, pooled_sheet = {}, {}
        for idx in target_layers:
            hs = captured_hidden[idx]
            zs = captured_sheet[idx]
            if mask.any():
                pooled_hidden[idx] = hs[0, mask, :].mean(dim=0).float().numpy()
                pooled_sheet[idx]  = zs[0, mask, :].mean(dim=0).float().numpy()
            else:
                pooled_hidden[idx] = np.zeros(hs.shape[-1], dtype=np.float32)
                pooled_sheet[idx]  = np.zeros(zs.shape[-1], dtype=np.float32)
        return pooled_hidden, pooled_sheet

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("TOPOOMNI_UNIMODAL_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"TOPOOMNI_UNIMODAL_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    D = 2048
    res_h_a  = {idx: [] for idx in TARGET_LAYERS}
    res_h_v  = {idx: [] for idx in TARGET_LAYERS}
    res_s_a  = {idx: [] for idx in TARGET_LAYERS}
    res_s_v  = {idx: [] for idx in TARGET_LAYERS}
    res_h_lt = {idx: [] for idx in TARGET_LAYERS}
    res_s_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp in tqdm(all_segs, desc="Unimodal + lasttoken extraction"):
        try:
            ph_a, ps_a = extract_unimodal(vp, "audio")
            ph_v, ps_v = extract_unimodal(vp, "video")
            hlt, slt   = extract_lasttoken(vp)
            for idx in TARGET_LAYERS:
                res_h_a[idx].append(ph_a[idx]); res_h_v[idx].append(ph_v[idx])
                res_s_a[idx].append(ps_a[idx]); res_s_v[idx].append(ps_v[idx])
                res_h_lt[idx].append(hlt[idx]); res_s_lt[idx].append(slt[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                res_h_a[idx].append(np.zeros(D, dtype=np.float32)); res_h_v[idx].append(np.zeros(D, dtype=np.float32))
                res_s_a[idx].append(np.zeros(D, dtype=np.float32)); res_s_v[idx].append(np.zeros(D, dtype=np.float32))
                res_h_lt[idx].append(np.zeros(D, dtype=np.float32)); res_s_lt[idx].append(np.zeros(D, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    def _save(model_name, suffix, arr):
        out_dir = EMBEDDINGS_BASE / model_name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{model_name}_{suffix}.npy", np.array(arr, dtype=np.float32))
        print(f"[{model_name}] saved _{suffix}={np.array(arr).shape} -> {out_dir}")

    for idx in TARGET_LAYERS:
        hidden_name = f"{MODEL_TAG}_layer{idx}_mp"
        sheet_name  = f"{MODEL_TAG}_layer{idx}_sheet_mp"
        _save(hidden_name, "a", res_h_a[idx])
        _save(hidden_name, "v", res_h_v[idx])
        _save(sheet_name,  "a", res_s_a[idx])
        _save(sheet_name,  "v", res_s_v[idx])
        _save(f"{MODEL_TAG}_layer{idx}_lt", "av", res_h_lt[idx])
        _save(f"{MODEL_TAG}_layer{idx}_sheet_lt",  "av", res_s_lt[idx])

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
