"""
notebooks/feature_extraction/omni3b_extract_scramble.py
==========================================================
Move 3 (temporal-scramble binding control) for omni3b -- companion to
pe_av_extract_scramble.py and extract_cav_mae_sync.py's --scramble-av mode.
omni3b/topo-omni are the two models where the av=(a+v)/2 circularity bug was
found and fixed (see omni3b_extract_unimodal.py); the same models are central
enough to Move 1/4/5 that the binding control needs to run on them too, not
just the smaller cav-mae-sync check.

Unimodal "_a"/"_v" embeddings do NOT need re-extraction here: since
omni3b_extract_unimodal.py, they come from genuinely separate forward passes
with no cross-modal tokens present, so they cannot depend on which audio was
paired with which video. Only the JOINT (audio+video) forward pass depends on
pairing -- so this script re-runs only the joint pass, with video[i] paired
against a RANDOMLY PERMUTED audio[perm(i)] (fixed seed 42, same convention as
the other scramble scripts), and re-extracts both joint-pass readouts:
  {tag}_layer{N}_avscramble_av.npy            -- masked-pool (a+v)/2, scrambled pairing
  {tag}_layer{N}_lasttoken_avscramble_av.npy   -- last-token joint readout, scrambled pairing
Both compared against their INTACT counterparts (already on disk from
omni3b_extract_unimodal.py's _av and _lasttoken_av) to test whether the
integration signal survives temporal/identity mismatch between modalities.

Run with:
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/omni3b_extract_scramble.py"
"""

import gc
import os
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
SCRAMBLE_SEED   = 42

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

    def _build_joint_inputs(video_path, audio_path):
        """Joint audio+video message, with an EXPLICIT (possibly mismatched) audio_path."""
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

    def extract_joint(video_path, audio_path, target_layers=TARGET_LAYERS):
        """Joint AV forward pass with an explicit (video, audio) pair.
        Returns (masked_pool_av, lasttoken_av) dicts keyed by layer idx."""
        text, frames, audio_array = _build_joint_inputs(video_path, audio_path)
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

        ids = inputs["input_ids"].squeeze(0).cpu()
        a_mask = (ids == AUDIO_TOKEN_ID)
        v_mask = (ids == VIDEO_TOKEN_ID)
        del inputs
        torch.cuda.empty_cache()

        pooled_av, lasttoken_av = {}, {}
        for idx in target_layers:
            hs = captured[idx]  # (1, seq_len, D)
            a_pool = hs[0, a_mask, :].mean(dim=0).float().numpy() if a_mask.any() else np.zeros(hs.shape[-1], dtype=np.float32)
            v_pool = hs[0, v_mask, :].mean(dim=0).float().numpy() if v_mask.any() else np.zeros(hs.shape[-1], dtype=np.float32)
            pooled_av[idx] = (a_pool + v_pool) / 2.0
            lasttoken_av[idx] = hs[0, -1, :].float().numpy()
        return pooled_av, lasttoken_av

    # ── Segment list + scrambled pairing (same convention as pe_av/cavmae scramble scripts) ──
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    audio_paths = [_find_audio_path(vp) for vp in all_segs]
    rng = np.random.default_rng(SCRAMBLE_SEED)
    perm = rng.permutation(len(audio_paths))
    n_fixed = int((perm == np.arange(len(audio_paths))).sum())
    print(f"[SCRAMBLE_AV] Permuted {len(audio_paths)} audio segments "
          f"(seed={SCRAMBLE_SEED}, {n_fixed} incidental self-pairs)")
    audio_paths_scrambled = [audio_paths[i] for i in perm]

    _limit = os.environ.get("OMNI3B_SCRAMBLE_LIMIT")
    if _limit:
        n = int(_limit)
        all_segs = all_segs[:n]
        audio_paths_scrambled = audio_paths_scrambled[:n]
        print(f"OMNI3B_SCRAMBLE_LIMIT set -- truncated to {n} segments (smoke test).")

    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp, ap in tqdm(list(zip(all_segs, audio_paths_scrambled)), desc="Scrambled joint extraction"):
        try:
            pooled_av, lasttoken_av = extract_joint(vp, ap)
            for idx in TARGET_LAYERS:
                results_av[idx].append(pooled_av[idx])
                results_lt[idx].append(lasttoken_av[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_av[idx].append(np.zeros(2048, dtype=np.float32))
                results_lt[idx].append(np.zeros(2048, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        av_model_name = f"{MODEL_TAG}_layer{idx}_avscramble"
        av_out_dir = EMBEDDINGS_BASE / av_model_name / f"bin{dur_int}s_skip{skip_int}s"
        av_out_dir.mkdir(parents=True, exist_ok=True)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        np.save(av_out_dir / f"{av_model_name}_av.npy", arr_av)
        print(f"[{av_model_name}] saved scrambled av={arr_av.shape} -> {av_out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lasttoken_avscramble"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_lt = np.array(results_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_lt)
        print(f"[{lt_model_name}] saved scrambled lasttoken av={arr_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
