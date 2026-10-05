"""
notebooks/feature_extraction/omni3b_extract_dummy_modality.py
=================================================================
Omni3B analogue of topo_omni_extract_dummy_modality.py / nemotron_extract_
dummy_modality.py -- extracts Qwen2.5-Omni-3B's own joint-pass hidden
embedding with one real modality and one fixed content-free dummy
substitute, at TARGET_LAYERS=[9,18,27] (matching omni3b_extract_scramble.py).

omni3b is not used as a localizer driver/sheet in this round -- it feeds
ONLY the whole-embedding dummy-modality diff study (rsa/dummy_diff_maps.py),
the same role its existing omni3b_layer{N}_mp_avscramble.npy plays in
rsa/scramble_diff_maps.py.

--dummy-modality a: dummy VIDEO + real AUDIO -> omni3b_layer{N}_mp_clsav_from_a
--dummy-modality v: dummy AUDIO + real VIDEO -> omni3b_layer{N}_mp_clsav_from_v

Dummy stimuli (shared across every dummy-modality extraction in this repo):
  data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4
  data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav

Run with:
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/omni3b_extract_dummy_modality.py --dummy-modality a
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/omni3b_extract_dummy_modality.py --dummy-modality v
"""

import argparse
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

from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_vl_utils import process_vision_info
from natsort import natsorted

MODEL_PATH      = "Qwen/Qwen2.5-Omni-3B"
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS  # noqa: E402

DATA_BASE       = DATA / "segmented_stimulus/filtered"
EMBEDDINGS_BASE = OUTPUTS / "model_embeddings"
DUMMY_VIDEO     = DATA / "segmented_stimulus/dummy_blank/dummy_black_5s.mp4"
DUMMY_AUDIO     = DATA / "segmented_stimulus/dummy_blank/dummy_silence_5s.wav"
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


def find_all_segments(data_base: Path, bin_sec: float, skip_sec: float) -> list[Path]:
    dur_int, skip_int = int(bin_sec), int(skip_sec)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    return natsorted(list(data_base.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dummy-modality", required=True, choices=["a", "v"], dest="dummy_modality",
                   help="'a' = dummy VIDEO + real AUDIO; 'v' = dummy AUDIO + real VIDEO")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="smoke-test: only process first N segments")
    args = p.parse_args()

    assert DUMMY_VIDEO.exists(), f"missing {DUMMY_VIDEO}"
    assert DUMMY_AUDIO.exists(), f"missing {DUMMY_AUDIO}"

    tag = f"clsav_from_{args.dummy_modality}"
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    out_paths = {idx: EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_mp_{tag}"
                 / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_mp_{tag}_av.npy"
                 for idx in TARGET_LAYERS}
    out_paths.update({
        f"{idx}_lt": EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_lt_{tag}"
        / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_lt_{tag}_av.npy"
        for idx in TARGET_LAYERS
    })
    if not args.force and all(pth.exists() for pth in out_paths.values()):
        print(f"All outputs already exist for tag={tag} -- skipping (use --force to overwrite).")
        return

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

    def _build_dummy_inputs(video_path: Path):
        real_audio_path = _find_audio_path(video_path)
        if args.dummy_modality == "a":
            vid_for_msg, aud_for_msg = DUMMY_VIDEO, real_audio_path  # dummy VIDEO + real AUDIO
        else:
            vid_for_msg, aud_for_msg = video_path, DUMMY_AUDIO       # dummy AUDIO + real VIDEO
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": str(vid_for_msg)},
                {"type": "audio", "audio": str(aud_for_msg)},
                {"type": "text", "text": "Describe what you see and hear."},
            ]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, frames = process_vision_info(msgs)
        audio_array = _load_audio(aud_for_msg, AUDIO_SR)
        return text, frames, audio_array

    def extract_dummy_av(video_path: Path, target_layers=TARGET_LAYERS):
        text, frames, audio_array = _build_dummy_inputs(video_path)
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

    all_segs = find_all_segments(DATA_BASE, BIN_SEC, SKIP_SEC)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments. dummy_modality={args.dummy_modality} (tag={tag})")

    if args.limit:
        all_segs = all_segs[: args.limit]
        print(f"--limit set -- truncated to {len(all_segs)} segments (smoke test).")

    D = 2048
    results_av = {idx: [] for idx in TARGET_LAYERS}
    results_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp in tqdm(all_segs, desc=f"Omni3B dummy-modality ({tag}) extraction"):
        try:
            pooled_av, lasttoken_av = extract_dummy_av(vp)
            for idx in TARGET_LAYERS:
                results_av[idx].append(pooled_av[idx])
                results_lt[idx].append(lasttoken_av[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                results_av[idx].append(np.zeros(D, dtype=np.float32))
                results_lt[idx].append(np.zeros(D, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        av_model_name = f"{MODEL_TAG}_layer{idx}_mp_{tag}"
        av_out_dir = EMBEDDINGS_BASE / av_model_name / f"bin{dur_int}s_skip{skip_int}s"
        av_out_dir.mkdir(parents=True, exist_ok=True)
        arr_av = np.array(results_av[idx], dtype=np.float32)
        np.save(av_out_dir / f"{av_model_name}_av.npy", arr_av)
        print(f"[{av_model_name}] saved shape={arr_av.shape} -> {av_out_dir}")

        lt_model_name = f"{MODEL_TAG}_layer{idx}_lt_{tag}"
        lt_out_dir = EMBEDDINGS_BASE / lt_model_name / f"bin{dur_int}s_skip{skip_int}s"
        lt_out_dir.mkdir(parents=True, exist_ok=True)
        arr_lt = np.array(results_lt[idx], dtype=np.float32)
        np.save(lt_out_dir / f"{lt_model_name}_av.npy", arr_lt)
        print(f"[{lt_model_name}] saved shape={arr_lt.shape} -> {lt_out_dir}")

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
