"""
notebooks/feature_extraction/topo_omni_extract_dummy_modality.py
====================================================================
TopoOmni analogue of pe_av_extract_dummy_modality.py -- extracts TopoOmni's
own cortical-sheet (Z) activation for the SAME joint-AV forward pass used by
topo_omni_extract.py's extract_av_embeddings() / get_all_masks(), pooled with
the SAME (mean_pool(a_mask) + mean_pool(v_mask)) / 2 formula (equal
per-modality weighting, NOT a single joint mean over av_mask=a_mask|v_mask,
which would silently weight by each modality's token count instead), but
with one modality replaced by a fixed content-free dummy stimulus.

Why this is needed: the in-silico AV-integration localizer requires testing
whether TopoOmni's own cortical sheet represents genuinely-paired (real
audio + real video) input differently from a unimodal-plus-dummy substitute
(real audio + blank video, or blank audio + real video) -- i.e. does
Ward's-linkage clustering over {av, a+dummy, v+dummy} (626 x 3 = 1878
"stimuli") put the true-av bins in a cluster distinct from the dummy-padded
ones? That test needs TopoOmni's OWN sheet response to all three conditions,
not just the real av response already saved as topoomni_layer{N}_sheet_mp_av.npy.

Dummy stimuli (already generated for the PE-AV dummy-modality extraction,
reused here unchanged):
  data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4
  data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav

Pooling matches topo_omni_extract.py's sheet_av computation exactly:
(mean_pool(a_mask) + mean_pool(v_mask)) / 2 from a SINGLE joint forward pass,
with the same Whisper-frame audio-token trimming (n_valid_audio) logic -- so
the new clsav_from_a/v sheet arrays are directly comparable, row-for-row, to
the existing topoomni_layer{N}_sheet_mp_av.npy.

ALSO saves the true-last-token variant (topoomni_layer{N}_sheet_lt_
clsav_from_{a,v}) from the SAME forward pass, matching
topo_omni_extract_intact.py's non-degenerate alternative to the masked-pool
(a+v)/2 average -- needed wherever topoomni's sheet is used as either the
driver or the scored sheet, to avoid the (a+v)/2 circularity.

ALSO saves the HIDDEN (X_hat, pre-cortical-bottleneck) pooled + last-token
variants (topoomni_layer{N}_mp_clsav_from_{a,v} / topoomni_layer{N}_lt_
clsav_from_{a,v}), matching the real topoomni_layer{N}_mp_av.npy naming -- needed
for the whole-embedding dummy-modality diff study (rsa/dummy_diff_maps.py),
which diffs against the real hidden embedding, not the cortical sheet.

--dummy-modality a: dummy VIDEO + real AUDIO -> topoomni_layer{N}_sheet_mp_clsav_from_a
--dummy-modality v: dummy AUDIO + real VIDEO -> topoomni_layer{N}_sheet_mp_clsav_from_v

Run with:
    conda run --no-capture-output -n topo_omni \
        python notebooks/feature_extraction/topo_omni_extract_dummy_modality.py --dummy-modality a
    conda run --no-capture-output -n topo_omni \
        python notebooks/feature_extraction/topo_omni_extract_dummy_modality.py --dummy-modality v
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

import argparse
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
DUMMY_VIDEO     = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_black_5s.mp4")
DUMMY_AUDIO     = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/dummy_blank/dummy_silence_5s.wav")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
TARGET_LAYERS   = [9, 18, 27]
MODEL_TAG       = "topoomni"
AUDIO_SR        = 16000

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
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


def _find_audio_path(video_path: Path) -> Path:
    parts = [p.replace("Video", "Audio") for p in video_path.parts]
    wav_path = Path(*parts).with_suffix(".wav")
    return wav_path if wav_path.exists() else video_path


def find_all_segments(data_base: Path, bin_sec: float, skip_sec: float) -> list[Path]:
    dur_int, skip_int = int(bin_sec), int(skip_sec)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    segs = natsorted(list(data_base.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    return segs


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

    out_paths = {idx: EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_sheet_mp_{tag}"
                 / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_sheet_mp_{tag}_av.npy"
                 for idx in TARGET_LAYERS}
    out_paths.update({
        f"{idx}_lt": EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_sheet_lt_{tag}"
        / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_sheet_lt_{tag}_av.npy"
        for idx in TARGET_LAYERS
    })
    out_paths.update({
        f"{idx}_hidden": EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_mp_{tag}"
        / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_mp_{tag}_av.npy"
        for idx in TARGET_LAYERS
    })
    out_paths.update({
        f"{idx}_hidden_lt": EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_lt_{tag}"
        / f"bin{dur_int}s_skip{skip_int}s" / f"{MODEL_TAG}_layer{idx}_lt_{tag}_av.npy"
        for idx in TARGET_LAYERS
    })
    if not args.force and all(p_.exists() for p_ in out_paths.values()):
        print(f"All outputs already exist for tag={tag} -- skipping (use --force to overwrite).")
        return

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

    AUDIO_TOKEN_ID = model.config.audio_token_id
    VIDEO_TOKEN_ID = model.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

    N_LAYERS = model.config.text_config.num_hidden_layers

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

    def _build_joint_inputs(video_path: Path):
        real_audio_path = _find_audio_path(video_path)
        if args.dummy_modality == "a":
            # dummy VIDEO + real AUDIO
            vid_for_msg, aud_for_msg = DUMMY_VIDEO, real_audio_path
        else:
            # dummy AUDIO + real VIDEO
            vid_for_msg, aud_for_msg = video_path, DUMMY_AUDIO

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": str(vid_for_msg)},
                {"type": "audio", "audio": str(aud_for_msg)},
                {"type": "text",  "text":  "Describe what you see and hear."},
            ]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, frames = process_vision_info(msgs)
        audio_array = _load_audio(aud_for_msg, AUDIO_SR)
        return text, frames, audio_array

    _masks_verified = {"done": False}

    def _get_all_masks(input_ids: torch.Tensor):
        ids = input_ids.squeeze(0)
        a_mask = ids == AUDIO_TOKEN_ID
        v_mask = ids == VIDEO_TOKEN_ID
        av_mask = a_mask | v_mask
        if not _masks_verified["done"]:
            n_a, n_v, n_av = int(a_mask.sum()), int(v_mask.sum()), int(av_mask.sum())
            print(f"[Mask verify] audio={n_a} video={n_v} av={n_av} total={len(ids)}")
            assert n_a > 0 and n_v > 0 and n_av == n_a + n_v
            _masks_verified["done"] = True
        return av_mask.cpu(), a_mask.cpu(), v_mask.cpu()

    def _hook_capture(target_layers):
        captured_hidden = {}
        captured_sheet = {}
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

    def extract_dummy_av(video_path: Path, target_layers=TARGET_LAYERS):
        text, frames, audio_array = _build_joint_inputs(video_path)
        n_samples = len(audio_array)
        n_mel = int(np.ceil(n_samples / 160))
        n_valid_audio = min(int(np.ceil(n_mel / 2)), 1500)

        inputs = processor(text=[text], videos=frames, audio=audio_array,
                            sampling_rate=AUDIO_SR, return_tensors="pt").to(_MODEL_DEVICE)

        captured_hidden, captured_sheet, handles = _hook_capture(target_layers)
        try:
            with torch.inference_mode():
                model(**inputs, output_hidden_states=False)
        finally:
            for h in handles:
                h.remove()

        _, a_mask_full, v_mask = _get_all_masks(inputs["input_ids"])
        del inputs
        torch.cuda.empty_cache()

        a_positions = a_mask_full.nonzero(as_tuple=True)[0]
        n_clip = min(n_valid_audio, len(a_positions))
        a_mask = torch.zeros_like(a_mask_full)
        a_mask[a_positions[:n_clip]] = True

        def _pool(zs, mask):
            if mask.any():
                return zs[0, mask, :].mean(dim=0).float().numpy()
            return np.zeros(zs.shape[-1], dtype=np.float32)

        # Match topo_omni_extract.py's real sheet_av formula exactly:
        # (mean_pool(a_mask) + mean_pool(v_mask)) / 2 -- equal per-modality
        # weighting, NOT a single joint mean over av_mask=a_mask|v_mask (which
        # would implicitly weight by each modality's token count instead).
        # Same pooling is applied to the hidden (X_hat) stream for the
        # whole-embedding dummy-modality diff study.
        pooled_sheet, lasttoken_sheet = {}, {}
        pooled_hidden, lasttoken_hidden = {}, {}
        for idx in target_layers:
            zs = captured_sheet[idx]
            hs = captured_hidden[idx]
            pooled_sheet[idx] = (_pool(zs, a_mask) + _pool(zs, v_mask)) / 2.0
            lasttoken_sheet[idx] = zs[0, -1, :].float().numpy()
            pooled_hidden[idx] = (_pool(hs, a_mask) + _pool(hs, v_mask)) / 2.0
            lasttoken_hidden[idx] = hs[0, -1, :].float().numpy()
        return pooled_sheet, lasttoken_sheet, pooled_hidden, lasttoken_hidden

    all_segs = find_all_segments(DATA_BASE, BIN_SEC, SKIP_SEC)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments. dummy_modality={args.dummy_modality} (tag={tag})")

    if args.limit:
        all_segs = all_segs[: args.limit]
        print(f"--limit set -- truncated to {len(all_segs)} segments (smoke test).")

    D = 2048
    res_sheet = {idx: [] for idx in TARGET_LAYERS}
    res_sheet_lt = {idx: [] for idx in TARGET_LAYERS}
    res_hidden = {idx: [] for idx in TARGET_LAYERS}
    res_hidden_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    for vp in tqdm(all_segs, desc=f"Dummy-modality ({tag}) extraction"):
        try:
            ps, ps_lt, ph, ph_lt = extract_dummy_av(vp)
            for idx in TARGET_LAYERS:
                res_sheet[idx].append(ps[idx])
                res_sheet_lt[idx].append(ps_lt[idx])
                res_hidden[idx].append(ph[idx])
                res_hidden_lt[idx].append(ph_lt[idx])
        except Exception as e:
            failed.append((vp.name, repr(e)))
            for idx in TARGET_LAYERS:
                res_sheet[idx].append(np.zeros(D, dtype=np.float32))
                res_sheet_lt[idx].append(np.zeros(D, dtype=np.float32))
                res_hidden[idx].append(np.zeros(D, dtype=np.float32))
                res_hidden_lt[idx].append(np.zeros(D, dtype=np.float32))

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    def _save(name: str, rows: list) -> None:
        out_dir = EMBEDDINGS_BASE / name / f"bin{dur_int}s_skip{skip_int}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array(rows, dtype=np.float32)
        out_path = out_dir / f"{name}_av.npy"
        np.save(out_path, arr)
        print(f"[{name}] saved shape={arr.shape} -> {out_path}")

    for idx in TARGET_LAYERS:
        _save(f"{MODEL_TAG}_layer{idx}_sheet_mp_{tag}", res_sheet[idx])
        _save(f"{MODEL_TAG}_layer{idx}_sheet_lt_{tag}", res_sheet_lt[idx])
        _save(f"{MODEL_TAG}_layer{idx}_mp_{tag}", res_hidden[idx])
        _save(f"{MODEL_TAG}_layer{idx}_lt_{tag}", res_hidden_lt[idx])

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
