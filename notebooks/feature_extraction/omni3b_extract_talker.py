"""
notebooks/feature_extraction/omni3b_extract_talker.py
=======================================================================
Talker-stage embeddings for omni3b, sibling to the Thinker-stage family
(omni3b_extract_intact.py / omni3b_extract_thinker_penultimate.py).

Talker has NO audio/video placeholder tokens of its own (confirmed: its
config carries no audio_token_id/video_token_id-style field, and its input
stream is Thinker's fused hidden states + generated codec/text token
embeddings, not raw modality tokens) -- so there is no "_a"/"_v" split to
extract here, only a single joint "_av" readout per layer/pooling pair.
This matches the Thinker family's masked-mean-pool rationale (mask =
"which positions are this modality") degrading to "no mask needed" when
there is nothing to mask by.

Conditioning pathway (ported from transformers' own
Qwen2_5OmniForConditionalGeneration.generate(), modeling_qwen2_5_omni.py,
minus Token2Wav which we never load):
  1. thinker.generate(..., output_hidden_states=True, return_dict_in_generate=True)
     -- greedy (do_sample=False) so the whole pipeline is deterministic.
  2. Thinker's per-generated-token hidden states + input embeddings are
     spliced into `talker_inputs_embeds` / `thinker_reply_part` exactly as
     the official generate() does (audio/video token positions zeroed out
     of the embed stream, since Talker gets the FUSED hidden state there
     instead).
  3. talker.generate(inputs_embeds=..., thinker_reply_part=..., ...) then
     autoregressively emits codec tokens (also greedy for determinism).
     We never construct or call token2wav -- no waveform is produced.

Hidden-state capture: forward hooks on talker.model.layers[i], identical
mechanism to the Thinker family's `_find_decoder_layers` + hook pattern
(NOT output_hidden_states=True on the layer stack itself -- hooks are the
sole capture path, matching project convention).

Talker decoder has N_LAYERS=24 (0-indexed), hidden_size=896 (verified via
Qwen2_5OmniConfig().talker_config -- distinct from Thinker's 36 layers /
2048 hidden, so layer indices are NOT directly comparable across the two
model stages despite reusing the same decoder-layer class). Requested
"first/middle/last transformer layer" -> indices 0, 12, 23.

Pooling positions: a Talker forward call during generate() is invoked once
for the prefill (processes the whole spliced conditioning context in one
shot) and once per subsequently generated codec token (KV-cached, one new
position each). The prefill's *last* position is the state that produced
codec token #1's logits -- exactly analogous in kind to every later
decode-step hidden state (the state that produced token k+1's logits) --
so we build a single per-generated-token trajectory as
    [ prefill_hidden[:, -1:, :] ] + [ decode_step_hidden for each later call ]
and pool over that (mean over all positions for "_mp", last position for
"_lt"). This uniformly excludes the non-generated conditioning-context
positions (the injected Thinker fusion + prompt), and needs no padding
mask since every segment is processed as its own batch-of-1 sequence.

Run with:
    conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/omni3b_extract_talker.py"

Smoke test one segment first:
    OMNI3B_TALKER_LIMIT=1 conda run --no-capture-output -n avtransformer \
        python "notebooks/feature_extraction/omni3b_extract_talker.py"
"""

import gc
import os
import time
from pathlib import Path

# Must be set before `import torch` initializes the CUDA allocator: reduces
# fragmentation-driven OOM (recommended directly in PyTorch's own OOM error
# text) -- the 12 GB card runs at ~95%+ peak utilization for this pipeline.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import av
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniTalkerForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from natsort import natsorted

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
for _v in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)

# ── Config ────────────────────────────────────────────────────────────────
# ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B is a dead symlink to
# an unmounted drive on this machine -- resolve the checkpoint snapshot dir
# directly against whichever mirror is actually mounted (same pattern as
# topo_omni_extract_full_sheet.py's _HF_SNAPSHOT_CANDIDATES).
_MODEL_SNAPSHOT_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B",
    Path("/media/amin/ADATA HD710 PRO/hf_models/hub/models--Qwen--Qwen2.5-Omni-3B"),
]


def _resolve_model_path() -> str:
    for base in _MODEL_SNAPSHOT_CANDIDATES:
        snap_dir = base / "snapshots"
        if snap_dir.is_dir():
            for child in sorted(snap_dir.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    return str(child)
    raise RuntimeError(f"No usable Qwen2.5-Omni-3B snapshot found. Tried: {_MODEL_SNAPSHOT_CANDIDATES}")


_DATA_BASE_CANDIDATES = [
    Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered"),
    Path("/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus/filtered"),
]
DATA_BASE = next((p for p in _DATA_BASE_CANDIDATES if p.is_dir() and any(p.iterdir())), None)
assert DATA_BASE is not None, f"No usable segmented_stimulus/filtered directory. Tried: {_DATA_BASE_CANDIDATES}"

MODEL_PATH      = _resolve_model_path()
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
BIN_SEC, SKIP_SEC = 5.0, 5.0
MODEL_TAG       = "omni3b_talker"
AUDIO_SR        = 16000
SPEAKER         = "Chelsie"  # fixed for determinism/reproducibility across all 626 segments
THINKER_MAX_NEW_TOKENS = int(os.environ.get("THINKER_MAX_NEW_TOKENS", "128"))
TALKER_MAX_NEW_TOKENS  = int(os.environ.get("TALKER_MAX_NEW_TOKENS", "256"))

_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
_INSTRUCTION_PROMPT = "Describe what you see and hear."


def main():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f"Loading Qwen2.5-Omni-3B Thinker + Talker from {MODEL_PATH} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, device_map={"": DEVICE}, torch_dtype=DTYPE,
        attn_implementation="sdpa", local_files_only=True,
    )
    thinker.eval()
    talker = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained(
        MODEL_PATH, device_map={"": "cpu"}, torch_dtype=DTYPE,
        attn_implementation="sdpa", local_files_only=True,
    )
    talker.eval()
    _DEVICE = next(thinker.parameters()).device
    print(f"Thinker loaded on {_DEVICE}; Talker held on CPU until its phase.")

    # Thinker alone is ~9.4 GB bf16 resident (verified: vision+audio towers +
    # 36-layer 2048-hidden LM -- most of the 12 GB budget by itself), so it
    # cannot stay GPU-resident while Talker also runs (Talker + its KV cache
    # over up to TALKER_MAX_NEW_TOKENS decode steps was enough to OOM on
    # top of it). Two full passes instead: Phase A runs Thinker for EVERY
    # segment and caches the resulting Talker-conditioning tensors to disk;
    # Thinker is then fully released (`.to("cpu")`, confirmed empirically to
    # drop allocated CUDA memory to 0) and Phase B loads Talker and consumes
    # the cached conditioning per segment. This is the "cache complete
    # conditioning tensors on CPU/disk and load Talker independently"
    # alternative the extraction spec explicitly authorizes. Disk (not just
    # RAM) so a crash mid-Phase-B doesn't force re-running Phase A too.
    CACHE_DIR = EMBEDDINGS_BASE / "_talker_conditioning_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    speaker_map = torch.load(Path(MODEL_PATH) / "spk_dict.pt", weights_only=True)
    assert SPEAKER in speaker_map, f"speaker {SPEAKER!r} not in {list(speaker_map.keys())}"
    speaker_params = speaker_map[SPEAKER]

    N_LAYERS = talker.model.config.num_hidden_layers
    TARGET_LAYERS = sorted({0, N_LAYERS // 2, N_LAYERS - 1})
    print(f"Talker N_LAYERS={N_LAYERS}  TARGET_LAYERS(first/middle/last)={TARGET_LAYERS}  hidden_size={talker.config.hidden_size}")

    decoder_layers = talker.model.layers

    AUDIO_TOKEN_ID = thinker.config.audio_token_id
    VIDEO_TOKEN_ID = thinker.config.video_token_id
    assert AUDIO_TOKEN_ID is not None and VIDEO_TOKEN_ID is not None

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
            {"type": "text", "text": _INSTRUCTION_PROMPT},
        ]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": _QWEN_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        _, frames = process_vision_info(msgs)
        audio_array = _load_audio(audio_path, AUDIO_SR)
        return text, frames, audio_array

    def _thinker_to_talker_conditioning(inputs):
        """Port of Qwen2_5OmniForConditionalGeneration.generate() steps 1-2
        (modeling_qwen2_5_omni.py), stopping before token2wav."""
        input_ids = inputs["input_ids"]
        thinker_result = thinker.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
            video_second_per_grid=inputs["video_second_per_grid"],
            use_audio_in_video=False,
            max_new_tokens=THINKER_MAX_NEW_TOKENS,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(_DEVICE)
        for token_id in (AUDIO_TOKEN_ID, VIDEO_TOKEN_ID):
            ids_mask = input_ids == token_id
            mask = ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            zeros = torch.zeros([ids_mask.sum(), embeds_to_talker.shape[-1]],
                                dtype=embeds_to_talker.dtype, device=_DEVICE)
            embeds_to_talker.masked_scatter_(mask, zeros)

        processed_thinker_hidden = (
            (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        ) + thinker_result.hidden_states[1:]
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1):].to(_DEVICE)
        thinker_token_embeds = [t[0].to(_DEVICE) for t in processed_thinker_hidden]
        thinker_hidden_states = [t[-1].to(_DEVICE) for t in processed_thinker_hidden]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [input_ids,
             torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=_DEVICE),
             thinker_generate_ids[:, :1]], dim=-1,
        )
        talker_input_ids = torch.cat(
            [torch.full_like(input_ids, fill_value=talker.codec_mask_token),
             torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=_DEVICE),
             torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=_DEVICE)], dim=1,
        )

        thinker_embed_tokens = thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_embed = thinker_embed_tokens(
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=_DEVICE)
        )
        talker_inputs_embeds = torch.cat(
            [talker_inputs_embeds, talker_text_bos_embed, thinker_reply_part[:, :1, :]], dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor([[talker.text_eos_token]], dtype=torch.long, device=_DEVICE)
        )
        pad_embedding = thinker_embed_tokens(
            torch.tensor([[talker.text_pad_token]], dtype=torch.long, device=_DEVICE)
        )
        thinker_reply_part = torch.cat(
            [thinker_reply_part[:, 1:, :], eos_embedding, pad_embedding], dim=1,
        )

        talker_attention_mask = torch.cat(
            [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 2))], dim=1
        ).to(_DEVICE)

        return dict(
            talker_input_ids=talker_input_ids,
            talker_input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            talker_inputs_embeds=talker_inputs_embeds,
            talker_attention_mask=talker_attention_mask,
            video_grid_thw=inputs["video_grid_thw"],
            video_second_per_grid=inputs["video_second_per_grid"],
            audio_feature_lengths=torch.sum(inputs["feature_attention_mask"], dim=1),
        )

    def build_conditioning(video_path):
        """Phase A (runs while Thinker is GPU-resident): build the Talker
        conditioning tensors for one segment and return them on CPU."""
        text, frames, audio_array = _build_joint_inputs(video_path)
        inputs = processor(text=[text], videos=frames, audio=audio_array,
                           sampling_rate=AUDIO_SR, return_tensors="pt").to(_DEVICE)
        with torch.inference_mode():
            cond = _thinker_to_talker_conditioning(inputs)
        cond_cpu = {k: v.cpu() for k, v in cond.items()}
        del inputs, cond
        torch.cuda.empty_cache()
        return cond_cpu

    def run_talker(cond_cpu, target_layers=TARGET_LAYERS):
        """Phase B (runs while Talker is GPU-resident): autoregressively
        generate codec tokens from cached conditioning, hooking hidden
        states at target_layers, and return pooled (mean, last-token) reps."""
        cond = {k: v.to(_DEVICE) for k, v in cond_cpu.items()}

        captured = {idx: [] for idx in target_layers}
        handles = []

        def _make_hook(idx):
            def _hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                captured[idx].append(hs.detach().cpu())
            return _hook

        for i in target_layers:
            handles.append(decoder_layers[i].register_forward_hook(_make_hook(i)))
        try:
            with torch.inference_mode():
                talker.generate(
                    input_ids=cond["talker_input_ids"],
                    input_text_ids=cond["talker_input_text_ids"],
                    thinker_reply_part=cond["thinker_reply_part"],
                    inputs_embeds=cond["talker_inputs_embeds"],
                    attention_mask=cond["talker_attention_mask"],
                    video_grid_thw=cond["video_grid_thw"],
                    video_second_per_grid=cond["video_second_per_grid"],
                    audio_feature_lengths=cond["audio_feature_lengths"],
                    use_audio_in_video=False,
                    suppress_tokens=[talker.codec_bos_token],
                    max_new_tokens=TALKER_MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                    eos_token_id=[talker.codec_pad_token, talker.codec_eos_token],
                )
        finally:
            for h in handles:
                h.remove()

        del cond
        torch.cuda.empty_cache()

        pooled_mp, pooled_lt = {}, {}
        for idx in target_layers:
            chunks = captured[idx]
            assert len(chunks) >= 1, f"layer {idx}: talker produced zero forward calls"
            trajectory = torch.cat([chunks[0][:, -1:, :]] + chunks[1:], dim=1)  # (1, n_generated, D)
            assert torch.isfinite(trajectory).all(), f"layer {idx}: non-finite hidden states"
            pooled_mp[idx] = trajectory[0].mean(dim=0).float().numpy()
            pooled_lt[idx] = trajectory[0, -1, :].float().numpy()
        return pooled_mp, pooled_lt

    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    assert len(all_segs) > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {len(all_segs)} segments.")

    _limit = os.environ.get("OMNI3B_TALKER_LIMIT")
    if _limit:
        all_segs = all_segs[: int(_limit)]
        print(f"OMNI3B_TALKER_LIMIT set -- truncated to {len(all_segs)} segments (smoke test).")

    HIDDEN = talker.config.hidden_size
    results_mp = {idx: [] for idx in TARGET_LAYERS}
    results_lt = {idx: [] for idx in TARGET_LAYERS}
    failed = []

    # Resumable: skip segments already saved from a prior partial run.
    out_dirs_mp = {idx: EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_mp" / f"bin{dur_int}s_skip{skip_int}s" for idx in TARGET_LAYERS}
    out_dirs_lt = {idx: EMBEDDINGS_BASE / f"{MODEL_TAG}_layer{idx}_lt" / f"bin{dur_int}s_skip{skip_int}s" for idx in TARGET_LAYERS}
    ckpt_path = EMBEDDINGS_BASE / f"{MODEL_TAG}_checkpoint.npz"
    start_idx = 0
    if ckpt_path.exists() and not _limit:
        ck = np.load(ckpt_path, allow_pickle=True)
        if int(ck["n_done"]) <= len(all_segs):
            start_idx = int(ck["n_done"])
            for idx in TARGET_LAYERS:
                results_mp[idx] = list(ck[f"mp_{idx}"])
                results_lt[idx] = list(ck[f"lt_{idx}"])
            failed = list(ck["failed"])
            print(f"Resuming from checkpoint: {start_idx}/{len(all_segs)} segments already done.")

    t_start = time.time()

    # ── Phase A: Thinker resident on GPU, build + cache conditioning for
    # every segment (resumable: a segment whose cache file already exists,
    # or is marked failed, is skipped). ──────────────────────────────────
    print(f"mem_allocated before Phase A: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    phase_a_failed = {}
    for i, vp in enumerate(tqdm(all_segs, desc="Phase A: Thinker conditioning")):
        if i < start_idx:
            continue  # already completed (and its cache file consumed) in a prior run
        cache_file = CACHE_DIR / f"seg_{i:04d}.pt"
        fail_marker = CACHE_DIR / f"seg_{i:04d}.failed"
        if cache_file.exists() or fail_marker.exists():
            continue
        try:
            cond_cpu = build_conditioning(vp)
            torch.save(cond_cpu, cache_file)
        except Exception as e:
            phase_a_failed[i] = repr(e)
            fail_marker.write_text(repr(e))
            print(f"Phase A FAILED {vp.name}: {e!r}")
    print(f"Phase A done. mem_allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    thinker.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After thinker->cpu: mem_allocated={torch.cuda.memory_allocated()/1e9:.2f} GB, "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
    talker.to(_DEVICE)
    print(f"After talker->gpu: mem_allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Phase B: Talker resident on GPU, consume cached conditioning. ────
    for i, vp in enumerate(tqdm(all_segs[start_idx:], initial=start_idx, total=len(all_segs), desc="Phase B: Talker generation")):
        real_i = start_idx + i
        cache_file = CACHE_DIR / f"seg_{real_i:04d}.pt"
        err = phase_a_failed.get(real_i)
        if err is None and not cache_file.exists():
            err = "cache file missing"
        if err is None:
            try:
                cond_cpu = torch.load(cache_file, weights_only=True)
                pooled_mp, pooled_lt = run_talker(cond_cpu)
                for idx in TARGET_LAYERS:
                    results_mp[idx].append(pooled_mp[idx])
                    results_lt[idx].append(pooled_lt[idx])
                cache_file.unlink()
            except Exception as e:
                err = repr(e)
        if err is not None:
            failed.append((vp.name, err))
            print(f"FAILED {vp.name}: {err}")
            for idx in TARGET_LAYERS:
                results_mp[idx].append(np.zeros(HIDDEN, dtype=np.float32))
                results_lt[idx].append(np.zeros(HIDDEN, dtype=np.float32))

        n_done = i + 1
        if not _limit and (n_done % 20 == 0 or n_done == len(all_segs) - start_idx):
            save_kwargs = {"n_done": start_idx + n_done, "failed": np.array(failed, dtype=object)}
            for idx in TARGET_LAYERS:
                save_kwargs[f"mp_{idx}"] = np.array(results_mp[idx], dtype=np.float32)
                save_kwargs[f"lt_{idx}"] = np.array(results_lt[idx], dtype=np.float32)
            np.savez(ckpt_path, **save_kwargs)

    elapsed = time.time() - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nElapsed: {elapsed:.1f}s for {len(all_segs) - start_idx} segments "
          f"({elapsed / max(1, len(all_segs) - start_idx):.2f}s/segment). Peak GPU memory: {peak_mem_gb:.2f} GB.")

    if failed:
        print(f"\n{len(failed)} segments FAILED (filled with zeros):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")

    for idx in TARGET_LAYERS:
        mp_name = f"{MODEL_TAG}_layer{idx}_mp"
        mp_dir = out_dirs_mp[idx]
        mp_dir.mkdir(parents=True, exist_ok=True)
        arr_mp = np.array(results_mp[idx], dtype=np.float32)
        np.save(mp_dir / f"{mp_name}_av.npy", arr_mp)
        print(f"[{mp_name}] saved mean-pool _av={arr_mp.shape} -> {mp_dir}")

        lt_name = f"{MODEL_TAG}_layer{idx}_lt"
        lt_dir = out_dirs_lt[idx]
        lt_dir.mkdir(parents=True, exist_ok=True)
        arr_lt = np.array(results_lt[idx], dtype=np.float32)
        np.save(lt_dir / f"{lt_name}_av.npy", arr_lt)
        print(f"[{lt_name}] saved last-token _av={arr_lt.shape} -> {lt_dir}")

    if not _limit and ckpt_path.exists():
        ckpt_path.unlink()
    if not _limit:
        for leftover in CACHE_DIR.glob("seg_*.failed"):
            leftover.unlink()
        if CACHE_DIR.is_dir() and not any(CACHE_DIR.iterdir()):
            CACHE_DIR.rmdir()

    print(f"Done. {len(failed)} / {len(all_segs)} segments failed.")


if __name__ == "__main__":
    main()
