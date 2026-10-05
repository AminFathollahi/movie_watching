"""
extract_cav_mae_sync.py
=======================
Standalone script to extract CAV-MAE-Sync embeddings for 5 s and 10 s RSA bins.

Run with:
    conda run --no-capture-output -n cav-mae-sync \
        python extract_cav_mae_sync.py

    # Move 3 (temporal-scramble binding control): each video segment is paired
    # with a RANDOMLY PERMUTED audio segment (fixed seed) instead of its own
    # temporally-corresponding audio, breaking correct A-V temporal binding
    # while preserving each modality's own marginal content distribution.
    # Output goes to a separate "cav-mae-sync_avscramble" model directory.
    CAV_MAE_SCRAMBLE_AV=1 conda run --no-capture-output -n cav-mae-sync \
        python extract_cav_mae_sync.py

Source data
-----------
Uses segmented_stimulus/filtered/ — pre-chunked mp4/wav files for all 18 videos.
Structure:
    Video{N}/Video{N}_chunks_{D}s/Video{N}_part_XXX.mp4
    Audio{N}/Audio{N}_chunks_{D}s/Audio{N}_part_XXX.wav

Audio files: stereo, 44100 Hz — converted to mono 16 kHz before fbank.
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT, OUTPUTS, EXTERNAL  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO_ROOT      = ROOT / "Model Repos/cav-mae-sync"
DATA_ROOT      = Path(os.environ.get("STIMULUS_DIR", str(EXTERNAL / "data/segmented_stimulus/filtered")))
MODEL_PATH     = REPO_ROOT / "pretrained_models" / "cav_mae_sync.pth"
SCRAMBLE_AV    = bool(int(os.environ.get("CAV_MAE_SCRAMBLE_AV", "0")))
SCRAMBLE_SEED  = 42
_MODEL_NAME    = "cav-mae-sync_avscramble" if SCRAMBLE_AV else "cav-mae-sync"
OUTPUT_ROOT    = Path(os.environ.get("EMBEDDINGS_BASE", str(OUTPUTS / "model_embeddings"))) / _MODEL_NAME

AUDIO_MEAN     = -5.081
AUDIO_STD      = 4.4849
TOTAL_FRAMES   = 16
BATCH_SIZE     = 4
CLIP_SR        = 16000    # target sample rate for fbank

SEGMENT_DURATIONS = ([float(os.environ["BIN_SEC"])] if "BIN_SEC" in os.environ
                     else [5.0] if SCRAMBLE_AV else [5.0, 10.0])

# Mel time-frames — must be a multiple of 16 for the patch embedding.
TARGET_LENGTHS = {1.0: 96, 2.0: 192, 5.0: 496, 10.0: 992}

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "src"))
import models  # noqa: E402


# ---------------------------------------------------------------------------
# Pair gathering
# ---------------------------------------------------------------------------
def gather_pairs(seg_sec: float) -> list:
    """
    Return sorted list of (mp4_path, wav_path) pairs for all 18 videos.
    Parts are matched by sort order within each video (both lists must be same length).
    """
    d = int(seg_sec)
    mp4_all, wav_all = [], []
    for vid_id in range(1, 19):
        mp4_dir = DATA_ROOT / f"Video{vid_id}" / f"Video{vid_id}_chunks_{d}s"
        wav_dir = DATA_ROOT / f"Audio{vid_id}" / f"Audio{vid_id}_chunks_{d}s"
        if not mp4_dir.exists() or not wav_dir.exists():
            print(f"  WARNING: missing chunks for video {vid_id} at {d}s — skipping")
            continue
        mp4s = sorted(mp4_dir.glob("*.mp4"))
        wavs = sorted(wav_dir.glob("*.wav"))
        if len(mp4s) != len(wavs):
            print(f"  WARNING: video {vid_id} has {len(mp4s)} mp4s but {len(wavs)} wavs — skipping")
            continue
        mp4_all.extend(mp4s)
        wav_all.extend(wavs)

    if SCRAMBLE_AV:
        # Move 3: break correct A-V temporal binding. Permute the GLOBAL audio
        # segment order (across all 18 videos, not within-video only) so each
        # video segment is paired with a random OTHER segment's audio, fixed
        # seed for reproducibility. A permutation with no fixed points (a
        # derangement) is not required -- occasional self-pairing is fine and
        # expected under a uniform random permutation at this N.
        rng = np.random.default_rng(SCRAMBLE_SEED)
        perm = rng.permutation(len(wav_all))
        n_fixed = int((perm == np.arange(len(wav_all))).sum())
        print(f"  [SCRAMBLE_AV] Permuted {len(wav_all)} audio segments "
              f"(seed={SCRAMBLE_SEED}, {n_fixed} incidental self-pairs)")
        wav_all = [wav_all[i] for i in perm]

    return list(zip(mp4_all, wav_all))


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(wav_path: Path, target_length: int) -> torch.Tensor:
    """Load wav → mono 16 kHz → Kaldi fbank → normalise → pad/truncate."""
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != CLIP_SR:
        wav = torchaudio.functional.resample(wav, sr, CLIP_SR)
    fbank = torchaudio.compliance.kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=CLIP_SR,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )  # (T, 128)
    fbank = (fbank - AUDIO_MEAN) / (AUDIO_STD * 2)
    T = fbank.shape[0]
    if T < target_length:
        fbank = F.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length]
    return fbank  # (target_length, 128)


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------
def load_video(mp4_path: Path) -> torch.Tensor:
    """
    Sample TOTAL_FRAMES uniformly from the mp4.
    Returns (TOTAL_FRAMES, 3, 224, 224) float32 in [0, 1].
    Falls back to OpenCV if torchvision fails.
    """
    try:
        from torchvision.io import read_video
        vid, _, _ = read_video(str(mp4_path), pts_unit="sec", output_format="TCHW")
        if vid.shape[0] == 0:
            raise ValueError("empty video")
        indices = np.linspace(0, vid.shape[0] - 1, TOTAL_FRAMES, dtype=int)
        frames  = vid[indices].float() / 255.0
        if frames.shape[-2:] != (224, 224):
            frames = F.interpolate(frames, size=(224, 224),
                                   mode="bilinear", align_corners=False)
        return frames
    except Exception:
        pass

    try:
        import cv2
        cap   = cv2.VideoCapture(str(mp4_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            raise ValueError("no frames")
        indices = np.linspace(0, total - 1, TOTAL_FRAMES, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
            else:
                frames.append(torch.zeros(3, 224, 224))
        cap.release()
        return torch.stack(frames)
    except Exception as e:
        print(f"    WARNING: video decode failed ({mp4_path.name}): {e}")
        return torch.zeros(TOTAL_FRAMES, 3, 224, 224)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _resize_pos_embed(sd: dict, model_sd: dict, key: str) -> None:
    if key not in sd or key not in model_sd:
        return
    src, dst = sd[key], model_sd[key]
    if src.shape == dst.shape:
        return
    src_t = src.permute(0, 2, 1).float()
    src_t = F.interpolate(src_t, size=dst.shape[1], mode="linear", align_corners=False)
    sd[key] = src_t.permute(0, 2, 1).to(src.dtype)
    print(f"  Resized {key}: {tuple(src.shape)} → {tuple(sd[key].shape)}")


def load_model(target_length: int):
    state = torch.load(MODEL_PATH, map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    n_reg = (state["register_tokens"].shape[0] // 2
             if "register_tokens" in state else 4)

    model = models.CAVMAESync(
        audio_length=target_length,
        modality_specific_depth=11,
        num_register_tokens=n_reg,
        cls_token=True,
        total_frame=TOTAL_FRAMES,
    )

    for key in ("pos_embed_a", "pos_embed_v", "decoder_pos_embed_a", "decoder_pos_embed_v"):
        _resize_pos_embed(state, model.state_dict(), key)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing  : {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  Unexpected: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract(seg_sec: float) -> None:
    out_dir = OUTPUT_ROOT / f"bin{int(seg_sec)}s_skip{int(seg_sec)}s"
    out_a   = out_dir / f"{_MODEL_NAME}_a.npy"
    out_v   = out_dir / f"{_MODEL_NAME}_v.npy"
    out_av  = out_dir / f"{_MODEL_NAME}_av.npy"

    if out_a.exists() and out_v.exists() and out_av.exists():
        shape = np.load(out_a).shape
        print(f"\n{int(seg_sec)}s — already done ({shape}), skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    target_length = TARGET_LENGTHS[seg_sec]

    print(f"\n{'='*65}")
    print(f"  {int(seg_sec)}s segments  (target_length={target_length})")
    print(f"{'='*65}")

    pairs = gather_pairs(seg_sec)
    print(f"  Total pairs: {len(pairs)}")
    if not pairs:
        print("  Nothing to extract.")
        return

    print(f"  Loading model ...")
    model = load_model(target_length)
    print(f"  Model loaded.")

    all_a, all_v, all_av = [], [], []
    n_err = 0

    for batch_start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[batch_start : batch_start + BATCH_SIZE]
        audio_list, video_list = [], []

        for mp4_path, wav_path in batch:
            audio_list.append(load_audio(wav_path, target_length))
            video_list.append(load_video(mp4_path))

        try:
            n = len(batch)
            audio_batch = torch.stack(audio_list).to(DEVICE)   # (n, T, 128)
            video_batch = torch.stack(video_list).to(DEVICE)   # (n, F, 3, 224, 224)

            audio_rep  = (audio_batch.unsqueeze(1)
                          .expand(-1, TOTAL_FRAMES, -1, -1)
                          .reshape(n * TOTAL_FRAMES, target_length, 128))
            video_flat = video_batch.view(n * TOTAL_FRAMES, 3, 224, 224)

            core = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                tokens_a, tokens_v = core.forward_feat(audio_rep, video_flat)[:2]

            cls_a  = tokens_a[:, 0, :].view(n, TOTAL_FRAMES, -1).mean(1)
            cls_v  = tokens_v[:, 0, :].view(n, TOTAL_FRAMES, -1).mean(1)
            cls_av = torch.cat([cls_a, cls_v], dim=-1)

            all_a.append(cls_a.detach().cpu().float().numpy())
            all_v.append(cls_v.detach().cpu().float().numpy())
            all_av.append(cls_av.detach().cpu().float().numpy())

        except Exception as exc:
            print(f"  ERROR batch@{batch_start}: {exc}")
            n_err += 1
            continue

        done = batch_start + len(batch)
        print(f"  [{done:4d}/{len(pairs)}]", end="\r", flush=True)

    print()
    if n_err:
        print(f"  Batches with errors: {n_err}")

    if not all_a:
        print("  No embeddings extracted.")
        return

    emb_a  = np.concatenate(all_a,  axis=0).astype(np.float32)
    emb_v  = np.concatenate(all_v,  axis=0).astype(np.float32)
    emb_av = np.concatenate(all_av, axis=0).astype(np.float32)

    np.save(out_a,  emb_a)
    np.save(out_v,  emb_v)
    np.save(out_av, emb_av)

    print(f"\n  Saved to {out_dir}/")
    print(f"    _a : {emb_a.shape}")
    print(f"    _v : {emb_v.shape}")
    print(f"    _av: {emb_av.shape}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for seg_sec in SEGMENT_DURATIONS:
        extract(seg_sec)
    print("\nDone.")
