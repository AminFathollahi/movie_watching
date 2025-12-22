#!/usr/bin/env python
"""
Extract fused, time-resolved MERLOT Reserve embeddings from an MP4.
- Splits video into fixed-length segments (SEG_SEC)
- For each segment: grabs the central frame + audio slice
- Runs MERLOT Reserve (video+audio) and returns one fused embedding per segment
- Saves: 
    embeddings.npy        (T x D)
    segments.csv          (start_s, end_s, center_s -> for alignment to fMRI TRs)
Best practices included (device, batching, deterministic-ish, logging).
"""

import os, sys, math, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Video I/O
import decord
decord.bridge.set_bridge('torch')
from decord import VideoReader, cpu

# Audio I/O
import librosa
import soundfile as sf

# ---- MERLOT Reserve imports (assumes repo is on PYTHONPATH) ----
# You may need to adjust these paths/classes if the repo API differs;
# repo usually exposes a model builder and config loader.
try:
    from merlot_reserve.models import builder as mr_builder
    from merlot_reserve.models.utils import load_state_dict
except ImportError as e:
    print("ERROR: Could not import MERLOT Reserve modules. "
          "Make sure merlot_reserve/ is on PYTHONPATH.\n"
          "   export PYTHONPATH=$PWD/merlot_reserve:$PYTHONPATH")
    raise e


# ------------- Config -------------
SEG_SEC = 5.0             # segment length in seconds (3–5s is typical)
FPS_FOR_FRAME = 8         # decode at modest fps; we’ll pick the central frame
AUDIO_SR = 16000          # target audio sample rate
N_MELS = 64               # mel bins for audio encoder (adjust to model’s config if needed)
HOP_LENGTH = 320          # 20ms @ 16kHz, STFT hop
N_FFT = 1024

BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

if DEVICE != 'cuda':
    raise RuntimeError("CUDA GPU not available. This script is intended to run on GPU.")

# Paths
CKPT_PATH = "checkpoints/merlot_reserve_base.ckpt"   # <-- put your checkpoint here
VIDEO_PATH = "movie.mp4"                              # <-- your input movie
OUT_DIR = "outputs_merlot"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ------------- Utilities -------------
def load_audio(path, target_sr=AUDIO_SR):
    wav, sr = librosa.load(path, sr=None, mono=True)  # read as mono
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')
        sr = target_sr
    return wav, sr

def video_duration_seconds(path):
    # decord reads length in frames; but we need duration ⇒ use ffprobe-like estimate via decord
    vr = VideoReader(path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames = len(vr)
    return num_frames / float(fps), fps

def sample_central_frame(vr, t_center, fps_decode=FPS_FOR_FRAME):
    """
    Decode a short window around t_center at fps_decode and take the central frame.
    If the decoder is slow, you can compute index directly from source fps.
    """
    src_fps = vr.get_avg_fps()
    idx = int(t_center * src_fps)
    idx = max(0, min(idx, len(vr)-1))
    frame = vr[idx].permute(2,0,1)  # HWC -> CHW, torch
    return frame  # uint8 CHW

def slice_audio(wav, sr, t_start, t_end):
    s = int(t_start * sr)
    e = int(t_end   * sr)
    s = max(0, min(s, wav.shape[0]-1))
    e = max(s+1, min(e, wav.shape[0]))
    return wav[s:e]

def wav_to_logmel(wav, sr=AUDIO_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Returns log-mel spectrogram [n_mels, time]
    """
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    logS = librosa.power_to_db(S + 1e-10)
    return logS.astype(np.float32)

def preprocess_frame_torch(frame_chw_uint8, image_size=224):
    """
    Basic center-crop + resize to match ViT expectations; normalize to [0,1].
    Adjust transforms to match MERLOT’s vision encoder preprocessing if needed.
    """
    import torchvision.transforms as T
    c, h, w = frame_chw_uint8.shape
    img = frame_chw_uint8.float() / 255.0
    transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        # Optional normalization if the ViT expects ImageNet stats; uncomment if needed:
        # T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return transform(img)  # CHW float32

def pad_or_truncate_time_axis(mel, target_T):
    """
    mel: [n_mels, T]
    returns [n_mels, target_T]
    """
    n_mels, T = mel.shape
    if T == target_T:
        return mel
    if T > target_T:
        return mel[:, :target_T]
    # pad
    out = np.zeros((n_mels, target_T), dtype=np.float32)
    out[:, :T] = mel
    return out

def collate_segments(frames, mels):
    """
    frames: list of CHW torch tensors (float32 [0,1])
    mels: list of np arrays [n_mels, Tmel]
    Return tensors ready for model
    """
    # Stack frames => [B, 3, H, W]
    x_img = torch.stack(frames, dim=0)  # float32
    # Stack audio => [B, n_mels, Tmel]
    max_T = max(m.shape[1] for m in mels)
    mels_pad = [pad_or_truncate_time_axis(m, max_T) for m in mels]
    x_mel = torch.from_numpy(np.stack(mels_pad, axis=0))  # float32
    return x_img, x_mel


# ------------- MERLOT Model Wrapper -------------
class MerlotWrapper(torch.nn.Module):
    """
    Thin wrapper to:
     - build MERLOT Reserve base
     - load checkpoint
     - run forward on (frame, mel) to get fused segment embeddings
    NOTE: Adjust code paths and keys to match the actual repo checkpoint/state dict.
    """
    def __init__(self, ckpt_path: str, device: str = DEVICE, dtype: torch.dtype = TORCH_DTYPE):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Build model from repo's builder; choose base config
        # The exact API may differ; adapt according to the repo you cloned.
        self.model = mr_builder.build_model(model_name='merlot_reserve_base')  # example name
        state = torch.load(ckpt_path, map_location='cpu')
        # Some repos store under 'state_dict' or require key remapping:
        if 'state_dict' in state:
            state = state['state_dict']
        load_state_dict(self.model, state, strict=False)

        self.model.eval().to(self.device)
        if dtype == torch.float16 and self.device.type == 'cuda':
            self.model = self.model.half()

    @torch.no_grad()
    def encode_batch(self, x_img, x_mel):
        """
        x_img: [B, 3, H, W] float32 in [0,1]
        x_mel: [B, n_mels, Tmel] float32 (log-mel)
        Returns: fused embeddings [B, D]
        """
        x_img = x_img.to(self.device, dtype=self.dtype)
        x_mel = x_mel.to(self.device, dtype=self.dtype)

        # The repo usually exposes a forward that accepts dicts like:
        #   model(video=..., audio=..., text=None, mask_positions=...)
        # and returns per-segment fused outputs or token sequences.
        # We aim to get a single fused vector per segment (e.g., mask/CLS token).
        out = self.model.forward_video_audio(
            video=x_img,          # [B, 3, H, W]
            audio=x_mel,          # [B, n_mels, T]
            return_fused=True     # ask for fused summary per segment (repo-specific)
        )
        # If your repo returns a dict, adjust:
        if isinstance(out, dict):
            if 'fused' in out:
                fused = out['fused']  # [B, D]
            elif 'joint_repr' in out:
                fused = out['joint_repr']
            else:
                # Fallback: mean pool last joint tokens
                fused = out['joint_tokens'].mean(dim=1)
        else:
            fused = out  # assume it already is [B, D]

        fused = F.normalize(fused.float(), dim=-1)  # optional L2 normalize
        return fused.cpu()


# ------------- Main -------------
def main():
    torch.backends.cudnn.benchmark = True

    # 1) Probe movie
    duration_s, src_fps = video_duration_seconds(VIDEO_PATH)
    print(f"Video duration: {duration_s:.2f}s @ {src_fps:.2f} fps")

    # 2) Load entire audio waveform (fast, single pass)
    wav, sr = load_audio(VIDEO_PATH, target_sr=AUDIO_SR)

    # 3) Set up segments
    num_segments = int(math.floor(duration_s / SEG_SEC))
    segments = []
    for i in range(num_segments):
        t0 = i * SEG_SEC
        t1 = (i+1) * SEG_SEC
        tc = 0.5*(t0 + t1)
        segments.append((t0, t1, tc))
    seg_df = pd.DataFrame(segments, columns=['start_s','end_s','center_s'])
    seg_df.to_csv(os.path.join(OUT_DIR, 'segments.csv'), index=False)

    # 4) Video reader
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    # 5) Build model
    model = MerlotWrapper(CKPT_PATH, device=DEVICE, dtype=TORCH_DTYPE)

    # 6) Iterate in batches
    all_embeds = []
    frames_buf, mels_buf, idx_buf = [], [], []

    for idx, (t0, t1, tc) in enumerate(tqdm(segments, desc="Segments")):
        # grab frame
        frame = sample_central_frame(vr, t_center=tc)          # uint8 CHW
        frame = preprocess_frame_torch(frame)                  # float CHW [0,1]

        # grab audio slice -> logmel
        wav_seg = slice_audio(wav, sr, t0, t1)
        mel = wav_to_logmel(wav_seg, sr=AUDIO_SR)              # [n_mels, T]

        frames_buf.append(frame)
        mels_buf.append(mel)
        idx_buf.append(idx)

        if len(frames_buf) == BATCH_SIZE or (idx == num_segments-1 and len(frames_buf)>0):
            x_img, x_mel = collate_segments(frames_buf, mels_buf)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(DEVICE=='cuda')):
                fused = model.encode_batch(x_img, x_mel)       # [B, D]

            all_embeds.append(fused.numpy())
            frames_buf, mels_buf, idx_buf = [], [], []

    if len(all_embeds) == 0:
        print("No embeddings computed (check duration/SEG_SEC)."); return
    E = np.vstack(all_embeds)  # [T, D]
    np.save(os.path.join(OUT_DIR, 'embeddings.npy'), E)
    print(f"Saved embeddings: {E.shape} → {os.path.join(OUT_DIR, 'embeddings.npy')}")
    print(f"Saved segments:   {os.path.join(OUT_DIR, 'segments.csv')}")
    print("Done.")

if __name__ == "__main__":
    main()
