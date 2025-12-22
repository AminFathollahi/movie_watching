"""
Minimal CAV-MAE Sync video pipeline.
Takes an mp4 video, runs the pretrained CAV-MAE Sync model in the cav-mae-sync
conda env, and writes per-segment audio and video embeddings plus
per-modality cosine RDMs. audio is extracted and resampled before running this
script using this:
ffmpeg -i 7T_MOVIE1_CC1_v2.mp4 -vn -ac 1 -ar 16000 {filename.wav}
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image


# export PYTHONPATH before running:
# PYTHONPATH="path/to/repo/cav-mae-sync/src:${PYTHONPATH}"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR.parent / "cav-mae-sync").resolve()
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import CAVMAESync  # type: ignore


DEFAULT_TOTAL_FRAMES = 16
DEFAULT_AUDIO_MEAN = -5.081
DEFAULT_AUDIO_STD = 4.4849
DEFAULT_SEG_SEC = 4.0
DEFAULT_MODEL_PATH = REPO_ROOT / "pretrained_models" / "cav_mae_sync.pth"
DEFAULT_OUTDIR = SCRIPT_DIR / "outputs_cavmae"


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine-distance RDM for segment embeddings."""
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    cos_sim = normed @ normed.T
    return 1.0 - cos_sim


def map_frame_to_spectrogram(frame_index: int, num_frames: int, spectrogram_length: int, target_length: int):
    """Mirror dataloader_sync mapping from frame idx to spectrogram window."""
    frame_position = int(round(frame_index * spectrogram_length / num_frames))
    start = max(0, frame_position - target_length // 2)
    end = start + target_length
    if end > spectrogram_length:
        end = spectrogram_length
        start = max(0, end - target_length)
    return start, end


def guess_target_length(seg_sec: float) -> int:
    """
    Heuristic to match repo defaults (multiples of 16 used in retrieval script).
    Falls back to rounding to the nearest multiple of 16 based on duration.
    """
    lookup = {
        2: 192,
        3: 304,
        4: 416,
        5: 512,
        6: 624,
        7: 720,
        10: 1024,
    }
    rounded = int(round(seg_sec))
    if rounded in lookup:
        return lookup[rounded]
    approx = seg_sec / 10.0 * 1024.0
    return int(math.ceil(max(approx, 1) / 16) * 16)


def load_waveform(audio_path: Path) -> Tuple[torch.Tensor, int, float]:
    """Load full audio once; return mono waveform, sample_rate, duration."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio from {audio_path} with torchaudio ({e}). "
            "Extract the audio to a WAV (e.g., ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 video.wav) and rerun."
        )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    num_samples = waveform.shape[1]
    duration = num_samples / sample_rate
    return waveform, sample_rate, duration


def slice_waveform(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float) -> torch.Tensor:
    """Slice preloaded waveform to [start, end) seconds."""
    start = int(start_s * sample_rate)
    end = int(end_s * sample_rate)
    start = max(start, 0)
    end = min(end, waveform.shape[1])
    if end <= start:
        return waveform[:, :1].clone()
    return waveform[:, start:end]


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int,
    mel_bins: int = 128,
) -> torch.Tensor:
    """Compute log-mel filterbank with repo defaults and pad/truncate to 1024 like dataloader_sync._wav2fbank."""
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    base_length = 1024  # matches dataloader_sync._wav2fbank
    if fbank.shape[0] < base_length:
        pad = base_length - fbank.shape[0]
        fbank = F.pad(fbank, (0, 0, 0, pad))
    elif fbank.shape[0] > base_length:
        fbank = fbank[:base_length]
    return fbank


def build_audio_windows(fbank: torch.Tensor, num_frames: int, target_length: int, audio_mean: float, audio_std: float):
    """Slice spectrogram into per-frame windows and normalize."""
    specs = []
    for frame_idx in range(num_frames):
        start, end = map_frame_to_spectrogram(frame_idx, num_frames, fbank.shape[0], target_length)
        window = fbank[start:end, :]
        if window.shape[0] < target_length:
            window = F.pad(window, (0, 0, 0, target_length - window.shape[0]))
        window = (window - audio_mean) / audio_std
        specs.append(window)
    return torch.stack(specs)


def load_video_frames(
    *,
    video_path: Path,
    start_s: float,
    end_s: float,
    num_frames: int,
    transform,
) -> torch.Tensor:
    """Decode frames from the mp4 and sample uniformly within [start_s, end_s)."""
    frames, _, _ = read_video(str(video_path), start_pts=start_s, end_pts=end_s, pts_unit="sec")
    if frames.numel() == 0:
        raise RuntimeError(f"No frames decoded between {start_s} and {end_s} seconds")
    indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).round().clamp(0, frames.shape[0] - 1).long()
    imgs = []
    for idx in indices:
        frame = frames[idx].numpy()
        img = Image.fromarray(frame)
        imgs.append(transform(img))
    return torch.stack(imgs)


def load_model(
    model_path: Path,
    device: torch.device,
    target_length: int,
    total_frames: int,
    num_register_tokens: Optional[int] = None,
):
    state = torch.load(model_path, map_location=device)
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}

    if num_register_tokens is None and "register_tokens" in cleaned:
        num_register_tokens = cleaned["register_tokens"].shape[0] // 2
        print(f"Auto-detected num_register_tokens={num_register_tokens} from checkpoint")

    # audio_length must match the spectrogram window length used for patches
    audio_length = target_length

    model = CAVMAESync(
        audio_length=audio_length,
        modality_specific_depth=11,
        num_register_tokens=num_register_tokens if num_register_tokens is not None else 4,
        cls_token=True,
        total_frame=total_frames,
    )
    msg = model.load_state_dict(cleaned, strict=False)
    print("Loaded model:", msg)
    model.to(device)
    model.eval()
    return model



def run_segment(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    start_s: float,
    end_s: float,
    target_length: int,
    total_frames: int,
    video_path: Path,
    device: torch.device,
    transform,
):
    """Compute per-modality embeddings for a [start,end) segment."""
    segment_waveform = slice_waveform(waveform, sample_rate, start_s, end_s)
    fbank = waveform_to_logmel(segment_waveform, sample_rate)
    audio_windows = build_audio_windows(fbank, total_frames, target_length, DEFAULT_AUDIO_MEAN, DEFAULT_AUDIO_STD)
    video_frames = load_video_frames(
        start_s=start_s,
        end_s=end_s,
        num_frames=total_frames,
        transform=transform,
        video_path=video_path,
    )

    audio_in = audio_windows.to(device)
    video_in = video_frames.to(device)
    with torch.no_grad():
        _, _, cls_a, cls_v = model.forward_feat(audio_in, video_in)

    if cls_a.ndim == 1:
        cls_a = cls_a.unsqueeze(0)
    if cls_v.ndim == 1:
        cls_v = cls_v.unsqueeze(0)

    audio_embed = cls_a.view(total_frames, -1).mean(dim=0)
    video_embed = cls_v.view(total_frames, -1).mean(dim=0)
    return (
        F.normalize(audio_embed, dim=0).cpu().numpy().astype(np.float32),
        F.normalize(video_embed, dim=0).cpu().numpy().astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to mp4 file")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH), help="Path to cav_mae_sync.pth")
    parser.add_argument("--seg_sec", type=float, default=DEFAULT_SEG_SEC, help="Segment length in seconds")
    parser.add_argument("--target_length", type=int, default=None, help="Override target_length for log-mel windows")
    parser.add_argument("--total_frames", type=int, default=DEFAULT_TOTAL_FRAMES, help="Frames sampled per segment")
    parser.add_argument("--num_register_tokens", type=int, default=None, help="Number of register tokens to use (default: auto from checkpoint)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Force device (default cpu).")
    args = parser.parse_args()

    audio_path = Path(args.video).with_suffix(".wav").expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.target_length is None or args.target_length == 0:
        target_length = guess_target_length(args.seg_sec)
    else:
        target_length = args.target_length

    print(f"Target log-mel length: {target_length}")

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Requested cuda but CUDA is not available; falling back to cpu.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    model = load_model(Path(args.model_path), device, target_length, args.total_frames, args.num_register_tokens)

    waveform, sample_rate, duration = load_waveform(audio_path)
    print(f"Loaded audio (sr={sample_rate}) duration={duration:.2f}s")

    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    segments = []
    audio_embeddings = []
    video_embeddings = []
    num_segments = int(math.ceil(duration / args.seg_sec))
    for idx in range(num_segments):
        start_s = idx * args.seg_sec
        end_s = min(duration, (idx + 1) * args.seg_sec)
        if end_s <= start_s:
            continue
        print(f"Processing segment {idx + 1}/{num_segments}: {start_s:.2f}s-{end_s:.2f}s")
        audio_embed, video_embed = run_segment(
            model=model,
            waveform=waveform,
            sample_rate=sample_rate,
            start_s=start_s,
            end_s=end_s,
            target_length=target_length,
            total_frames=args.total_frames,
            video_path=video_path,
            device=device,
            transform=transform,
        )
        segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "center_s": 0.5 * (start_s + end_s),
            }
        )
        audio_embeddings.append(audio_embed)
        video_embeddings.append(video_embed)

    if not audio_embeddings:
        raise RuntimeError("No embeddings were produced; check video path and decoding support.")

    audio_arr = np.stack(audio_embeddings, axis=0)
    video_arr = np.stack(video_embeddings, axis=0)

    np.save(outdir / "embeddings_audio.npy", audio_arr)
    np.save(outdir / "embeddings_video.npy", video_arr)
    print("Saved embeddings_audio.npy with shape", audio_arr.shape)
    print("Saved embeddings_video.npy with shape", video_arr.shape)

    import csv

    with open(outdir / "segments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_s", "end_s", "center_s"])
        writer.writeheader()
        writer.writerows(segments)

    rdm_audio = compute_rdm(audio_arr)
    rdm_video = compute_rdm(video_arr)
    np.save(outdir / "rdm_audio_cosine.npy", rdm_audio.astype(np.float32))
    np.save(outdir / "rdm_video_cosine.npy", rdm_video.astype(np.float32))
    print("Saved rdm_audio_cosine.npy and rdm_video_cosine.npy")


if __name__ == "__main__":
    main()
