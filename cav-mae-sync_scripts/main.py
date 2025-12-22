"""
Pipeline driver:
1) Extract mono 16 kHz audio with ffmpeg.
2) Run embedding.py to compute per-segment audio/video embeddings.
3) Run segment_selection.py to build similarity plots and significance flags.

Usage example:
python main.py --videos /path/to/video1.mp4 /path/to/video2.mp4 \
    --seg-sec 4 --total-frames 16 --model-path cav-mae-sync/pretrained_models/cav_mae_sync.pth
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent


def run_ffmpeg(video_path: Path, ffmpeg_path: str = "ffmpeg") -> Path:
    """Extract mono 16 kHz audio; returns path to wav."""
    wav_path = video_path.with_suffix(".wav")
    cmd = [
        ffmpeg_path,
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(wav_path),
    ]
    print(f"[ffmpeg] {video_path.name} -> {wav_path.name}")
    subprocess.run(cmd, check=True)
    return wav_path


def run_cavmae(
    video_path: Path,
    outdir: Path,
    seg_sec: float,
    target_length: Optional[int],
    total_frames: int,
    num_register_tokens: Optional[int],
    model_path: Path,
    device: str,
):
    """Invoke embedding.py with provided args."""
    cavmae_script = SCRIPT_DIR / "embedding.py"
    cmd = [
        sys.executable,
        str(cavmae_script),
        "--video",
        str(video_path),
        "--outdir",
        str(outdir),
        "--seg_sec",
        str(seg_sec),
        "--total_frames",
        str(total_frames),
        "--model_path",
        str(model_path),
        "--device",
        device,
    ]
    if target_length is not None:
        cmd.extend(["--target_length", str(target_length)])
    if num_register_tokens is not None:
        cmd.extend(["--num_register_tokens", str(num_register_tokens)])

    print(f"[cavmae] {video_path.name} -> {outdir}")
    subprocess.run(cmd, check=True)


def run_segment_selection(outdir: Path, alpha: float):
    """Invoke segment_selection.py on a given embeddings directory."""
    segsel_script = SCRIPT_DIR / "segment_selection.py"
    cmd = [
        sys.executable,
        str(segsel_script),
        "--embeddings_dir",
        str(outdir),
        "--alpha",
        str(alpha),
    ]
    print(f"[segment_selection] {outdir}")
    subprocess.run(cmd, check=True)


def process_video(
    video_path: Path,
    outputs_root: Path,
    ffmpeg_path: str,
    seg_sec: float,
    target_length: int,
    total_frames: int,
    num_register_tokens: Optional[int],
    model_path: Path,
    alpha: float,
    device: str,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 1) Audio extraction
    run_ffmpeg(video_path, ffmpeg_path)

    # 2) Run cavmae; keep outputs under outputs_root/<video_stem>
    outdir = outputs_root / video_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    run_cavmae(
        video_path=video_path,
        outdir=outdir,
        seg_sec=seg_sec,
        target_length=target_length,
        total_frames=total_frames,
        num_register_tokens=num_register_tokens,
        model_path=model_path,
        device=device,
    )

    # 3) Run plotting + significance
    run_segment_selection(outdir, alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAV-MAE end-to-end pipeline.")
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="One or more video files to process.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=str(SCRIPT_DIR / "outputs_cavmae"),
        help="Root directory for outputs (each video gets its own subfolder).",
    )
    parser.add_argument(
        "--seg_sec",
        type=float,
        default=4.0,
        help="Segment length in seconds.",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=None,
        help="Override target_length for log-mel windows; omit/0 to auto (embedding.py default).",
    )
    parser.add_argument(
        "--total_frames",
        type=int,
        default=16,
        help="Frames sampled per segment.",
    )
    parser.add_argument(
        "--num_register_tokens",
        type=int,
        default=None,
        help="Number of register tokens (default: auto from checkpoint).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(SCRIPT_DIR / "cav-mae-sync" / "pretrained_models" / "cav_mae_sync.pth"),
        help="Path to cav_mae_sync.pth.",
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg binary.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for one-sided t-test (diag > off-diagonal).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Force device for embedding (default: cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    for vid in args.videos:
        video_path = Path(vid).expanduser().resolve()
        process_video(
            video_path=video_path,
            outputs_root=outputs_root,
            ffmpeg_path=args.ffmpeg_path,
            seg_sec=args.seg_sec,
            target_length=args.target_length if args.target_length != 0 else None,
            total_frames=args.total_frames,
            num_register_tokens=args.num_register_tokens,
            model_path=model_path,
            alpha=args.alpha,
            device=args.device,
        )


if __name__ == "__main__":
    main()
