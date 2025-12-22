"""

Outputs fused (MASK-token) embeddings per 5s segment plus simple RDM.

after adding the directory of the repo to python path,and creating a conda env for the repo based on the requirements.txt for the repo, and activating the conda env; run as :
python merlot.py --video path/to/video.mp4 --outdir outdir --model large --seg_sec 5.0

"""

import argparse
from pathlib import Path

import csv
import numpy as np

from mreserve.modeling import PretrainedMerlotReserve
from mreserve.preprocess import MASK, preprocess_video, video_to_segments


DEFAULT_GRID = (18, 32)
MAX_SEGMENTS_PER_CALL = 8


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine-distance RDM for segment embeddings.
    Returns [T, T] matrix where entry (i,j) is 1 - cosine(emb_i, emb_j).
    """
    # Normalize to unit length and use dot product for cosine similarity.
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    cos_sim = normed @ normed.T
    return 1.0 - cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to mp4 file")
    parser.add_argument("--outdir", default="outputs_merlot", help="Output directory")
    parser.add_argument("--model", default="large", help="MERLOT model size (large/base)")
    parser.add_argument("--seg_sec", type=float, default=5.0, help="Segment length in seconds")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("1/3 Extracting segments with MERLOT Reserve...")
    video_segments = video_to_segments(args.video, time_interval=args.seg_sec, num_segments_max=None)
    for seg in video_segments:
        seg["use_text_as_input"] = False  # no text conditioning
    print(f"   Got {len(video_segments)} segments.")

    # Save segment timing for downstream RDM/RSA alignment (without pandas).
    seg_rows = [
        {
            "start_s": s["start_time"],
            "end_s": s["end_time"],
            "center_s": 0.5 * (s["start_time"] + s["end_time"]),
        }
        for s in video_segments
    ]
    with open(outdir / "segments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_s", "end_s", "center_s"])
        writer.writeheader()
        writer.writerows(seg_rows)

    print(f"2/3 Running MERLOT Reserve in <= {MAX_SEGMENTS_PER_CALL}-segment chunks...")
    model = PretrainedMerlotReserve.from_pretrained(
        model_name=args.model, image_grid_size=DEFAULT_GRID
    )
    fused_chunks = []
    num_chunks = (len(video_segments) + MAX_SEGMENTS_PER_CALL - 1) // MAX_SEGMENTS_PER_CALL
    for chunk_idx in range(num_chunks):
        chunk = video_segments[
            chunk_idx * MAX_SEGMENTS_PER_CALL : (chunk_idx + 1) * MAX_SEGMENTS_PER_CALL
        ]
        print(f"   Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} segments)")
        video_pre = preprocess_video(chunk, output_grid_size=DEFAULT_GRID, verbose=False)
        out_h = model.embed_video(**video_pre)
        fused_chunk = np.array(out_h[video_pre["tokens"] == MASK], dtype=np.float32)
        fused_chunks.append(fused_chunk)
    fused = np.concatenate(fused_chunks, axis=0)
    np.save(outdir / "embeddings.npy", fused)
    print("   Saved embeddings.npy with shape:", fused.shape)

    print("3/3 Computing RDM...")
    rdm = compute_rdm(fused)
    np.save(outdir / "rdm_cosine.npy", rdm.astype(np.float32))
    print("   Saved rdm_cosine.npy")


if __name__ == "__main__":
    main()
