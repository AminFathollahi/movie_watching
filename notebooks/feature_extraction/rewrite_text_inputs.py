"""Fuse video captions + audio captions + Whisper transcripts into clean AV descriptions.

Uses Claude Haiku to merge three noisy caption sources into one sentence per segment,
following the PE-AV data engine approach (paper Figure 3/6). Strict no-hallucination
rule: only information present in the inputs is allowed in the output.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python rewrite_text_inputs.py

Reads:
    outputs/model_embeddings/text/{seg}s/captions.json
    outputs/model_embeddings/text/{seg}s/audio_captions.json
    outputs/model_embeddings/text/{seg}s/transcripts.json

Writes:
    outputs/model_embeddings/text/{seg}s/rewritten_text_inputs.json
"""
import json
import os
import time
from pathlib import Path

import anthropic

SEGMENT_DURATIONS = [5, 10]
TEXT_BASE = Path(
    "/home/amin/Research/Representation/Movie/outputs/model_embeddings/text"
)

REWRITE_MODEL      = "claude-haiku-4-5-20251001"
REWRITE_RATE_DELAY = 0.05   # seconds between API calls; raise if rate-limited

_SYSTEM = """\
You are an audiovisual caption editor. You receive three inputs about a short video clip:
  Video caption : what is visually happening (from a video captioner)
  Audio caption : what sounds are present (from an audio captioner)
  Transcript    : words spoken, if any (from Whisper speech-to-text — may be empty or noisy)

Your task: write EXACTLY ONE clean sentence that captures both the visual scene and its
acoustic character.

Rules (follow exactly):
1. Include ONLY information present in the inputs — do not add, infer, or hallucinate.
2. If the transcript contains hallucinations (song lyrics not matching the scene, repetitive
   filler phrases, or random words when the audio caption indicates no speech), discard it.
3. If the transcript contains real speech relevant to the scene, incorporate its gist.
4. Merge the visual and acoustic perspectives naturally.
5. Present tense, third-person, factual. No hedging.
6. Output the sentence only — no preamble or explanation."""


def _is_hallucinated_transcript(transcript: str, audio_caption: str) -> bool:
    t = transcript.strip()
    if not t:
        return False
    words = t.split()
    # Repeated phrase pattern (e.g. "Thank you. Thank you. Thank you.")
    if len(words) >= 6:
        phrases = [" ".join(words[i : i + 3]) for i in range(0, len(words) - 2, 3)]
        if len(set(phrases)) <= 2:
            return True
    # Transcribed music lyrics when audio caption says instrumental/music
    ac_lower = audio_caption.lower()
    if any(kw in ac_lower for kw in ["music", "instrumental", "melody", "song", "tune"]):
        if len(words) < 20:
            return True
    return False


def rewrite_one(
    client: anthropic.Anthropic,
    video_cap: str,
    audio_cap: str,
    transcript: str,
) -> str:
    trn = transcript.strip()
    if _is_hallucinated_transcript(trn, audio_cap):
        trn = ""
    trn_display = trn if trn else "(no speech)"

    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=120,
        system=_SYSTEM,
        messages=[{"role": "user", "content": (
            f"Video caption: {video_cap}\n"
            f"Audio caption: {audio_cap}\n"
            f"Transcript: {trn_display}\n\n"
            "Write one clean audiovisual sentence."
        )}],
    )
    return resp.content[0].text.strip()


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)
    print(f"Model: {REWRITE_MODEL}\n")

    for seg_sec in SEGMENT_DURATIONS:
        out_dir = TEXT_BASE / f"{seg_sec}s"
        rw_path = out_dir / "rewritten_text_inputs.json"

        if rw_path.exists():
            print(f"{seg_sec}s -- rewritten_text_inputs.json already exists, skipping")
            continue

        cap_path = out_dir / "captions.json"
        ac_path  = out_dir / "audio_captions.json"
        trn_path = out_dir / "transcripts.json"

        missing = [p.name for p in [cap_path, ac_path, trn_path] if not p.exists()]
        if missing:
            print(f"{seg_sec}s -- MISSING: {missing}  (skipping)")
            continue

        captions       = json.loads(cap_path.read_text())
        audio_captions = json.loads(ac_path.read_text())
        transcripts    = json.loads(trn_path.read_text())

        assert len(captions) == len(audio_captions) == len(transcripts), (
            f"{seg_sec}s: length mismatch  "
            f"caps={len(captions)}  ac={len(audio_captions)}  trn={len(transcripts)}"
        )

        print(f"{seg_sec}s -- rewriting {len(captions)} segments ...")
        rewritten = []
        for i, (vc, ac, tr) in enumerate(zip(captions, audio_captions, transcripts)):
            rewritten.append(rewrite_one(client, vc, ac, tr))
            if REWRITE_RATE_DELAY:
                time.sleep(REWRITE_RATE_DELAY)
            if (i + 1) % 50 == 0 or (i + 1) == len(captions):
                print(f"  {i+1}/{len(captions)}", end="\r")
        print(f"  {len(captions)}/{len(captions)} done")

        rw_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2))
        print(f"  Saved: {rw_path}\n")

        # Print 3 samples
        import random
        print("  Sample rewrites:")
        for idx in random.sample(range(len(rewritten)), min(3, len(rewritten))):
            print(f"\n  [{idx}]")
            print(f"  VID  : {captions[idx][:90]}")
            print(f"  AUD  : {audio_captions[idx][:90]}")
            print(f"  TRANS: {transcripts[idx][:70]}")
            print(f"  → RW : {rewritten[idx]}")
        print()


if __name__ == "__main__":
    main()
