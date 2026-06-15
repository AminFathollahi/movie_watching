"""Fuse video captions + Whisper transcripts (+ optional audio captions) into clean descriptions.

Uses Claude Haiku to merge two or three caption sources into one sentence per segment,
following the PE-AV data engine approach. Strict no-hallucination rule: only information
present in the inputs is allowed in the output.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python rewrite_text_inputs.py

Reads (from TEXT_BASE/bin{N}s_skip{N}s/):
    captions.json          — required  (InternVL3 visual captions)
    transcripts.json       — required  (VAD-filtered Whisper)
    audio_captions.json    — optional  (Gemma / AudioFlamingo audio captions)

Writes:
    rewritten_text_inputs.json
"""
import json
import os
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None

SEGMENT_DURATIONS = [(5, 5), (10, 10)]
TEXT_BASE = Path(
    "/home/amin/Research/Representation/Movie/outputs/model_embeddings/text"
)

# Set False to skip LLM and write "caption:...\ntranscript:..." directly (temp patch)
USE_LLM = False

REWRITE_MODEL      = "claude-haiku-4-5-20251001"
REWRITE_RATE_DELAY = 0.05   # seconds between API calls; raise if rate-limited

_SYSTEM_THREE = """\
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

_SYSTEM_TWO = """\
You are an audiovisual caption editor. You receive two inputs about a short video clip:
  Video caption : what is visually happening (from a video captioner)
  Transcript    : words spoken, if any (from Whisper speech-to-text — may be empty or noisy)

Your task: write EXACTLY ONE clean sentence that describes the visual scene and incorporates
any spoken content when relevant.

Rules (follow exactly):
1. Include ONLY information present in the inputs — do not add, infer, or hallucinate.
2. If the transcript is empty, looks like noise, or consists of repetitive filler phrases,
   discard it and describe only the visual scene.
3. If the transcript contains real speech relevant to the scene, incorporate its gist.
4. Present tense, third-person, factual. No hedging.
5. Output the sentence only — no preamble or explanation."""


def _is_hallucinated_transcript(transcript: str, audio_caption: str = "") -> bool:
    t = transcript.strip()
    if not t:
        return False
    words = t.split()
    # Repeated 3-word phrase pattern (e.g. "Thank you. Thank you. Thank you.")
    # Require at least 3 chunks to avoid false positives on short legitimate sentences.
    if len(words) >= 9:
        phrases = [" ".join(words[i : i + 3]) for i in range(0, len(words) - 2, 3)]
        if len(set(phrases)) <= 2:
            return True
    # Transcribed music lyrics when audio caption says instrumental/music
    if audio_caption:
        ac_lower = audio_caption.lower()
        if any(kw in ac_lower for kw in ["music", "instrumental", "melody", "song", "tune"]):
            if len(words) < 20:
                return True
    return False


def rewrite_one(
    client,
    video_cap: str,
    transcript: str,
    audio_cap: str = "",
) -> str:
    trn = transcript.strip()
    if _is_hallucinated_transcript(trn, audio_cap):
        trn = ""
    trn_display = trn if trn else "(no speech)"

    if audio_cap:
        system = _SYSTEM_THREE
        user_content = (
            f"Video caption: {video_cap}\n"
            f"Audio caption: {audio_cap}\n"
            f"Transcript: {trn_display}\n\n"
            "Write one clean audiovisual sentence."
        )
    else:
        system = _SYSTEM_TWO
        user_content = (
            f"Video caption: {video_cap}\n"
            f"Transcript: {trn_display}\n\n"
            "Write one clean sentence."
        )

    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=120,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text.strip()


def _format_no_llm(video_cap: str, transcript: str, audio_cap: str = "") -> str:
    trn = transcript.strip()
    if _is_hallucinated_transcript(trn, audio_cap):
        trn = ""
    parts = [f"caption: {video_cap}"]
    if audio_cap:
        parts.append(f"audio: {audio_cap}")
    parts.append(f"transcript: {trn}")
    return "\n".join(parts)


def main() -> None:
    if USE_LLM:
        if anthropic is None:
            raise SystemExit("anthropic package not installed. Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set.\n"
                "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Model: {REWRITE_MODEL}\n")
    else:
        client = None
        print("USE_LLM=False — writing caption+transcript strings directly (no API calls)\n")

    for bin_sec, skip_sec in SEGMENT_DURATIONS:
        dir_name = f"bin{bin_sec}s_skip{skip_sec}s"
        out_dir  = TEXT_BASE / dir_name
        rw_path  = out_dir / "rewritten_text_inputs.json"

        if rw_path.exists():
            print(f"{dir_name} -- rewritten_text_inputs.json already exists, skipping")
            continue

        cap_path = out_dir / "captions.json"
        trn_path = out_dir / "transcripts.json"
        ac_path  = out_dir / "audio_captions.json"

        missing = [p.name for p in [cap_path, trn_path] if not p.exists()]
        if missing:
            print(f"{dir_name} -- MISSING required: {missing}  (skipping)")
            continue

        captions    = json.loads(cap_path.read_text())
        transcripts = json.loads(trn_path.read_text())

        has_audio_caps = ac_path.exists()
        if has_audio_caps:
            audio_captions = json.loads(ac_path.read_text())
            assert len(captions) == len(audio_captions) == len(transcripts), (
                f"{dir_name}: length mismatch  caps={len(captions)}  "
                f"ac={len(audio_captions)}  trn={len(transcripts)}"
            )
            print(f"{dir_name} -- rewriting {len(captions)} segments "
                  f"(3-source: caption + audio + transcript) ...")
        else:
            audio_captions = [""] * len(captions)
            print(f"{dir_name} -- rewriting {len(captions)} segments "
                  f"(2-source: caption + transcript; no audio_captions.json) ...")

        assert len(captions) == len(transcripts), (
            f"{dir_name}: length mismatch  caps={len(captions)}  trn={len(transcripts)}"
        )

        rewritten = []
        for i, (vc, tr, ac) in enumerate(zip(captions, transcripts, audio_captions)):
            if USE_LLM:
                rewritten.append(rewrite_one(client, vc, tr, ac))
                if REWRITE_RATE_DELAY:
                    time.sleep(REWRITE_RATE_DELAY)
            else:
                rewritten.append(_format_no_llm(vc, tr, ac))
            if (i + 1) % 50 == 0 or (i + 1) == len(captions):
                print(f"  {i+1}/{len(captions)}", end="\r")
        print(f"  {len(captions)}/{len(captions)} done")

        rw_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2))
        print(f"  Saved: {rw_path}\n")

        import random
        print("  Sample rewrites:")
        for idx in random.sample(range(len(rewritten)), min(3, len(rewritten))):
            print(f"\n  [{idx}]")
            print(f"  VID  : {captions[idx][:90]}")
            if has_audio_caps:
                print(f"  AUD  : {audio_captions[idx][:90]}")
            print(f"  TRANS: {transcripts[idx][:70]}")
            print(f"  → RW : {rewritten[idx]}")
        print()


if __name__ == "__main__":
    main()
