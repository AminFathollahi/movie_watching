"""Fuse concept captions + Whisper transcripts into clean per-segment event descriptions.

Uses Claude Haiku to merge the two sources into one sentence per segment,
following the PE-AV data engine approach. Strict no-hallucination rule: only information
present in the inputs is allowed in the output.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python rewrite_text_inputs.py

Reads:
    CONCEPT_EXCEL              — hand-labelled concept captions (5 s segments only)
    text/bin5s_skip5s/transcripts.json  — VAD-filtered Whisper transcripts

Writes:
    text/bin5s_skip5s/event_descriptions.json   — one merged sentence per segment
"""
import json
import os
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import pandas as pd
except ImportError:
    pd = None

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS  # noqa: E402

CONCEPT_EXCEL = Path(
    str(DATA / "Video_Audio_Pair_Concepts_All.xlsx")
)
TEXT_BASE = Path(
    str(OUTPUTS / "model_embeddings/text")
)

# Set False to skip LLM and concatenate caption + transcript directly
USE_LLM = False

REWRITE_MODEL      = "claude-haiku-4-5-20251001"
REWRITE_RATE_DELAY = 0.1   # seconds between API calls; raise if rate-limited

_SYSTEM_TWO = """\
You are an audiovisual caption editor. You receive two inputs about a short video clip:
  Caption   : a concise concept description of what is happening
  Transcript: words spoken, if any (from Whisper speech-to-text — may be empty or noisy)

Your task: write a clean description that captures the event, incorporating
any relevant spoken content when present.

Rules (follow exactly):
1. Include ONLY information present in the inputs — do not add, infer, or hallucinate.
2. If the transcript is empty, looks like noise, or consists of repetitive filler phrases,
   discard it and use only the caption.
3. If the transcript contains real speech relevant to the scene, incorporate its gist.
4. Present tense, third-person, factual. No hedging."""


def _is_hallucinated_transcript(transcript: str) -> bool:
    t = transcript.strip()
    if not t:
        return False
    words = t.split()
    if len(words) >= 9:
        phrases = [" ".join(words[i : i + 3]) for i in range(0, len(words) - 2, 3)]
        if len(set(phrases)) <= 2:
            return True
    return False


def rewrite_one(client, caption: str, transcript: str) -> str:
    trn = transcript.strip()
    if _is_hallucinated_transcript(trn):
        trn = ""
    trn_display = trn if trn else "(no speech)"

    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=120,
        system=_SYSTEM_TWO,
        messages=[{"role": "user", "content": (
            f"Caption: {caption}\n"
            f"Transcript: {trn_display}\n\n"
            "Write one clean sentence."
        )}],
    )
    return resp.content[0].text.strip()


def _format_no_llm(caption: str, transcript: str) -> str:
    trn = transcript.strip()
    if _is_hallucinated_transcript(trn):
        trn = ""
    return f"caption:{caption.strip()} \n transcript:{trn}"


def main() -> None:
    if pd is None:
        raise SystemExit("pandas not installed. Run: pip install pandas openpyxl")

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
        print("USE_LLM=False — concatenating caption + transcript directly (no API calls)\n")

    # Load and sort concepts (5 s segments only)
    df = pd.read_excel(CONCEPT_EXCEL)
    df = df.sort_values(["Series", "Part"]).reset_index(drop=True)
    captions = df["Clean concept"].tolist()
    print(f"Concepts loaded: {len(captions)} segments from {CONCEPT_EXCEL.name}")

    # Only 5 s bin — the Excel describes 5 s chunks
    bin_dir   = TEXT_BASE / "bin5s_skip5s"
    out_path  = bin_dir / "event_descriptions.json"

    if out_path.exists():
        print(f"bin5s_skip5s -- event_descriptions.json already exists, skipping")
        return

    trn_path = bin_dir / "transcripts.json"
    if not trn_path.exists():
        raise FileNotFoundError(
            f"{trn_path}\nRun text.ipynb to generate transcripts.json first."
        )

    transcripts = json.loads(trn_path.read_text())

    if len(captions) != len(transcripts):
        raise ValueError(
            f"Length mismatch: {len(captions)} concepts vs {len(transcripts)} transcripts. "
            "Check that the Excel covers the same 5 s segments as transcripts.json."
        )

    print(f"bin5s_skip5s -- merging {len(captions)} segments "
          f"({'LLM' if USE_LLM else 'concat'} mode) ...")

    events = []
    for i, (cap, trn) in enumerate(zip(captions, transcripts)):
        if USE_LLM:
            events.append(rewrite_one(client, cap, trn))
            if REWRITE_RATE_DELAY:
                time.sleep(REWRITE_RATE_DELAY)
        else:
            events.append(_format_no_llm(cap, trn))
        if (i + 1) % 50 == 0 or (i + 1) == len(captions):
            print(f"  {i+1}/{len(captions)}", end="\r")
    print(f"  {len(captions)}/{len(captions)} done")

    out_path.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    print(f"  Saved: {out_path}\n")

    import random
    print("  Sample events:")
    for idx in random.sample(range(len(events)), min(3, len(events))):
        print(f"\n  [{idx}]")
        print(f"  CAPTION : {captions[idx][:90]}")
        print(f"  TRANS   : {transcripts[idx][:70]}")
        print(f"  → EVENT : {events[idx]}")
    print()


if __name__ == "__main__":
    main()
