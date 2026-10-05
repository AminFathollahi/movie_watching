"""
notebooks/feature_extraction/whisper_transcribe_bin2s.py
==========================================================
Standalone bin2s_skip2s Whisper transcription, extracted from text.ipynb's
Whisper+VAD cells (that notebook only loops SEGMENT_DURATIONS = [(5,5),(10,10)]).
Needed so rsa/topoomni_sheet_localizer.py's whisper_speech_proxy (word count per
bin, see whisper_speech_proxy_a.npy) can be computed at bin2s.

Run: conda run -n avtransformer python notebooks/feature_extraction/whisper_transcribe_bin2s.py
"""
import gc
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from natsort import natsorted
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA, OUTPUTS, MODELS  # noqa: E402

WHISPER_LOCAL_PATH = str(MODELS / "manual/models--openai--whisper-large-v3")
WHISPER_BATCH_SIZE = 16
USE_VAD = True
BIN_SEC = SKIP_SEC = 2.0

DATA_BASE = DATA / "segmented_stimulus/filtered"
TEXT_OUT = OUTPUTS / "model_embeddings/text"
PROXY_OUT = OUTPUTS / "model_embeddings/whisper_speech_proxy"


def find_chunk_pairs(data_base, bin_sec, skip_sec):
    dur_int, skip_int = int(bin_sec), int(skip_sec)
    suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    pairs = []
    for vid_subdir in natsorted(data_base.glob("Video*")):
        if not re.search(r"Video(\d+)$", vid_subdir.name):
            continue
        chunk_dir = vid_subdir / f"{vid_subdir.name}{suffix}"
        if not chunk_dir.exists():
            continue
        for av in natsorted(chunk_dir.glob("*_part_*.mp4")):
            pairs.append(av)
    return pairs


def main():
    out_dir = TEXT_OUT / f"bin{int(BIN_SEC)}s_skip{int(SKIP_SEC)}s"
    out_path = out_dir / "transcripts.json"
    if out_path.exists():
        print(f"{out_path} already exists, skipping transcription")
    else:
        pairs = find_chunk_pairs(DATA_BASE, BIN_SEC, SKIP_SEC)
        print(f"bin{int(BIN_SEC)}s_skip{int(SKIP_SEC)}s -> {len(pairs)} segments")
        assert pairs, "no chunk pairs found for bin2s"

        print("Loading Whisper ...")
        whisper_proc = WhisperProcessor.from_pretrained(WHISPER_LOCAL_PATH, local_files_only=True)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_LOCAL_PATH, local_files_only=True, torch_dtype=torch.float16,
        ).to(DEVICE).eval()
        forced_decoder_ids = whisper_proc.get_decoder_prompt_ids(language="en", task="transcribe")

        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, onnx=False, trust_repo=True,
        )
        get_speech_ts, _, vad_read_audio, _, _ = vad_utils
        vad_model = vad_model.to(DEVICE)

        def has_speech(wav_path):
            wav = vad_read_audio(str(wav_path), sampling_rate=16000).to(DEVICE)
            return len(get_speech_ts(wav, vad_model, sampling_rate=16000,
                                      threshold=0.5, min_silence_duration_ms=300)) > 0

        @torch.no_grad()
        def generate_transcripts(aud_paths, batch_size):
            if USE_VAD:
                print("  Running Silero VAD ...", end=" ")
                speech_flags = [has_speech(p) for p in aud_paths]
                print(f"{sum(speech_flags)}/{len(aud_paths)} segments have speech")
            else:
                speech_flags = [True] * len(aud_paths)

            results = []
            cur_size = batch_size
            i = 0
            while i < len(aud_paths):
                if not speech_flags[i]:
                    results.append("")
                    i += 1
                    continue
                batch, j = [], i
                while j < len(aud_paths) and len(batch) < cur_size and speech_flags[j]:
                    batch.append(aud_paths[j])
                    j += 1
                try:
                    audios = [librosa.load(p, sr=16000, mono=True)[0] for p in batch]
                    inputs = whisper_proc(audios, sampling_rate=16000, return_tensors="pt", padding=True)
                    feats = inputs.input_features.to(DEVICE, dtype=torch.float16)
                    ids = whisper_model.generate(feats, forced_decoder_ids=forced_decoder_ids)
                    texts = whisper_proc.batch_decode(ids, skip_special_tokens=True)
                    results.extend(t.strip() for t in texts)
                    i += len(batch)
                    print(f"    {i}/{len(aud_paths)} transcribed (batch={cur_size})", end="\r")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); gc.collect()
                    if cur_size == 1:
                        raise
                    cur_size = max(1, cur_size // 2)
                    print(f"\n    OOM -- reducing batch size to {cur_size}")
            print()
            return results

        transcripts = generate_transcripts(pairs, WHISPER_BATCH_SIZE)
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        out_path.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2))
        print(f"Saved: {out_path} ({len(transcripts)} entries)")

        del whisper_model, whisper_proc, vad_model
        torch.cuda.empty_cache(); gc.collect()

    # -- word-count proxy (matches whisper_speech_proxy_a.npy convention at bin5s/10s) --
    import json
    transcripts = json.loads(out_path.read_text())
    word_counts = np.array([[len(s.split()) if s else 0] for s in transcripts], dtype=np.float32)
    proxy_dir = PROXY_OUT / f"bin{int(BIN_SEC)}s_skip{int(SKIP_SEC)}s"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = proxy_dir / "whisper_speech_proxy_a.npy"
    np.save(proxy_path, word_counts)
    print(f"Saved: {proxy_path}  shape={word_counts.shape}")


if __name__ == "__main__":
    main()
