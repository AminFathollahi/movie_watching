"""
notebooks/feature_extraction/extract_embeddings.py
===================================================
Unimodal encoder embeddings over the filtered stimulus chunks of the 18 clips, in float32.

    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/extract_embeddings.py --models dasheng-0.6b vjepa2-vitl --bins 5 2 1
    conda run --no-capture-output -n avtransformer \
        python notebooks/feature_extraction/extract_embeddings.py --verify

Writes {embeddings-dir}/{model}/bin{B}s_skip{B}s/{model}_{a|v}.npy, one row per chunk in clip order
then time order. Encoders with hidden states also write {model}-d50 and {model}-d75 (encoder layers at
50% and 75% of the depth, mean-pooled over tokens); the plain name is the final output. Each array has
a {model}_layers.json with the layer used.
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio
from natsort import natsorted
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import OUTPUTS, EXTERNAL, MODELS  # noqa: E402

STIMULUS_DIR = str(EXTERNAL / "data/segmented_stimulus/filtered")
EMBEDDINGS_DIR = str(OUTPUTS / "model_embeddings")
MODELS_HOME = str(MODELS)
N_FRAMES = 16
DECODER_KEYS = ("decoder.", "encoder_to_decoder.", "mask_token")
DEPTHS = {"-d50": 0.5, "-d75": 0.75, "": 1.0}


def chunk_paths(stimulus_dir, kind, ext, bin_sec):
    return [p for n in range(1, 19)
            for p in natsorted((Path(stimulus_dir) / f"{kind}{n}" / f"{kind}{n}_chunks_{bin_sec}s").glob(f"*.{ext}"))]


def load_video(path):
    import decord
    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    index = np.linspace(0, len(reader) - 1, N_FRAMES).astype(np.int64)
    return reader.get_batch(index).permute(0, 3, 1, 2)


def load_audio(path, rate, bin_sec):
    wav, native = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if native != rate:
        wav = torchaudio.functional.resample(wav, native, rate)
    return wav[:bin_sec * rate]


def pool(x, frames=None):
    return x[:, :frames].mean(dim=1)[0]


def depth_embeddings(states, frames=None, final=None):
    n = len(states)
    out = {suffix: pool(states[round(fraction * n) - 1], frames) for suffix, fraction in DEPTHS.items()}
    if final is not None:
        out[""] = final
    return out


def capture(layers, unwrap=lambda out: out[0] if isinstance(out, tuple) else out):
    states = []
    for layer in layers:
        layer.register_forward_hook(lambda module, args, out: states.append(unwrap(out)))
    return states


def source(home, spec):
    return str(home / spec) if spec.startswith("manual/") else spec


def wavlm_large(home, device):
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    path = next((home / "transformers/models--microsoft--wavlm-large/snapshots").iterdir())
    model = WavLMModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).to(device).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)

    def embed(p, bin_sec):
        inputs = extractor(load_audio(p, 16000, bin_sec).numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        return depth_embeddings(model(**inputs, output_hidden_states=True).hidden_states[1:])

    return embed, model.config.num_hidden_layers


def whisper_large_v3(home, device):
    from transformers import WhisperModel, WhisperProcessor
    path = home / "manual/models--openai--whisper-large-v3"
    encoder = WhisperModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).encoder.to(device).eval()
    processor = WhisperProcessor.from_pretrained(path, local_files_only=True)

    def embed(p, bin_sec):
        wav = load_audio(p, 16000, bin_sec)
        features = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        real = max(1, round(wav.numel() / 16000 * 50))
        return depth_embeddings(encoder(features.to(device), output_hidden_states=True).hidden_states[1:], frames=real)

    return embed, encoder.config.encoder_layers


def w2v_bert(home, device):
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
    extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True)
    model = Wav2Vec2BertModel.from_pretrained(
        "facebook/w2v-bert-2.0", local_files_only=True, dtype=torch.float32).to(device).eval()

    def embed(p, bin_sec):
        inputs = extractor(load_audio(p, 16000, bin_sec).numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        return depth_embeddings(model(**inputs, output_hidden_states=True).hidden_states[1:])

    return embed, model.config.num_hidden_layers


def clap_larger(home, device):
    from transformers import ClapFeatureExtractor, ClapModel
    path = home / "manual/models--laion--larger_clap_music_and_speech"
    model = ClapModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).to(device).eval()
    extractor = ClapFeatureExtractor.from_pretrained(path, local_files_only=True)
    rate = extractor.sampling_rate

    def embed(p, bin_sec):
        inputs = extractor(load_audio(p, rate, bin_sec).numpy(), sampling_rate=rate, return_tensors="pt").to(device)
        return {"": model.get_audio_features(**inputs).pooler_output[0]}

    return embed, None


def dasheng(repo):
    def load(home, device):
        from transformers import AutoFeatureExtractor, AutoModel
        extractor = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(
            repo, outputdim=None, trust_remote_code=True, local_files_only=True, dtype=torch.float32).to(device).eval()
        states = capture(model.encoder.blocks)

        def embed(p, bin_sec):
            states.clear()
            inputs = extractor(load_audio(p, 16000, bin_sec)[None], sampling_rate=16000, return_tensors="pt")
            final = pool(model(input_values=inputs["input_values"].to(device)).hidden_states)
            return depth_embeddings(states, final=final)

        return embed, len(model.encoder.blocks)

    return load


def openbeats_large_i2(home, device):
    import yaml
    from safetensors.torch import load_file
    code = home / "manual/OpenBEATs_code"
    checkpoint = home / "hub/models-shikhar7ssu--OpenBEATS-Large-i2-esc50f4"
    stubs = {
        "espnet2": {}, "espnet2.asr": {}, "espnet2.asr.encoder": {}, "espnet2.asr.specaug": {},
        "espnet2.legacy": {}, "espnet2.legacy.nets": {}, "espnet2.legacy.nets.pytorch_backend": {},
        "espnet2.torch_utils": {}, "espnet2.beats": {},
        "espnet2.asr.encoder.abs_encoder": {"AbsEncoder": torch.nn.Module},
        "espnet2.asr.specaug.specaug": {"SpecAug": None},
        "espnet2.legacy.nets.pytorch_backend.nets_utils": {"roll_tensor": None},
        "espnet2.torch_utils.safe_torch_load": {"safe_torch_load": None},
    }
    for name, attributes in stubs.items():
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__dict__.update(attributes)
    sys.modules["espnet2"].__path__ = [str(code / "espnet2")]
    sys.modules["espnet2.beats"].__path__ = [str(code / "espnet2/beats")]
    from espnet2.beats.encoder import BeatsEncoder
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())["encoder_conf"]["beats_config"]
    config = {k: v for k, v in config.items() if k not in ("fbank_mean", "fbank_std")}
    model = BeatsEncoder(input_size=1, beats_config=config, fbank_mean=15.2913, fbank_std=5.90532,
                         is_pretraining=False)
    model.load_state_dict(load_file(checkpoint / "model.safetensors"))
    model = model.float().to(device).eval()
    states = capture(model.encoder.layers, unwrap=lambda out: out[0].transpose(0, 1))

    def embed(p, bin_sec):
        states.clear()
        wav = load_audio(p, 16000, bin_sec)[None].to(device)
        hidden, lengths, _ = model(wav, torch.tensor([wav.shape[1]], device=device))
        return depth_embeddings(states, frames=int(lengths[0]), final=pool(hidden, int(lengths[0])))

    return embed, len(model.encoder.layers)


def videomaev2(spec):
    def load(home, device):
        from safetensors.torch import load_file
        from transformers import AutoImageProcessor
        path = home / spec
        package = types.ModuleType("videomaev2")
        package.__path__ = [str(path)]
        sys.modules["videomaev2"] = package
        from videomaev2.modeling_videomaev2 import VisionTransformer
        model = VisionTransformer(**json.loads((path / "config.json").read_text())["model_config"])
        state = {k.removeprefix("model."): v for k, v in load_file(path / "model.safetensors").items()}
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith(DECODER_KEYS)})
        model = model.float().to(device).eval()
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        final_norm = model.fc_norm if model.fc_norm is not None else model.norm
        states = capture(model.blocks)

        def embed(p, bin_sec):
            states.clear()
            pixels = processor(list(load_video(p)), return_tensors="pt")["pixel_values"].permute(0, 2, 1, 3, 4)
            model(pixels.to(device))
            return depth_embeddings(states, final=final_norm(pool(states[-1])))

        return embed, len(model.blocks)

    return load


def pe_core_l14(home, device):
    sys.path.append(str(home / "perception_models"))
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
    from torchvision.transforms.functional import to_pil_image
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    state = torch.load(home / "manual/PE-Core-L14-336/PE-Core-L14-336.pt", map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    model = model.float().to(device).eval()
    preprocess = transforms.get_image_transform(int(model.image_size))

    def embed(p, bin_sec):
        frames = torch.stack([preprocess(to_pil_image(f)) for f in load_video(p)]).to(device)
        return {"": model.visual(frames).mean(dim=0)}

    return embed, None


def vjepa2(spec):
    def load(home, device):
        from transformers import AutoVideoProcessor, VJEPA2Model
        path = source(home, spec)
        processor = AutoVideoProcessor.from_pretrained(path, local_files_only=True)
        encoder = VJEPA2Model.from_pretrained(path, local_files_only=True, dtype=torch.float32).encoder
        encoder = encoder.to(device).eval()
        states = capture(encoder.layer)

        def embed(p, bin_sec):
            states.clear()
            pixels = processor(load_video(p), return_tensors="pt")["pixel_values_videos"].to(device)
            return depth_embeddings(states, final=pool(encoder(pixels).last_hidden_state))

        return embed, len(encoder.layer)

    return load


REGISTRY = {
    "wavlm-large": ("a", wavlm_large, True),
    "whisper-large-v3": ("a", whisper_large_v3, True),
    "dasheng-0.6b": ("a", dasheng("mispeech/dasheng-0.6B"), True),
    "dasheng-1.2b": ("a", dasheng("mispeech/dasheng-1.2B"), True),
    "openbeats-large-i2": ("a", openbeats_large_i2, True),
    "w2v-bert-2.0": ("a", w2v_bert, True),
    "clap-larger": ("a", clap_larger, False),
    "videomaev2-large": ("v", videomaev2("manual/models--OpenGVLab--VideoMAEv2-Large"), True),
    "videomaev2-giant": ("v", videomaev2("manual/models--OpenGVLab--VideoMAEv2-Giant"), True),
    "pe-core-l14": ("v", pe_core_l14, False),
    "vjepa2-vitl": ("v", vjepa2("manual/vjepa2-vitl-fpc64-256"), True),
    "vjepa2-vitg": ("v", vjepa2("facebook/vjepa2-vitg-fpc64-256"), True),
}


def output_names(name):
    return [f"{name}{suffix}" for suffix in DEPTHS] if REGISTRY[name][2] else [name]


def output_path(embeddings_dir, output, modality, bin_sec, extension="npy"):
    stem = f"{output}_{modality}" if extension == "npy" else f"{output}_layers"
    return Path(embeddings_dir) / output / f"bin{bin_sec}s_skip{bin_sec}s" / f"{stem}.{extension}"


def save(args, name, modality, bin_sec, rows, n_layers):
    for suffix in (DEPTHS if n_layers else [""]):
        output = f"{name}{suffix}"
        array = np.stack(rows[suffix])
        assert np.isfinite(array).all(), f"{output}: non-finite values"
        path = output_path(args.embeddings_dir, output, modality, bin_sec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        info = {"model": name, "dtype": "float32"}
        if n_layers:
            info |= {"layer": round(DEPTHS[suffix] * n_layers), "n_layers": n_layers, "pooling": "mean over tokens"}
        output_path(args.embeddings_dir, output, modality, bin_sec, "json").write_text(json.dumps(info, indent=1))
        print(f"saved {path} shape={array.shape}", flush=True)


def extract(args, name):
    modality, load, _ = REGISTRY[name]
    todo = [b for b in args.bins if args.overwrite
            or not all(output_path(args.embeddings_dir, o, modality, b).exists() for o in output_names(name))]
    if not todo:
        print(f"{name}: all bins present, skipping", flush=True)
        return
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    embed, n_layers = load(Path(args.models_home), args.device)
    kind, ext = ("Video", "mp4") if modality == "v" else ("Audio", "wav")
    for bin_sec in todo:
        paths = chunk_paths(args.stimulus_dir, kind, ext, bin_sec)[:args.limit]
        assert paths, f"no {bin_sec}s chunks under {args.stimulus_dir}"
        rows = {suffix: [] for suffix in (DEPTHS if n_layers else [""])}
        start = time.time()
        with torch.inference_mode():
            for p in tqdm(paths, desc=f"{name} bin{bin_sec}s", mininterval=60):
                for suffix, vector in embed(p, bin_sec).items():
                    rows[suffix].append(vector.float().cpu().numpy())
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else float("nan")
        print(f"{name} bin{bin_sec}s: {len(paths)} chunks, {(time.time() - start) / len(paths):.3f} s/chunk, "
              f"peak GPU memory {peak:.2f} GiB", flush=True)
        save(args, name, modality, bin_sec, rows, n_layers)


def verify(args):
    def adjacent_cosine(x):
        x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
        return float((x[1:] * x[:-1]).sum(axis=1).mean())

    print("model bin shape expected_rows nan zero_rows adjacent_cosine adjacent_cosine_centered status")
    problems = 0
    for name in args.models:
        modality = REGISTRY[name][0]
        kind, ext = ("Video", "mp4") if modality == "v" else ("Audio", "wav")
        for output in output_names(name):
            for bin_sec in args.bins:
                path = output_path(args.embeddings_dir, output, modality, bin_sec)
                expected = len(chunk_paths(args.stimulus_dir, kind, ext, bin_sec))
                if not path.exists():
                    problems += 1
                    print(output, bin_sec, "missing", expected, "-", "-", "-", "-", "MISSING")
                    continue
                x = np.load(path).astype(np.float64)
                nan = int((~np.isfinite(x)).sum())
                zero = int((np.abs(x).sum(axis=1) == 0).sum())
                ok = x.ndim == 2 and len(x) == expected and nan == 0 and zero == 0 and bool((x.std(axis=0) > 0).any())
                problems += not ok
                print(output, bin_sec, tuple(x.shape), expected, nan, zero, f"{adjacent_cosine(x):.4f}",
                      f"{adjacent_cosine(x - x.mean(axis=0)):.4f}", "OK" if ok else "MISMATCH")
    print(f"{problems} problem files")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=sorted(REGISTRY), default=list(REGISTRY))
    parser.add_argument("--bins", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--stimulus-dir", default=STIMULUS_DIR)
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--models-home", default=MODELS_HOME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N chunks of each bin.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check row counts and values of saved embeddings.")
    args = parser.parse_args()

    if args.verify:
        sys.exit(int(verify(args) > 0))

    os.environ["HF_HOME"] = args.models_home
    for variable in ("SOCKS_PROXY", "socks_proxy", "ALL_PROXY", "all_proxy",
                     "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.pop(variable, None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in args.models:
        extract(args, name)


if __name__ == "__main__":
    main()
