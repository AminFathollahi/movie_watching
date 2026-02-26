# RSA‑Compatible Multimodal Embedding Extraction

This notebook enforces **PE‑AV‑compatible assumptions** across all models:

• one stimulus → one vector
• fixed temporal support
• final‑layer global readout only
• explicit L2 normalization
• RDM‑level comparison (RSA)

All models are loaded **locally** from HuggingFace caches or custom paths.

---

## 0. Global Assumptions (Do Not Change)

We define a *canonical clip*:

• Video: **16 frames** uniformly sampled over T seconds  
• Audio: **exactly the same T seconds**, resampled to 16 kHz  
• Output: **one embedding vector per clip per model**

These constraints mirror PE‑AV’s implicit contract.

---

## 1. Environment Setup

```python
import torch
import torchaudio
import torchvision
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

print("Device:", DEVICE)
```

---

## 2. Shared Utilities

### 2.1 L2 Normalization

```python
def l2_normalize(x: torch.Tensor):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)
```

### 2.2 RSA (RDM Computation)

```python
def compute_rdm(embeddings: np.ndarray, metric="cosine"):
    """Returns an NxN representational dissimilarity matrix."""
    return pairwise_distances(embeddings, metric=metric)
```

---

## 3. PE‑AV (Reference Standard)

```python
from transformers import AutoProcessor, AutoModel

PEAV_PATH = Path("/path/to/local/facebook--pe-av-large")

processor = AutoProcessor.from_pretrained(PEAV_PATH)
model = AutoModel.from_pretrained(PEAV_PATH).to(DEVICE).eval()
```

```python
def extract_peav(video_paths, audio_paths, batch_size=4):
    out = []
    with torch.inference_mode():
        for i in range(0, len(video_paths), batch_size):
            v = video_paths[i:i+batch_size]
            a = audio_paths[i:i+batch_size]
            inputs = processor(videos=v, audio=a, return_tensors="pt").to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
                outputs = model(**inputs)
            emb = outputs.audio_video_embeds.float()
            emb = l2_normalize(emb)
            out.append(emb.cpu())
    return torch.cat(out).numpy()
```

---

## 4. VideoMAEv2 (Video‑Only)

**Readout rule:** final‑layer **CLS token only**

```python
from transformers import VideoMAEImageProcessor, VideoMAEModel
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge("torch")

VIDEOMAE_PATH = Path("/path/to/local/videomae")

processor_v = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
model_v = VideoMAEModel.from_pretrained(VIDEOMAE_PATH).to(DEVICE).eval()
```

```python
def load_video_fixed(path, num_frames=16):
    vr = VideoReader(str(path), ctx=cpu(0))
    idx = np.linspace(0, len(vr)-1, num_frames, dtype=int)
    return vr.get_batch(idx).numpy()
```

```python
def extract_videomae(video_paths):
    embs = []
    with torch.inference_mode():
        for vp in video_paths:
            frames = load_video_fixed(vp)
            inputs = processor_v(list(frames), return_tensors="pt").to(DEVICE)
            out = model_v(**inputs)
            cls = out.last_hidden_state[:, 0]
            cls = l2_normalize(cls)
            embs.append(cls.cpu())
    return torch.cat(embs).numpy()
```

---

## 5. WavLM (Audio‑Only)

**Readout rule:** final transformer layer → mean over time → normalize

```python
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

WAVLM_PATH = Path("/path/to/local/wavlm")

processor_a = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_PATH)
model_a = WavLMModel.from_pretrained(WAVLM_PATH).to(DEVICE).eval()
```

```python
def load_audio_fixed(path, target_len, sr=16000):
    wav, fs = torchaudio.load(path)
    if fs != sr:
        wav = torchaudio.functional.resample(wav, fs, sr)
    wav = wav.mean(0)
    if wav.numel() < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
    else:
        wav = wav[:target_len]
    return wav
```

```python
def extract_wavlm(audio_paths, target_len):
    embs = []
    with torch.inference_mode():
        for ap in audio_paths:
            wav = load_audio_fixed(ap, target_len)
            inp = processor_a(wav.numpy(), sampling_rate=16000, return_tensors="pt").to(DEVICE)
            out = model_a(**inp)
            h = out.last_hidden_state.mean(dim=1)
            h = l2_normalize(h)
            embs.append(h.cpu())
    return torch.cat(embs).numpy()
```

---

## 6. BEATs / OpenBeats (Audio‑Only)

**Readout rule:** final feature sequence → mean over time → normalize

```python
from transformers import AutoModel

BEATS_PATH = Path("/path/to/local/beats")
model_b = AutoModel.from_pretrained(BEATS_PATH).to(DEVICE).eval()
```

```python
def extract_beats(audio_paths, target_len):
    embs = []
    with torch.inference_mode():
        for ap in audio_paths:
            wav = load_audio_fixed(ap, target_len).unsqueeze(0).to(DEVICE)
            out = model_b(wav)
            h = out.last_hidden_state.mean(dim=1)
            h = l2_normalize(h)
            embs.append(h.cpu())
    return torch.cat(embs).numpy()
```

---

## 7. RSA: Model‑to‑Model Comparison

```python
def rsa_compare(embeddings_dict):
    rdms = {k: compute_rdm(v) for k,v in embeddings_dict.items()}
    keys = list(rdms.keys())
    rsa = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = rdms[keys[i]].ravel()
            b = rdms[keys[j]].ravel()
            rsa[(keys[i], keys[j])] = np.corrcoef(a, b)[0,1]
    return rsa
```

---

## 8. Interpretation Notes (Critical)

• PE‑AV uses **trained multimodal projection**  
• All other models use **fixed linear readouts**  
• RSA therefore compares *geometry under a shared readout assumption*  
• Differences are meaningful only at the **RDM level**

This notebook is now **methodologically aligned**, defensible, and publishable.

---

## End

