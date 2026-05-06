<div align="center">

# FaceVeil

### Multi-stream Deepfake Detection — Spatial · Frequency · Temporal

<br />

![React](https://img.shields.io/badge/React-18-black?style=flat-square&logo=react&logoColor=61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-black?style=flat-square&logo=pytorch&logoColor=EE4C2C)
![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++-black?style=flat-square)
![Models](https://img.shields.io/badge/Models-3-black?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-97.6%25-black?style=flat-square&color=00dfa2)

<br />

**FaceVeil** is a multi-stream deepfake detection framework that combines spatial, frequency, and temporal representations using adaptive fusion and transformer-based multimodal learning.

</div>

---

# Overview

FaceVeil detects deepfake manipulation in both images and videos using three progressively powerful architectures trained on FaceForensics++.

The proposed **MSTF-Trans** framework fuses:

- Spatial RGB representations
- Frequency-domain FFT representations
- Temporal optical-flow representations

through an **Adaptive Gated Fusion (AGF)** mechanism and a **Cross-Stream Transformer Encoder**, achieving:

- **97.6% AUC**
- **96.3% Accuracy**
- **95.8% F1-score**

on the FaceForensics++ benchmark.

---

# Key Contributions

- Multi-stream deepfake detection using spatial, frequency, and temporal modalities
- Adaptive Gated Fusion (AGF) for per-sample dynamic stream weighting
- Cross-stream transformer encoder for multimodal interaction
- Frequency-domain analysis for GAN artifact detection
- Temporal consistency modeling using optical flow
- Strong robustness against multiple manipulation types

---

# Models

```text
SDB  ·  Spatial-Domain Baseline
     EfficientNet-B0 → GlobalAvgPool → FC(256) → FC(1)

     Streams  : RGB only
     AUC      : 87.2%
     Acc      : 85.6%
     F1       : 84.9%

──────────────────────────────────────────────────────

DSFN  ·  Dual-Branch Spatio-Frequency Network
      EfficientNet-B0 + FFT-CNN + CrossAttentionFusion

      Streams  : RGB + Frequency (FFT 56×56)
      AUC      : 93.1%
      Acc      : 91.4%
      F1       : 90.8%

──────────────────────────────────────────────────────

MSTF-Trans  ·  Multi-Stream Temporal Fusion Transformer ★
            EfficientNet-B4 + FFT-CNN + TemporalFlowEncoder
            + AdaptiveGatedFusion + Transformer (h=4, d=4)

            Streams  : RGB + Frequency + Optical Flow
            AUC      : 97.6%
            Acc      : 96.3%
            F1       : 95.8%
```

---

# Architecture

## MSTF-Trans Full Architecture

```text
                     ┌─────────────────────────────────────────────┐
Input RGB (3,224,224)│  SPATIAL STREAM                             │
──────────────────── │  EfficientNet-B4 → Linear(1792→256) → (B,256)
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input FFT (1,56,56)  │  FREQUENCY STREAM                           │
──────────────────── │  CNN[32→64→128] → Linear(2048→256) → (B,256)
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input Flow (1,56,56) │  TEMPORAL STREAM                            │
──────────────────── │  CNN[32→64→128] → Linear(1152→128→256)
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CROSS-STREAM TRANSFORMER                   │
                     │  Tokens: [CLS | s_tok | f_tok | t_tok]      │
                     │  4× TransformerEncoderLayer (heads=4, D=256)
                     │  → CLS token output (B, 256)                │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  ADAPTIVE GATED FUSION (AGF)                │
                     │  MLP(768→128→3) → Softmax → weights α       │
                     │  output = Σ αᵢ · streamᵢ → LayerNorm        │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CLASSIFICATION HEAD                        │
                     │  Cat[CLS, AGF] → FC(512→128→64→1) → Sigmoid │
                     └─────────────────────────────────────────────┘
```

---

# Adaptive Gated Fusion (AGF)

The **Adaptive Gated Fusion (AGF)** module dynamically learns which modality matters most for each sample.

Instead of fixed concatenation, AGF performs per-sample stream weighting:

```python
# Per-sample dynamic weighting
alpha = softmax(MLP(concat[s, f, t]))   # (B, 3)

output = (
    alpha[:,0] * s +
    alpha[:,1] * f +
    alpha[:,2] * t
)
```

## Example Behavior

- Texture-manipulated fake → spatial stream weighted higher
- Temporal inconsistency → flow stream weighted higher
- GAN-generated image → frequency stream weighted higher

---

# Why Frequency Matters

Deepfake generators often leave subtle artifacts in the frequency domain that are difficult to observe in RGB space.

The FFT branch captures:

- abnormal periodic patterns
- GAN upsampling artifacts
- spectral inconsistencies
- texture irregularities

that improve manipulation detection robustness.

---

# Why Temporal Modeling Matters

Video deepfakes often exhibit frame-level temporal inconsistencies:

- unnatural motion
- flickering artifacts
- inconsistent facial dynamics
- unstable expressions

The optical-flow stream models these temporal abnormalities.

---

# Experimental Results

| Model | Streams | AUC | Accuracy | F1-score | EER |
|:--|:--|--:|--:|--:|--:|
| **MSTF-Trans ★** | S + F + T | **97.6** | **96.3** | **95.8** | **3.1** |
| w/o Temporal | S + F | 93.5 | 91.8 | 91.2 | 7.9 |
| w/o Frequency | S + T | 90.2 | 88.7 | 88.1 | 11.2 |
| w/o Spatial | F + T | 88.4 | 87.0 | 86.6 | 13.4 |
| DSFN | S + F | 93.1 | 91.4 | 90.8 | 8.7 |
| SDB | S | 87.2 | 85.6 | 84.9 | 14.1 |

---

# Dataset

## Source

[Deepfake Detection Dataset — Kaggle](https://www.kaggle.com/datasets/himachhatbar1700/deepfake)

---

## Dataset Statistics

| Split | Total Samples | Real | Fake |
|:--|--:|--:|--:|
| Train | 126,786 | ~50% | ~50% |
| Validation | 27,172 | ~50% | ~50% |
| Test | 27,174 | ~50% | ~50% |
| **Grand Total** | **181,132** | — | — |

---

## Modalities

- Images (`.jpg`, `.jpeg`, `.png`, `.bmp`)
- Videos (`.mp4`, `.avi`, `.mov`, `.mkv`)

---

## Expected Directory Structure

```text
DEEPFAKE_DATASET/
├── split/
│   ├── train/
│   │   ├── images/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── videos/
│   │       ├── real/
│   │       └── fake/
│   ├── val/
│   │   ├── images/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── videos/
│   │       ├── real/
│   │       └── fake/
│   └── test/
│       ├── images/
│       │   ├── real/
│       │   └── fake/
│       └── videos/
│           ├── real/
│           └── fake/
└── cache/
    ├── fft_train.json
    └── flow_train.json
```

---

# Downloading the Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Configure credentials
# ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d himachhatbar1700/deepfake

# Extract dataset
unzip deepfake.zip -d DEEPFAKE_DATASET/
```

---

# Installation

## Requirements

- Python 3.10+
- CUDA 11.8+
- 8GB+ GPU VRAM
- Node.js 18+

---

## Clone Repository

```bash
git clone https://github.com/your-username/faceveil
cd faceveil
```

---

## Install Dependencies

```bash
# frontend
npm install

# backend / training
pip install -r requirements.txt
```

---

# Training Configuration

```text
backbone     EfficientNet-B0 (SDB, DSFN)
             EfficientNet-B4 (MSTF-Trans)

loss         LabelSmooth BCE  ε = 0.1

optimizer    AdamW
lr           3e-4
weight_decay 1e-4

scheduler    Warmup Cosine
             (3 warmup epochs / 10 total)

batch_size   32
img_size     224 × 224
fft_size     56 × 56
flow_size    56 × 56

threshold    0.70
feat_dim     256

tf_heads     4
tf_depth     4

seed         42
```

# Frontend Stack

```text
frontend    React 18 · Recharts · Lucide
training    PyTorch · torchvision · scikit-learn
tracking    Loopz framework (checkpoint + metrics)
dataset     FaceForensics++ , DFDC - selfconstructed.
```

---

# Quickstart

## Development

```bash
npm start
```

Opens at:

```text
http://localhost:3000
```

---

## Production Build

```bash
npm run build
```

---

# Connecting Your Models

The detector runs simulated inference by default.

To connect trained `.pt` weights, create a FastAPI backend.

---

## Example Backend

```python
from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile, model: str = "MSTF-Trans"):

    # load checkpoint
    # preprocess input
    # run inference

    return {
        "fake_prob": 0.82,
        "real_prob": 0.18,
        "verdict": "DEEPFAKE"
    }
```

---

## Frontend API Call

```javascript
const res = await fetch(`/predict?model=${model.id}`, {
  method: 'POST',
  body: formData,
});

const data = await res.json();
setResult(data);
```

---

# Project Structure

```text
faceveil/
├── src/
│   ├── App.js
│   ├── index.css
│   └── components/
│       ├── Navbar.js
│       ├── Hero.js
│       ├── Detector.js
│       ├── Models.js
│       ├── Ablation.js
│       └── Results.js
│
└── public/
    └── index.html
```

---

# Future Work

- Real-time video inference
- Lightweight mobile deployment
- Cross-dataset generalization
- Adversarial robustness
- Explainable multimodal attention maps
- Self-supervised pretraining

---

# Citation

```bibtex
@article{faceveil2026,
  title={FaceVeil: Multi-Stream Deepfake Detection via Adaptive Temporal-Frequency Fusion},
  author={Hima Chhatbar},
  year={2026}
}
```

---

# License

This project is released under the GNU License.

---

<div align="center">

### SDB · DSFN · MSTF-Trans

*Deepfake Detection Research*

</div>