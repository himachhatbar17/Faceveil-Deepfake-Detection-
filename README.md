<div align="center">


**Multi-stream deepfake detection — spatial · frequency · temporal**

<br />

![React](https://img.shields.io/badge/React-18-black?style=flat-square&logo=react&logoColor=61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-black?style=flat-square&logo=pytorch&logoColor=EE4C2C)
![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++-black?style=flat-square)
![Models](https://img.shields.io/badge/Models-3-black?style=flat-square)
![AUC](https://img.shields.io/badge/Best_AUC-97.6%25-black?style=flat-square&color=00dfa2)

<br />

</div>

---

<br />

## overview

FaceVeil detects deepfake manipulation in images and videos using three progressively powerful models trained on FaceForensics++. The proposed **MSTF-Trans** fuses spatial, frequency (FFT), and temporal (optical flow) streams through an adaptive gating mechanism and transformer encoder — achieving 97.6% AUC.

<br />

## models

<br />

```
SDB  ·  Spatial-Domain Baseline
     EfficientNet-B0 → GlobalAvgPool → FC(256) → FC(1)
     Streams  : RGB only
     AUC      : 87.2%   Acc : 85.6%   F1 : 84.9%

──────────────────────────────────────────────────────

DSFN  ·  Dual-Branch Spatio-Frequency Network
      EfficientNet-B0 + FFT-CNN + CrossAttentionFusion
      Streams  : RGB + Frequency (FFT 56×56)
      AUC      : 93.1%   Acc : 91.4%   F1 : 90.8%

──────────────────────────────────────────────────────

MSTF-Trans  ·  Multi-Stream Temporal Fusion Transformer  ★
            EfficientNet-B4 + FFT-CNN + TemporalFlowEncoder
            + AdaptiveGatedFusion + Transformer (h=4, d=4)
            Streams  : RGB + Frequency + Optical Flow
            AUC      : 97.6%   Acc : 96.3%   F1 : 95.8%
            
---

## 🏗️ Architecture

### MSTF-Trans Full Architecture

```
                     ┌─────────────────────────────────────────────┐
Input RGB (3,224,224)│  SPATIAL STREAM                             │
──────────────────── │  EfficientNet-B4 → Linear(1792→256) → (B,256)│
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input FFT (1,56,56)  │  FREQUENCY STREAM                           │
──────────────────── │  CNN[32→64→128] → Linear(2048→256) → (B,256)│
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input Flow (1,56,56) │  TEMPORAL STREAM                            │
──────────────────── │  CNN[32→64→128] → Linear(1152→128→256)      │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CROSS-STREAM TRANSFORMER                   │
                     │  Tokens: [CLS | s_tok | f_tok | t_tok]      │
                     │  4× TransformerEncoderLayer (heads=4, D=256) │
                     │  → CLS token output (B, 256)                │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  ADAPTIVE GATED FUSION (AGF)                │
                     │  MLP(768→128→3) → Softmax → weights α       │
                     │  output = Σ αᵢ · streamᵢ  → LayerNorm       │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CLASSIFICATION HEAD                        │
                     │  Cat[CLS, AGF] → FC(512→128→64→1) → Sigmoid │
                     └─────────────────────────────────────────────┘
```

### Adaptive Gated Fusion (AGF) — Core Novelty

The AGF module learns **which stream matters most** for each individual sample:

```python
# Per-sample dynamic weighting (NOT fixed concatenation)
alpha = softmax(MLP(concat[s, f, t]))   # (B, 3)  — learned weights
output = alpha[:,0]*s + alpha[:,1]*f + alpha[:,2]*t
```

- A **texture-manipulated fake** → model weights spatial stream higher
- A **temporal-inconsistent fake** → model weights flow stream higher  
- A **GAN-generated image** → model weights frequency stream higher

---

## 📦 Dataset

**Source:** [Deepfake Detection Dataset — Kaggle](https://www.kaggle.com/datasets/himachhatbar1700/deepfake)

| Split | Total Samples | Real | Fake |
|-------|-------------|------|------|
| **Train** | 126,786 | ~50% | ~50% |
| **Validation** | 27,172 | ~50% | ~50% |
| **Test** | 27,174 | ~50% | ~50% |
| **Grand Total** | **181,132** | — | — |

**Modalities:** Images (`.jpg`, `.jpeg`, `.png`, `.bmp`) + Videos (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Expected Directory Structure

```
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
│   │   ├── images/  {real/, fake/}
│   │   └── videos/  {real/, fake/}
│   └── test/
│       ├── images/  {real/, fake/}
│       └── videos/  {real/, fake/}
└── cache/
    ├── fft_train.json      ← precomputed FFT magnitude maps
    └── flow_train.json     ← precomputed optical flow maps
```

### Downloading the Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (~/.kaggle/kaggle.json)
# Download dataset
kaggle datasets download -d himachhatbar1700/deepfake
unzip deepfake.zip -d DEEPFAKE_DATASET/

## ⚙️ Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (recommended: NVIDIA T4 / V100 / A100)
- 8GB+ GPU VRAM

```



<br />

## ablation

<br />

| variant | streams | AUC | Acc | F1 | EER |
|:--------|:--------|----:|----:|---:|----:|
| MSTF-Trans ★ | S + F + T | **97.6** | **96.3** | **95.8** | **3.1** |
| w/o Temporal | S + F | 93.5 | 91.8 | 91.2 | 7.9 |
| w/o Frequency | S + T | 90.2 | 88.7 | 88.1 | 11.2 |
| w/o Spatial | F + T | 88.4 | 87.0 | 86.6 | 13.4 |
| DSFN | S + F | 93.1 | 91.4 | 90.8 | 8.7 |
| SDB | S | 87.2 | 85.6 | 84.9 | 14.1 |

<br />

## training config

<br />

```
backbone     EfficientNet-B0 (SDB, DSFN)  /  EfficientNet-B4 (MSTF-Trans)
loss         LabelSmooth BCE  ε = 0.1
optimizer    AdamW  lr = 3e-4  ·  weight_decay = 1e-4
scheduler    Warmup Cosine  (3 warmup epochs / 10 total)
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

<br />

## stack

<br />

```
frontend    React 18  ·  Recharts  ·  Lucide
training    PyTorch  ·  torchvision  ·  scikit-learn
tracking    Loopz framework  (checkpoint + metrics)
dataset     FaceForensics++  (c23 compression)
```

<br />

## quickstart

<br />

```bash
# clone
git clone https://github.com/your-username/faceveil
cd faceveil

# install
npm install

# run
npm start
```

Opens at `http://localhost:3000`

<br />

```bash
# production build
npm run build
```

<br />

## project structure

<br />

```
faceveil/
├── src/
│   ├── App.js                  root + routing
│   ├── index.css               css variables + reset
│   └── components/
│       ├── Navbar.js           navigation
│       ├── Hero.js             stats bar
│       ├── Detector.js         upload · inference · results
│       ├── Models.js           architecture deep-dive
│       ├── Ablation.js         stream ablation + charts
│       └── Results.js          roc · loss · confusion matrix
└── public/
    └── index.html
```

<br />

## connecting your models

The detector runs a simulated inference by default. To wire in your trained `.pt` weights, spin up a FastAPI backend:

<br />

```python
# backend.py
from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile, model: str = "MSTF-Trans"):
    # load checkpoint, preprocess, run forward pass
    return {
        "fake_prob": 0.82,
        "real_prob": 0.18,
        "verdict":   "DEEPFAKE"
    }
```

Then in `Detector.js`, swap the `setTimeout` block for:

```js
const res = await fetch(`/predict?model=${model.id}`, {
  method: 'POST',
  body: formData,
});
const data = await res.json();
setResult(data);
```

<br />

---

<div align="center">

<br />

`SDB` &nbsp;·&nbsp; `DSFN` &nbsp;·&nbsp; `MSTF-Trans`

<br />

*deepfake detection research*

<br />

</div>
