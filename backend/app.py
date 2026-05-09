from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import os
app = Flask(__name__)
CORS(app, origins=["https://faceveil-deepfake-detection.vercel.app"])
# ── Device ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.7
IMG_SIZE = 224
FFT_SIZE = 56

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (copied exactly from your notebook)
# ══════════════════════════════════════════════════════════════

# ── Model 1: SDB ──────────────────────────────────────────────
class SDB(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, rgb, fft=None, flow=None):
        x = self.features(rgb)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)


# ── Frequency CNN (shared by DSFN and MSTF) ───────────────────
class FreqCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 256), nn.ReLU())

    def forward(self, x):
        return self.proj(self.net(x))


# ── Model 2: DSFN ─────────────────────────────────────────────
class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return self.norm(out + q)


class DSFN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.spatial = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.spatial_proj = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())
        self.freq = FreqCNN()
        self.cross_attn = CrossAttention(256, 4)
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, rgb, fft, flow=None):
        fs = self.spatial_proj(self.spatial(rgb)).unsqueeze(1)
        ff = self.freq(fft).unsqueeze(1)
        fs2 = self.cross_attn(fs, ff, ff).squeeze(1)
        ff2 = self.cross_attn(ff, fs, fs).squeeze(1)
        return self.classifier(torch.cat([fs2, ff2], dim=1)).squeeze(1)


# ── Model 3: MSTF-Trans ───────────────────────────────────────
class MSTF(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.spatial = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.spatial_proj = nn.Sequential(nn.Linear(1792, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.freq = FreqCNN()
        self.temporal = FreqCNN()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gate = nn.Sequential(nn.Linear(768, 3), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, rgb, fft, flow):
        fs = self.spatial_proj(self.spatial(rgb))
        ff = self.freq(fft)
        ft = self.temporal(flow)
        tokens = torch.stack([fs, ff, ft], dim=1)
        h = self.transformer(tokens)
        hs, hf, ht = h[:, 0], h[:, 1], h[:, 2]
        fused = torch.cat([hs, hf, ht], dim=1)
        g = self.gate(fused)
        z = g[:, 0:1] * hs + g[:, 1:2] * hf + g[:, 2:3] * ht
        return self.classifier(torch.cat([z, fused], dim=-1) if False else fused).squeeze(1)


# ══════════════════════════════════════════════════════════════
# LOAD CHECKPOINTS
# ══════════════════════════════════════════════════════════════
def load_model(ModelClass, ckpt_path):
    model = ModelClass().to(DEVICE)
    if os.path.exists(ckpt_path):
        try:
            with torch.serialization.add_safe_globals([np._core.multiarray.scalar]):
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        except Exception:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # handle loopz checkpoint wrapper
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Loaded {ckpt_path}")
    else:
        print(f"  ⚠ Checkpoint not found: {ckpt_path} — using untrained weights")
    model.eval()
    return model

sdb_model   = load_model(SDB,  "checkpoints/best_SDB.pt")
dsfn_model  = load_model(DSFN, "checkpoints/best_DSFN.pt")
mstf_model  = load_model(MSTF, "checkpoints/best_MSTF-Trans.pt")

MODELS = {
    "SDB (Spatial only — fastest)":       sdb_model,
    "DSFN (Spatial + Frequency)":          dsfn_model,
    "MSTF-Trans (Full 3-stream — best)":   mstf_model,
}

# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def compute_fft(img_pil):
    gray = np.array(img_pil.convert("L").resize((FFT_SIZE*4, FFT_SIZE*4)))
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    magnitude = cv2.resize(magnitude, (FFT_SIZE, FFT_SIZE))
    return torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def zero_flow():
    return torch.zeros(1, 1, FFT_SIZE, FFT_SIZE)

# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
def predict(image: Image.Image, model_name: str):
    if image is None:
        return "Please upload an image.", {}, ""

    img = image.convert("RGB")

    rgb  = transform(img).unsqueeze(0).to(DEVICE)
    fft  = compute_fft(img).to(DEVICE)
    flow = zero_flow().to(DEVICE)

    model = MODELS[model_name]

    with torch.no_grad():
        if "SDB" in model_name:
            logit = model(rgb)
        elif "DSFN" in model_name:
            logit = model(rgb, fft)
        else:
            logit = model(rgb, fft, flow)

        prob_fake = torch.sigmoid(logit).item()

    prob_real = 1.0 - prob_fake
    verdict = "🔴 FAKE" if prob_fake >= THRESHOLD else "🟢 REAL"
    confidence = prob_fake if prob_fake >= THRESHOLD else prob_real

    label = f"{verdict}  —  Confidence: {confidence*100:.1f}%"
    scores = {"REAL": round(prob_real, 4), "FAKE": round(prob_fake, 4)}

    detail = (
        f"**Model:** {model_name}\n\n"
        f"**P(fake):** {prob_fake:.4f}  |  **Threshold:** {THRESHOLD}\n\n"
        f"**Verdict:** {verdict}"
    )

    return label, scores, detail


# Flask API
app = Flask(__name__)
CORS(app)

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': list(MODELS.keys()),
        'default': 'MSTF-Trans (Full 3-stream — best)'
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file upload under the key "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    model_name = request.form.get('model_name', '')
    if model_name not in MODELS:
        return jsonify({'error': 'Invalid model_name. Choose one of the available models.'}), 400

    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({'error': 'Unable to parse uploaded image. Please upload a valid image file.'}), 400

    label, scores, detail = predict(image, model_name)

    return jsonify({
        'label': label,
        'scores': scores,
        'detail': detail,
        'model_name': model_name,
        'threshold': THRESHOLD,
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'FaceVeil backend is running.', 'models': list(MODELS.keys())})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
