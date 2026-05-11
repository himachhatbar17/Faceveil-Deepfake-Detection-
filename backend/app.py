from flask import Flask, request, jsonify, send_from_directory
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

# ── App setup ─────────────────────────────────────────────────
# Serve React build from /app/static_frontend
app = Flask(__name__, static_folder='static_frontend', static_url_path='')
CORS(app, origins=["*"])  # Allow all origins for HF Spaces

# ── Config ────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.7
IMG_SIZE  = 224
FFT_SIZE  = 56

print(f"  Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

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


class SDB(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features   = backbone.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, rgb, fft=None, flow=None):
        x = self.features(rgb)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)


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
        self.spatial      = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.spatial_proj = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())
        self.freq         = FreqCNN()
        self.cross_attn   = CrossAttention(256, 4)
        self.classifier   = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, rgb, fft, flow=None):
        fs  = self.spatial_proj(self.spatial(rgb)).unsqueeze(1)
        ff  = self.freq(fft).unsqueeze(1)
        fs2 = self.cross_attn(fs, ff, ff).squeeze(1)
        ff2 = self.cross_attn(ff, fs, fs).squeeze(1)
        return self.classifier(torch.cat([fs2, ff2], dim=1)).squeeze(1)


class MSTF(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.spatial      = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.spatial_proj = nn.Sequential(nn.Linear(1792, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.freq         = FreqCNN()
        self.temporal     = FreqCNN()
        enc_layer         = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.gate         = nn.Sequential(nn.Linear(768, 3), nn.Sigmoid())
        self.classifier   = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, rgb, fft, flow):
        fs     = self.spatial_proj(self.spatial(rgb))
        ff     = self.freq(fft)
        ft     = self.temporal(flow)
        tokens = torch.stack([fs, ff, ft], dim=1)
        h      = self.transformer(tokens)
        hs, hf, ht = h[:, 0], h[:, 1], h[:, 2]
        fused  = torch.cat([hs, hf, ht], dim=1)
        return self.classifier(fused).squeeze(1)


# ══════════════════════════════════════════════════════════════
# LOAD CHECKPOINTS
# ══════════════════════════════════════════════════════════════
def load_model(ModelClass, ckpt_path):
    model = ModelClass().to(DEVICE)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Loaded {ckpt_path}")
    else:
        print(f"  ⚠ Checkpoint not found: {ckpt_path} — using random weights")
    model.eval()
    return model

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
sdb_model  = load_model(SDB,  os.path.join(CKPT_DIR, "best_SDB.pt"))
dsfn_model = load_model(DSFN, os.path.join(CKPT_DIR, "best_DSFN.pt"))
mstf_model = load_model(MSTF, os.path.join(CKPT_DIR, "best_MSTF-Trans.pt"))

MODELS = {
    "SDB":        sdb_model,
    "DSFN":       dsfn_model,
    "MSTF-Trans": mstf_model,
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
    gray      = np.array(img_pil.convert("L").resize((FFT_SIZE*4, FFT_SIZE*4)))
    f         = np.fft.fft2(gray.astype(np.float32))
    fshift    = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    magnitude = cv2.resize(magnitude, (FFT_SIZE, FFT_SIZE))
    return torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def zero_flow():
    return torch.zeros(1, 1, FFT_SIZE, FFT_SIZE)

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

# ── Serve React frontend for ALL non-API routes ───────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # If requesting a static asset that exists, serve it
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # Otherwise serve index.html (React Router handles the rest)
    return send_from_directory(app.static_folder, 'index.html')


# ── Health check ──────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "device": DEVICE, "models": list(MODELS.keys())})


# ── Main inference endpoint ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file       = request.files['file']
    model_name = request.form.get('model_name', 'MSTF-Trans')

    # Normalise model_name from frontend display names
    if 'MSTF' in model_name:
        model_name = 'MSTF-Trans'
    elif 'DSFN' in model_name:
        model_name = 'DSFN'
    else:
        model_name = 'SDB'

    if model_name not in MODELS:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        rgb  = transform(img).unsqueeze(0).to(DEVICE)
        fft  = compute_fft(img).to(DEVICE)
        flow = zero_flow().to(DEVICE)

        model = MODELS[model_name]

        with torch.no_grad():
            if model_name == 'SDB':
                logit = model(rgb)
            elif model_name == 'DSFN':
                logit = model(rgb, fft)
            else:
                logit = model(rgb, fft, flow)

            prob_fake = torch.sigmoid(logit).item()

        prob_real = 1.0 - prob_fake
        is_fake   = prob_fake >= THRESHOLD
        verdict   = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC"

        return jsonify({
            "label":      f"{'🔴 FAKE' if is_fake else '🟢 REAL'} — Confidence: {max(prob_fake, prob_real)*100:.1f}%",
            "scores":     {"FAKE": round(prob_fake, 4), "REAL": round(prob_real, 4)},
            "detail":     f"Model: {model_name} | P(fake): {prob_fake:.4f} | Threshold: {THRESHOLD}",
            "model_name": model_name,
            "verdict":    verdict,
            "is_fake":    is_fake,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # HF Spaces requires port 7860
    app.run(host='0.0.0.0', port=7860, debug=False)

# 🔁 Self-ping every 14 minutes to prevent HF Space sleep
import threading
import urllib.request

def self_ping():
    import time
    while True:
        time.sleep(14 * 60)  # 14 minutes
        try:
            url = "https://hima0017-faceveil-deepfake-detection.hf.space/api/health"
            urllib.request.urlopen(url, timeout=10)
            print("✅ Self-ping success!")
        except Exception as e:
            print(f"❌ Ping failed: {e}")

# Start ping thread when app starts
ping_thread = threading.Thread(target=self_ping, daemon=True)
ping_thread.start()

# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)