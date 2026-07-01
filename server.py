"""
server.py
─────────
Flask REST API that powers the Artify browser extension.

The extension (content.js / popup.js) already calls:
  POST  http://localhost:5000/detect   { image, generate_heatmap }
  GET   http://localhost:5000/health

This server loads the trained model and serves those endpoints.

Usage:
    # Basic (CPU)
    python server.py --checkpoint checkpoints/best_model.pt

    # GPU + custom port
    python server.py --checkpoint checkpoints/best_model.pt --device cuda --port 5000

    # Demo mode (no checkpoint – returns dummy response for UI testing)
    python server.py --demo
"""

import os
import io
import sys
import time
import base64
import argparse
import traceback
from pathlib import Path

import torch
import yaml
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────────────
#  Lazy imports (only load model libs when not in demo mode)
# ──────────────────────────────────────────────────────────────────────
_model     = None
_gradcam   = None
_transform = None
_device    = None


def _load_model(checkpoint: str, config: str, device_str: str):
    global _model, _gradcam, _transform, _device

    from models import build_detector
    from data.dataset import build_transforms
    from utils.visualization import GradCAM

    with open(config) as f:
        cfg = yaml.safe_load(f)

    _device = torch.device(device_str)
    print(f"[server] loading model on {_device} …")

    # force_load_visual=True: the server receives arbitrary browser images
    # (no CLIP cache for them), so the frozen ViT must be loaded to extract
    # tokens for both prediction and the heat map.
    _model = build_detector(cfg, force_load_visual=True).to(_device)
    ckpt   = torch.load(checkpoint, map_location=_device, weights_only=False)
    # strict=False: checkpoint was saved in CLIP-cache mode (load_visual=False)
    # so frozen ViT keys are absent; they are already correct from open_clip.
    _model.load_state_dict(ckpt["model"], strict=False)
    _model.eval()

    _transform = build_transforms(image_size=224, augment=False)
    _gradcam   = GradCAM(_model)

    print("[server] model ready ✓")


# ──────────────────────────────────────────────────────────────────────
#  Image decoding helpers
# ──────────────────────────────────────────────────────────────────────

def _decode_image(image_field: str) -> Image.Image:
    """
    Accept:
      • data:image/...;base64,<b64>   (from file reader / fetch)
      • raw base64 string
      • http/https URL (fetched server-side as fallback)
    Returns PIL Image in RGB.
    """
    if image_field.startswith("data:"):
        # data URL
        header, b64_data = image_field.split(",", 1)
        raw = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if image_field.startswith("http://") or image_field.startswith("https://"):
        import requests
        resp = requests.get(image_field, timeout=10,
                            headers={"User-Agent": "ArtifyBot/1.0"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # Bare base64
    try:
        raw = base64.b64decode(image_field)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ValueError("Cannot decode image. Provide a data URL or base64 string.")


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ──────────────────────────────────────────────────────────────────────
#  Flask app
# ──────────────────────────────────────────────────────────────────────

app  = Flask(__name__)
CORS(app, origins="*")   # allow extension origin

_DEMO_MODE = False


# ── /health ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "demo" if _DEMO_MODE else "loaded",
        "device":  str(_device) if _device else "n/a",
    })


# ── /detect ───────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    Request JSON:
        {
          "image":           "<data-URL or base64 string>",
          "generate_heatmap": true | false
        }

    Response JSON (matches Artify extension expectations):
        {
          "prediction":        "AI-Generated" | "Real",
          "confidence":        0.0-1.0,
          "probability":       0.0-1.0,   # raw P(AI)
          "processing_time_ms": float,
          "heatmap_overlay":   "<data-URL>"   // only if generate_heatmap=true
        }
    """
    t_start = time.perf_counter()

    try:
        body = request.get_json(force=True)
        if not body or "image" not in body:
            return jsonify({"error": "Missing 'image' field in request body"}), 400

        generate_heatmap = bool(body.get("generate_heatmap", False))

        # ── decode ────────────────────────────────────────────────────
        pil_img = _decode_image(body["image"])

        # ── demo mode: return synthetic result ────────────────────────
        if _DEMO_MODE:
            import random
            prob = round(random.uniform(0.1, 0.95), 4)
            label = "AI-Generated" if prob >= 0.5 else "Real"
            return jsonify({
                "prediction":          label,
                "confidence":          round(max(prob, 1 - prob), 4),
                "probability":         prob,
                "processing_time_ms":  round((time.perf_counter() - t_start) * 1000, 2),
                "heatmap_overlay":     None,
            })

        # ── real inference ────────────────────────────────────────────
        tensor = _transform(pil_img).unsqueeze(0).to(_device)   # [1,3,224,224]

        with torch.no_grad():
            logits = _model(tensor)
            prob   = torch.sigmoid(logits).item()

        label      = "AI-Generated" if prob >= 0.5 else "Real"
        confidence = max(prob, 1.0 - prob)

        # ── optional heatmap ──────────────────────────────────────────
        heatmap_data_url = None
        if generate_heatmap:
            try:
                from utils.visualization import heatmap_to_overlay
                cam     = _gradcam(tensor)
                overlay = heatmap_to_overlay(pil_img, cam, alpha=0.5)
                heatmap_data_url = _pil_to_data_url(overlay)
            except Exception as e:
                print(f"[server] heatmap generation failed: {e}", file=sys.stderr)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return jsonify({
            "prediction":          label,
            "confidence":          round(confidence, 6),
            "probability":         round(prob, 6),
            "processing_time_ms":  round(elapsed_ms, 2),
            "heatmap_overlay":     heatmap_data_url,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ── /batch_detect  (bonus endpoint for dashboard batch analysis) ──────

@app.route("/batch_detect", methods=["POST"])
def batch_detect():
    """
    Request JSON:  { "images": ["<data-URL>", ...] }
    Response JSON: { "results": [ { prediction, confidence, probability }, ... ] }
    """
    try:
        body   = request.get_json(force=True)
        images = body.get("images", [])
        if not images:
            return jsonify({"error": "No images provided"}), 400

        results = []
        for img_str in images:
            pil_img = _decode_image(img_str)
            if _DEMO_MODE:
                import random
                prob = round(random.uniform(0.1, 0.95), 4)
                label = "AI-Generated" if prob >= 0.5 else "Real"
                results.append({"prediction": label,
                                 "confidence": round(max(prob, 1-prob), 4),
                                 "probability": prob})
            else:
                tensor = _transform(pil_img).unsqueeze(0).to(_device)
                with torch.no_grad():
                    prob = torch.sigmoid(_model(tensor)).item()
                label = "AI-Generated" if prob >= 0.5 else "Real"
                results.append({"prediction": label,
                                 "confidence": round(max(prob, 1-prob), 6),
                                 "probability": round(prob, 6)})

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--device",     default=None,
                   help="cuda / cpu (auto if omitted)")
    p.add_argument("--port",       type=int, default=5000)
    p.add_argument("--host",       default="127.0.0.1")
    p.add_argument("--demo",       action="store_true",
                   help="Run in demo mode without a checkpoint (for UI testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _DEMO_MODE = args.demo or (args.checkpoint is None)

    if _DEMO_MODE:
        print("[server] ⚠  Running in DEMO mode – responses are random placeholders.")
        print("[server] Train a model and pass --checkpoint to enable real detection.\n")
    else:
        if not Path(args.checkpoint).exists():
            print(f"[server] Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        device_str = (
            args.device if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _load_model(args.checkpoint, args.config, device_str)

    print(f"[server] Starting on http://{args.host}:{args.port}")
    print(f"[server] Extension endpoint: POST http://localhost:{args.port}/detect\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
