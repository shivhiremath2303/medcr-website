"""Inference module for secondary document forgery detection.

Loads the pre-trained forgery detector bundle and evaluates document image bytes
using Error Level Analysis (ELA), multi-patch statistics, and HOG features.
"""

import io
import os
import joblib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import hog

BUNDLE_FILENAME = "forgery_detector_bundle.joblib"
# Resolve relative to backend root directory
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), BUNDLE_FILENAME)

_bundle = None
_model = None
_image_size = (128, 128)
_low_threshold = 0.35
_high_threshold = 0.65


def _load_bundle():
    global _bundle, _model, _image_size, _low_threshold, _high_threshold
    if _model is not None:
        return
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Model bundle not found at {BUNDLE_PATH}")
    
    _bundle = joblib.load(BUNDLE_PATH)
    _model = _bundle["model"]
    _image_size = tuple(_bundle.get("image_size", (128, 128)))
    thresholds = _bundle.get("review_thresholds", {})
    _low_threshold = float(thresholds.get("low", 0.35))
    _high_threshold = float(thresholds.get("high", 0.65))


def extract_features_from_bytes(image_bytes: bytes):
    _load_bundle()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(_image_size)
    except Exception as e:
        raise ValueError(f"Invalid or unreadable image file: {str(e)}")

    buff = io.BytesIO()
    image.save(buff, format="JPEG", quality=90)
    buff.seek(0)
    recompressed = Image.open(buff).convert("RGB")
    ela_img = ImageChops.difference(image, recompressed)
    ela_img = ImageEnhance.Brightness(ela_img).enhance(10)
    ela_rgb = np.asarray(ela_img, dtype=np.float32) / 255.0

    features = []
    for c in range(3):
        ch = ela_rgb[:, :, c]
        features.extend([
            float(ch.mean()), float(ch.std()),
            float(np.percentile(ch, 90)), float(np.percentile(ch, 95)),
            float(np.percentile(ch, 99)), float(ch.max())
        ])
        for gy in range(4):
            for gx in range(4):
                patch = ch[gy*32:(gy+1)*32, gx*32:(gx+1)*32]
                features.extend([float(patch.mean()), float(patch.std())])

    ela_gray = np.asarray(ela_img.convert("L"), dtype=np.uint8)
    features.extend(hog(
        ela_gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True
    ).tolist())

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    for c in range(3):
        features.extend([
            float(rgb[:, :, c].mean()), float(rgb[:, :, c].std()),
            float(np.percentile(rgb[:, :, c], 10)),
            float(np.percentile(rgb[:, :, c], 90))
        ])

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def predict_document(image_bytes: bytes) -> dict:
    """Predict forgery status from document image bytes.
    
    Returns standard display labels:
      - 'Likely Authentic'
      - 'Potentially Tampered'
      - 'Review Required'
    """
    _load_bundle()
    x = extract_features_from_bytes(image_bytes)
    tampered_score = float(_model.predict_proba(x)[0, 1])

    if tampered_score >= _high_threshold:
        display_label = "Potentially Tampered"
        raw_result = "POTENTIALLY TAMPERED"
    elif tampered_score <= _low_threshold:
        display_label = "Likely Authentic"
        raw_result = "LIKELY AUTHENTIC"
    else:
        display_label = "Review Required"
        raw_result = "REVIEW REQUIRED"

    return {
        "result": display_label,
        "raw_result": raw_result,
        "tampered_score": round(tampered_score, 4),
        "low_threshold": _low_threshold,
        "high_threshold": _high_threshold,
        "disclaimer": "This is an experimental, secondary screening feature. Output is not guaranteed proof of authenticity or tampering."
    }
