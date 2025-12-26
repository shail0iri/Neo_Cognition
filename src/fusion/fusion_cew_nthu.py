"""
FusionCEWNTHU: small helper class to combine CEW eye-state CNN + NTHU-style features.

Main API used by:
  - realtime_cognition.py  ->  predict_eye_state(eye_img)
  - notebooks/test_fusion_cew_nthu.py -> predict_from_eye_and_features(eye_img, features_dict)

This version:
  - Loads CEW model from models/cew_best.keras or models/cew_final.keras
  - Accepts eye crops as BGR or grayscale numpy arrays
  - Returns (p_open, p_closed) in [0,1]
  - Optionally computes simple EAR-based metrics from provided features
"""

import os
import sys
import numpy as np
import cv2

# ------------------------------------------------------------------
# Add project root to sys.path so `src.*` imports work everywhere
# ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try to import keras loader
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def preprocess_eye_for_cew(eye_img, target_size=(80, 80)):
    """
    Takes an eye crop (H,W) or (H,W,3) in BGR / gray.
    Returns (1, H, W, 3) float32 RGB in [0,1] for CEW model.
    """
    if eye_img is None:
        return None

    img = eye_img.copy()

    # Make sure 3 channels
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        # try to squeeze weird shapes
        gray = np.squeeze(img)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # resize to target_size (H,W)
    gray = cv2.resize(gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    gray = gray.astype("float32") / 255.0  # normalize
    gray = np.expand_dims(gray, axis=-1)   # (H, W, 1)
    gray = np.expand_dims(gray, axis=0)    # (1, H, W, 1)

    return gray


def _safe_sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _compute_ear_from_values(left_ear=None, right_ear=None):
    """
    Takes numeric left/right EAR values (or None).
    Returns (left, right, avg, asymmetry_ratio).
    """
    if left_ear is None and right_ear is None:
        return np.nan, np.nan, np.nan, np.nan

    le = float(left_ear) if left_ear is not None else np.nan
    re = float(right_ear) if right_ear is not None else np.nan
    avg = np.nanmean([le, re])

    asym = np.nan
    try:
        if not np.isnan(le) and not np.isnan(re) and (le + re) != 0:
            asym = abs(le - re) / max(1e-6, (le + re) / 2.0)
    except Exception:
        asym = np.nan

    return le, re, float(avg), float(asym)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------
class FusionCEWNTHU:
    def __init__(self, base_dir=None, cew_model_path=None, input_size=(80, 80)):
        """
        base_dir: project root, if None we infer from this file
        cew_model_path: explicit .keras path; if None, we try:
            models/cew_best.keras, then models/cew_final.keras
        input_size: (H, W) for CEW model input
        """
        if base_dir is None:
            base_dir = ROOT
        self.base_dir = base_dir
        self.input_size = input_size

        # Decide model path(s)
        self.candidate_paths = []
        if cew_model_path:
            self.candidate_paths.append(cew_model_path)
        # default candidates
        self.candidate_paths.append(os.path.join(self.base_dir, "models", "cew_best.keras"))
        self.candidate_paths.append(os.path.join(self.base_dir, "models", "cew_final.keras"))

        self.cew_model = None
        self.cew_model_path = None

        if TF_AVAILABLE:
            for p in self.candidate_paths:
                if os.path.exists(p):
                    try:
                        self.cew_model = load_model(p, compile=False)
                        self.cew_model_path = p
                        print(f"✅ CEW model loaded from: {p}")
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load CEW model at {p}: {e}")
        else:
            print("⚠️ TensorFlow/Keras not available, CEW model cannot be loaded.")

        if self.cew_model is None:
            print("⚠️ No CEW model loaded. Will use simple brightness heuristic.")

    # --------------------------------------------------------------
    # Core API: eye-state prediction from eye crop
    # --------------------------------------------------------------
    def predict_eye_state(self, eye_img):
        """
        Given an eye crop image (BGR or gray numpy array),
        returns (p_open, p_closed) in [0,1].
        """
        # If we have a real CEW model, use it
        if self.cew_model is not None:
            x = _preprocess_eye_for_cew(eye_img, target_size=self.input_size)
            if x is None:
                return 0.5, 0.5
            try:
                out = self.cew_model.predict(x, verbose=0)
                # Handle various CEW output formats
                if out.ndim == 2 and out.shape[1] == 2:
                    # assume [closed, open] or [open, closed]; we have to choose a convention
                    # We'll assume index 1 = "open". You can swap if needed.
                    p_open = float(out[0, 1])
                else:
                    p_open = float(out[0, 0])
                    if p_open < 0.0 or p_open > 1.0:
                        p_open = _safe_sigmoid(p_open)
                p_open = float(np.clip(p_open, 0.0, 1.0))
                return p_open, 1.0 - p_open
            except Exception as e:
                print(f"⚠️ CEW inference error, falling back to heuristic: {e}")

        # Fallback: simple brightness-based heuristic
        if eye_img is None:
            return 0.5, 0.5
        try:
            if eye_img.ndim == 2:
                gray = eye_img
            else:
                gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray) / 255.0)
            # Map brightness -> probability of open
            # threshold roughly around ~0.4–0.5
            p_open = float(np.clip((mean_brightness - 0.35) / (0.65 - 0.35), 0.0, 1.0))
            return p_open, 1.0 - p_open
        except Exception:
            return 0.5, 0.5

    # --------------------------------------------------------------
    # Extended API: used by your old notebook test
    # --------------------------------------------------------------
    def predict_from_eye_and_features(self, eye_img, features: dict):
        """
        Combines:
          - CEW eye-state probs from eye_img
          - Basic EAR metrics from features dict
        so your old notebooks/test_fusion_cew_nthu.py still works.

        features can contain:
          - 'left_ear'
          - 'right_ear'
          - 'avg_ear'
          - 'ear_asymmetry'
          - 'blink'
          - etc.
        """
        if features is None:
            features = {}

        result = {}

        # 1) CEW probabilities
        p_open, p_closed = self.predict_eye_state(eye_img)
        result["p_open"] = float(p_open)
        result["p_closed"] = float(p_closed)
        result["cew_model_used"] = self.cew_model_path

        # 2) EAR metrics (derived from features if available)
        left_ear = features.get("left_ear")
        right_ear = features.get("right_ear")
        avg_from_features = features.get("avg_ear")

        le, re, avg, asym = _compute_ear_from_values(left_ear, right_ear)

        # If avg_ear in features and le/re are None, prefer that
        if np.isnan(avg) and avg_from_features is not None:
            avg = float(avg_from_features)

        result["left_ear"] = None if np.isnan(le) else float(le)
        result["right_ear"] = None if np.isnan(re) else float(re)
        result["avg_ear"] = None if np.isnan(avg) else float(avg)
        result["ear_asymmetry"] = None if np.isnan(asym) else float(asym)

        # 3) Pass-through some original fields if you care
        for key in ["participant_id", "frame_id", "task_name", "source_file"]:
            if key in features:
                result[key] = features[key]

        # 4) Optionally copy blink flag/prob if present
        for key in ["blink", "blink_prob", "blink_pred"]:
            if key in features:
                try:
                    result[key] = float(features[key])
                except Exception:
                    result[key] = features[key]

        return result
