# src/webcam_fusion_fixed.py

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import sys
import os
import math
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning , module="sklearn")   

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.fusion_cew_nthu import FusionCEWNTHU
    print("✅ Fusion engine imported successfully")
except Exception as e:
    print(f"❌ Fusion engine import failed: {e}")
    sys.exit(1)


class RealTimeFusionDetector:
    def __init__(self):
        print("🚀 Initializing Real-Time Fusion Detector...")

        # Base dir (project root)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Initialize fusion engine
        self.fusion_engine = FusionCEWNTHU()

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Eye landmark indices (same as preprocess)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                                 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249,
                                  263, 466, 388, 387, 386, 385, 384, 398]

        # Temporal buffers
        self.ear_buffer = deque(maxlen=30)
        self.blink_buffer = deque(maxlen=150)
        self.gaze_history = deque(maxlen=30)  # for attention metrics
        self.frame_count = 0

        # Attention model stuff
        self.attention_enabled = False
        self.attention_model = None
        self.attention_scaler = None
        self.attention_features = [
            "avg_ear",
            "left_ear",
            "right_ear",
            "ear_asymmetry",
            "gaze_x",
            "gaze_y",
            "gaze_angle",
            "gaze_magnitude",
            "gaze_velocity",
            "gaze_entropy",
            "fixation_stability",
        ]
        self._load_attention_model()

        print("✅ Real-Time Fusion Detector Ready!")

    # ---------------- ATTENTION MODEL LOADING ----------------

    def _load_attention_model(self):
        try:
            models_dir = os.path.join(self.base_dir, "outputs", "models_mpiigaze")
            model_path = os.path.join(models_dir, "attention_xgb.pkl")
            scaler_path = os.path.join(models_dir, "attention_scaler.pkl")
            features_path = os.path.join(models_dir, "attention_features.txt")

            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                print("⚠️ Attention model artifacts not found, attention disabled.")
                self.attention_enabled = False
                return

            self.attention_model = joblib.load(model_path)
            self.attention_scaler = joblib.load(scaler_path)

            if os.path.exists(features_path):
                with open(features_path, "r") as f:
                    self.attention_features = [line.strip() for line in f if line.strip()]

            self.attention_enabled = True
            print("🧠 Attention model loaded successfully!")

        except Exception as e:
            print(f"⚠️ Failed to load attention model: {e}")
            self.attention_enabled = False

    # ---------------- LOW-LEVEL METRICS ----------------

    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        try:
            if eye_points.shape[0] < 6:
                return 0.25

            p0, p1, p2, p3, p4, p5 = (
                eye_points[0],
                eye_points[1],
                eye_points[2],
                eye_points[3],
                eye_points[4],
                eye_points[5],
            )

            A = np.linalg.norm(p1 - p5)
            B = np.linalg.norm(p2 - p4)
            C = np.linalg.norm(p0 - p3)

            if C == 0:
                return 0.25

            ear = (A + B) / (2.0 * C)
            return float(np.clip(ear, 0.1, 0.4))
        except Exception:
            return 0.25

    def extract_eye_region(self, frame, eye_landmarks):
        """Extract eye ROI from frame"""
        try:
            h, w = frame.shape[:2]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in eye_landmarks]

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min, x_max = max(0, min(x_coords) - 10), min(w, max(x_coords) + 10)
            y_min, y_max = max(0, min(y_coords) - 10), min(h, max(y_coords) + 10)

            eye_roi = frame[y_min:y_max, x_min:x_max]
            if eye_roi.size == 0:
                return None

            return eye_roi
        except Exception:
            return None

    # ---------------- ATTENTION FEATURE PIPELINE ----------------

    def _update_gaze_and_attention_features(self, left_eye_landmarks, right_eye_landmarks, frame_shape):
        """Compute gaze_x/y, gaze_velocity, entropy, fixation_stability."""
        h, w = frame_shape[:2]

        # Eye centers in pixel space (like preprocess)
        left_eye_px = np.array([[lm.x * w, lm.y * h] for lm in left_eye_landmarks])
        right_eye_px = np.array([[lm.x * w, lm.y * h] for lm in right_eye_landmarks])

        left_center = np.mean(left_eye_px, axis=0)
        right_center = np.mean(right_eye_px, axis=0)
        avg_center = (left_center + right_center) / 2.0

        # Normalize a bit to match training magnitude
        gaze_x = float(avg_center[0] / 60.0)
        gaze_y = float(avg_center[1] / 60.0)

        angle = math.atan2(gaze_y, gaze_x)
        magnitude = float(np.linalg.norm([gaze_x, gaze_y]))

        # Append to history for temporal stats
        gaze_point = {"gaze_x": gaze_x, "gaze_y": gaze_y}
        self.gaze_history.append(gaze_point)

        # Default values
        gaze_velocity = 0.0
        gaze_entropy = 0.0
        fixation_stability = 0.5

        if len(self.gaze_history) > 1:
            # Velocity over history
            velocities = []
            for i in range(1, len(self.gaze_history)):
                dx = self.gaze_history[i]["gaze_x"] - self.gaze_history[i - 1]["gaze_x"]
                dy = self.gaze_history[i]["gaze_y"] - self.gaze_history[i - 1]["gaze_y"]
                velocities.append(math.sqrt(dx * dx + dy * dy))

            gaze_velocity = float(np.mean(velocities)) if velocities else 0.0

            # Spread (entropy proxy)
            points = np.array([[g["gaze_x"], g["gaze_y"]] for g in self.gaze_history])
            gx_std = float(np.std(points[:, 0]))
            gy_std = float(np.std(points[:, 1]))
            gaze_entropy = float(math.sqrt(gx_std * gx_std + gy_std * gy_std))

            # Fixation stability similar to preprocess
            velocity_norm = 1.0 / (1.0 + gaze_velocity * 3.5)
            fixation_stability = float(velocity_norm)

        gaze_features = {
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "gaze_angle": float(angle),
            "gaze_magnitude": magnitude,
            "gaze_velocity": gaze_velocity,
            "gaze_entropy": gaze_entropy,
            "fixation_stability": fixation_stability,
        }

        return gaze_features

    def _predict_attention(self, feature_pack):
        """Run attention model on current features."""
        if not self.attention_enabled:
            return None, None, None

        try:
            x = np.array(
                [[feature_pack[col] for col in self.attention_features]],
                dtype=float,
            )
            x_scaled = self.attention_scaler.transform(x)
            att = float(self.attention_model.predict(x_scaled)[0])
            att = float(np.clip(att, 0.0, 1.0))

            # Categorize
            if att < 0.35:
                level = "LOW"
                mind_wandering = True
            elif att < 0.6:
                level = "MEDIUM"
                mind_wandering = False
            else:
                level = "HIGH"
                mind_wandering = False

            return att, level, mind_wandering

        except Exception as e:
            print(f"⚠️ Attention prediction error: {e}")
            return None, None, None

    # ---------------- MAIN FRAME PROCESSING ----------------

    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None, frame

            face_landmarks = results.multi_face_landmarks[0]

            # Extract eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[i] for i in self.LEFT_EYE_INDICES]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in self.RIGHT_EYE_INDICES]

            # EAR calculation
            left_pts = np.array([[lm.x, lm.y] for lm in left_eye_landmarks], dtype=np.float32)
            right_pts = np.array([[lm.x, lm.y] for lm in right_eye_landmarks], dtype=np.float32)

            left_ear = self.calculate_ear(left_pts)
            right_ear = self.calculate_ear(right_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            # Update buffers
            self.ear_buffer.append(avg_ear)
            ear_threshold = 0.2
            is_blink = 1 if avg_ear < ear_threshold else 0
            self.blink_buffer.append(is_blink)

            # Blink frequency
            if len(self.blink_buffer) > 1:
                blinks = sum(
                    1
                    for i in range(1, len(self.blink_buffer))
                    if self.blink_buffer[i - 1] == 0 and self.blink_buffer[i] == 1
                )
                buffer_duration = len(self.blink_buffer) / 30.0
                blink_frequency = (blinks / buffer_duration) * 60.0 if buffer_duration > 0 else 0.0
            else:
                blink_frequency = 0.0

            # Extract eye region
            left_eye_roi = self.extract_eye_region(frame, left_eye_landmarks)
            right_eye_roi = self.extract_eye_region(frame, right_eye_landmarks)
            eye_roi = left_eye_roi if left_eye_roi is not None else right_eye_roi

            if eye_roi is None:
                return None, frame

            # Build features for drowsiness fusion (existing)
            features = {
                "participant_id": 1,
                "frame_id": self.frame_count,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "avg_ear": avg_ear,
                "blink": 0,  # Simplified
                "ear_asymmetry": float(abs(left_ear - right_ear)),
            }

            # ---------- NEW: ATTENTION FEATURES ----------
            gaze_features = self._update_gaze_and_attention_features(
                left_eye_landmarks, right_eye_landmarks, frame.shape
            )

            # Merge EAR + gaze into single feature pack for attention model
            attention_features_pack = {
                "avg_ear": avg_ear,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "ear_asymmetry": float(abs(left_ear - right_ear)),
                **gaze_features,
            }

            attention_score, attention_level, mind_wandering = self._predict_attention(
                attention_features_pack
            )

            # Fusion prediction (fatigue/drowsiness)
            fusion = self.fusion_engine.predict_from_eye_and_features(eye_roi, features)
            fusion["avg_ear"] = float(avg_ear)
            fusion["blink_frequency"] = float(blink_frequency)

            # Attach attention outputs
            if attention_score is not None:
                fusion["attention_score"] = attention_score
                fusion["attention_level"] = attention_level
                fusion["mind_wandering"] = mind_wandering

            return fusion, frame

        except Exception as e:
            if "attention_features" not in str(e):
                print(f"⚠️ Frame processing error: {e}")
            return None, frame

    # ---------------- DRAWING ----------------

    def draw_results(self, frame, result):
        """Draw overlay for current fusion result"""
        h, w = frame.shape[:2]

        color_map = {
            "🟢 ALERT": (0, 255, 0),
            "🟡 SLIGHT FATIGUE": (0, 255, 255),
            "🟠 DROWSY": (0, 165, 255),
            "🔴 CRITICAL": (0, 0, 255),
        }
        color = color_map.get(result["state_label"], (255, 255, 255))

        cv2.putText(
            frame,
            result["state_label"],
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Metrics
        metrics = [
            f"Fatigue: {result['fatigue_score']:.3f}",
            f"Eye Open: {result['eye_open_prob']:.3f}",
            f"Drowsy Prob: {result['drowsy_prob']:.3f}",
            f"EAR: {result.get('avg_ear', 0.0):.3f}",
            f"Frame: {self.frame_count}",
        ]

        # Add attention metrics if present
        if "attention_score" in result:
            metrics.append(f"Attention: {result['attention_score']:.3f}")
            metrics.append(f"Attn Level: {result.get('attention_level', 'N/A')}")
            if result.get("mind_wandering", False):
                metrics.append("Mind Wandering: YES")
            else:
                metrics.append("Mind Wandering: NO")

        for i, text in enumerate(metrics):
            cv2.putText(
                frame,
                text,
                (20, 90 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return frame

    # ---------------- WEBCAM LOOP ----------------

    def run_webcam(self):
        """Run real-time webcam detection"""
        print("📹 Starting webcam on camera index 0...")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot access webcam at index 0")
            return

        print("✅ Webcam started successfully!")
        print("🎮 Controls: Press 'q' to quit, 'r' to reset buffers")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame from webcam")
                break

            result, processed_frame = self.process_frame(frame)

            if result is not None:
                processed_frame = self.draw_results(processed_frame, result)
                if self.frame_count % 30 == 0:
                    att = result.get("attention_score", None)
                    if att is not None:
                        print(
                            f"📊 Frame {self.frame_count}: {result['state_label']} | "
                            f"Fatigue: {result['fatigue_score']:.3f} | "
                            f"Attention: {att:.3f} ({result.get('attention_level')})"
                        )
                    else:
                        print(
                            f"📊 Frame {self.frame_count}: {result['state_label']} "
                            f"(Fatigue: {result['fatigue_score']:.3f})"
                        )
            else:
                cv2.putText(
                    processed_frame,
                    "No face detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Fusion Drowsiness + Attention Detection", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.ear_buffer.clear()
                self.blink_buffer.clear()
                self.gaze_history.clear()
                print("🔄 Buffers reset")

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Webcam stopped")


def main():
    print("=" * 60)
    print("🧠 REAL-TIME FUSION DETECTOR + ATTENTION (MPIIGAZE)")
    print("=" * 60)

    try:
        detector = RealTimeFusionDetector()
        detector.run_webcam()
    except Exception as e:
        print(f"💥 Critical error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()