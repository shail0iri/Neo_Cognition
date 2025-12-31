# Real-Time Drowsiness and Cognitive State Detection with Fusion Model

from email.mime import text
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
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")   

# ==================== IMPORT PATHS ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  

sys.path.insert(0, project_root)

# Try multiple import paths for FusionCEWNTHU
fusion_engine = None
import_attempts = [
    "src.fusion_cew_nthu",  # From src folder
    "fusion_cew_nthu",      # Direct import
    "models.fusion_cew_nthu",  # From models folder
]

for module_path in import_attempts:
    try:
        module = __import__(module_path, fromlist=['FusionCEWNTHU'])
        fusion_engine = getattr(module, 'FusionCEWNTHU')
        print(f"✅ Fusion engine imported from: {module_path}")
        break
    except ImportError:
        continue

if fusion_engine is None:
    print(" Creating dummy FusionCEWNTHU class")
    class FusionCEWNTHU:
        def predict_from_eye_and_features(self, eye_roi, features):
            return {
                "state_label": "🟢 ALERT",
                "fatigue_score": 0.1,
                "eye_open_prob": 0.9,
                "drowsy_prob": 0.1,
            }
    fusion_engine = FusionCEWNTHU


# ==================== CONFIG ====================
DEFAULT_EAR_THRESHOLD = 0.20
MIN_BLINK_DURATION = 0.1
MAX_BLINK_DURATION = 0.4  # Normal blinks are 100-400ms

# Eye landmarks
LEFT_EYE_EAR = {
    "p1": 33, "p4": 133,
    "p2": 159, "p6": 145,
    "p3": 158, "p5": 153
}
RIGHT_EYE_EAR = {
    "p1": 362, "p4": 263,
    "p2": 386, "p6": 374,
    "p3": 385, "p5": 380
}


class RealTimeFusionDetector:
    def __init__(self):
        print(" Initializing Real-Time Fusion Detector...")

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

        # ========== BLINK DETECTION STATE ==========
        self.ear_threshold = DEFAULT_EAR_THRESHOLD  # SINGLE SOURCE OF TRUTH
        
        # Blink tracking with hysteresis
        self.total_blinks = 0
        self.blink_state = "OPEN"
        self.blink_start_time = 0
        self.blink_timestamps = []  # List of all blink times
        
        # For rate calculation
        self.last_minute_blinks = deque(maxlen=100)
        
        # EAR history for smoothing
        self.ear_history = deque(maxlen=10)
        
        # ========== TEMPORAL BUFFERS ==========
        self.gaze_history = deque(maxlen=30)
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

        print(" Real-Time Fusion Detector Ready!")

    # ---------------- EAR CALCULATION ----------------
    def calculate_ear(self, landmarks, eye, w, h):
        def pt(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h])

        p1, p2, p3 = pt(eye["p1"]), pt(eye["p2"]), pt(eye["p3"])
        p4, p5, p6 = pt(eye["p4"]), pt(eye["p5"]), pt(eye["p6"])

        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        
        return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

    # ---------------- BLINK DETECTION ----------------
    def detect_blink(self, ear_smooth, current_time):
        """Accurate blink detection with single threshold source"""
        blink_detected_now = False
        
        # Use instance variables for thresholds
        close_threshold = self.ear_threshold
        open_threshold = self.ear_threshold + 0.02  # Hysteresis: need to open slightly more
        
        # State machine for blink detection
        if ear_smooth < close_threshold and self.blink_state == "OPEN":
            # Eye just closed - start potential blink
            self.blink_state = "CLOSED"
            self.blink_start_time = current_time
            
        elif ear_smooth > open_threshold and self.blink_state == "CLOSED":
            # Eye just opened - check if it was a valid blink
            blink_duration = current_time - self.blink_start_time
            
            # Validate blink duration
            if MIN_BLINK_DURATION <= blink_duration <= MAX_BLINK_DURATION:
                # VALID BLINK DETECTED!
                self.total_blinks += 1
                self.blink_timestamps.append(current_time)
                self.last_minute_blinks.append(current_time)
                blink_detected_now = True
                print(f"Blink #{self.total_blinks} - Duration: {blink_duration:.3f}s")
            
            # Reset state regardless
            self.blink_state = "OPEN"
        
        return blink_detected_now

    def calculate_blink_rate(self):
        """Calculate blinks per minute from last 60 seconds (time-based)"""
        current_time = time.time()
        
        # Clean old blinks (older than 60 seconds)
        self.blink_timestamps = [t for t in self.blink_timestamps 
                               if current_time - t <= 60.0]
        self.last_minute_blinks = deque(
            [t for t in self.last_minute_blinks if current_time - t <= 60.0],
            maxlen=100
        )
        
        # Use last_minute_blinks for recent rate calculation
        if len(self.last_minute_blinks) >= 2:
            time_window = current_time - min(self.last_minute_blinks)
            if time_window > 5:  # Need at least 5 seconds of data
                rate = (len(self.last_minute_blinks) / time_window) * 60.0
                return max(0.0, min(60.0, rate))
        
        # Fallback: overall rate
        if len(self.blink_timestamps) >= 2:
            total_time = current_time - min(self.blink_timestamps)
            if total_time > 0:
                rate = (len(self.blink_timestamps) / total_time) * 60.0
                return max(0.0, min(60.0, rate))
        
        return 0.0

    # ---------------- ATTENTION MODEL LOADING ----------------
    def _load_attention_model(self):
        try:
            models_dir = os.path.join(self.base_dir, "outputs", "models_mpiigaze")
            model_path = os.path.join(models_dir, "attention_xgb.pkl")
            scaler_path = os.path.join(models_dir, "attention_scaler.pkl")

            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                print("Attention model artifacts not found, attention disabled.")
                self.attention_enabled = False
                return

            self.attention_model = joblib.load(model_path)
            self.attention_scaler = joblib.load(scaler_path)
            self.attention_enabled = True
            print("Attention model loaded successfully!")

        except Exception as e:
            print(f"Failed to load attention model: {e}")
            self.attention_enabled = False

    # ---------------- EYE ROI EXTRACTION ----------------
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

    # ---------------- MAIN FRAME PROCESSING ----------------
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                # Reset blink state when face disappears
                self.blink_state = "OPEN"
                self.blink_start_time = 0
                return None, frame

            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # Calculate EAR using working method
            left_ear = self.calculate_ear(face_landmarks.landmark, LEFT_EYE_EAR, w, h)
            right_ear = self.calculate_ear(face_landmarks.landmark, RIGHT_EYE_EAR, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Smooth the EAR value
            self.ear_history.append(ear)
            if len(self.ear_history) >= 3:
                ear_smooth = np.median(self.ear_history)
            else:
                ear_smooth = ear
            
            # Standardized eye open ratio (0-1)
            if ear_smooth > self.ear_threshold:
                # When eye is open, scale from threshold to 0.3
                eye_open_ratio = np.clip((ear_smooth - 0.15) / 0.15, 0.0, 1.0)
            else:
                # When eye is closed, scale from 0 to threshold
                eye_open_ratio = np.clip(ear_smooth / self.ear_threshold, 0.0, 1.0)

            # Detect blink
            current_time = time.time()
            blink_now = self.detect_blink(ear_smooth, current_time)
            
            # Single blink rate calculation (time-based, NOT frame-based)
            blink_rate = self.calculate_blink_rate()

            # Extract eye landmarks for ROI
            LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
            eye_roi = self.extract_eye_region(frame, left_eye_landmarks)

            if eye_roi is None:
                return None, frame

            # Build features for fusion
            features = {
                "participant_id": 1,
                "frame_id": self.frame_count,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "avg_ear": ear_smooth,
                "blink": 1 if blink_now else 0,
                "ear_asymmetry": float(abs(left_ear - right_ear)),
            }

            # Fusion prediction
            fusion = self.fusion_engine.predict_from_eye_and_features(eye_roi, features)
            
            # Standardized naming
            fusion["avg_ear"] = float(ear_smooth)
            fusion["eye_open_ratio"] = float(eye_open_ratio)  # Consistent naming
            fusion["blink_rate"] = float(blink_rate)  # Single blink rate
            fusion["total_blinks"] = self.total_blinks
            fusion["blink_state"] = self.blink_state
            fusion["ear_threshold"] = float(self.ear_threshold)  # Debug info

            return fusion, frame

        except Exception as e:
            if self.frame_count % 30 == 0:  # Don't spam errors
                print(f"Frame processing error: {e}")
            return None, frame

    # ---------------- DISPLAY ----------------
    def draw_results(self, frame, result):
        """Draw overlay for current fusion result"""
        h, w = frame.shape[:2]

        # Color map for states
        color_map = {
            "🟢 ALERT": (0, 255, 0),
            "🟡 SLIGHT FATIGUE": (0, 255, 255),
            "🟠 DROWSY": (0, 165, 255),
            "🔴 CRITICAL": (0, 0, 255),
        }
        state_label = result.get("state_label", "🟢 ALERT")
        color = color_map.get(state_label, (255, 255, 255))

        # Draw state
        cv2.putText(
            frame,
            state_label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Metrics
        avg_ear = result.get('avg_ear', 0.0)
        ear_threshold = result.get('ear_threshold', self.ear_threshold)
        
        metrics = [
            f"Frame: {self.frame_count}",
            f"EAR: {avg_ear:.3f}",
            f"Eye Open: {result.get('eye_open_ratio', 0.0):.2f}",
            f"Blinks: {result.get('total_blinks', 0)}",
            f"Rate: {result.get('blink_rate', 0.0):.1f}/min",
            f"State: {result.get('blink_state', 'OPEN')}",
            f"Threshold: {ear_threshold:.3f}",
        ]

        # Add fatigue metrics if available
        if 'fatigue_score' in result:
            metrics.insert(1, f"Fatigue: {result['fatigue_score']:.3f}")
        if 'drowsy_prob' in result:
            metrics.insert(2, f"Drowsy: {result['drowsy_prob']:.3f}")

        metric_colors = [
            (200, 200, 200),   # Frame - grey
            (255, 255, 0),     # EAR - cyan
            (255, 200, 0),     # Eye Open - blue/yellow
            (0, 200, 255),     # Blinks - orange
            (255, 100, 255),   # Rate - pink
            (200, 255, 200),   # State - light green
            (200, 200, 100),   # Threshold - muted yellow
        ]
        for i, text in enumerate(metrics):
           color = metric_colors[i] if i < len(metric_colors) else (255, 255, 255)
           cv2.putText(
                frame,
                text,
                (20, 90 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            
        return frame

    # ---------------- WEBCAM LOOP ----------------
    
    def run_webcam(self):
        """Run real-time webcam detection"""
        print("Starting webcam on camera index 0...")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Cannot access webcam at index 0")
            return

        print("Webcam started successfully!")
        print("Controls: Press 'q' to quit, 'r' to reset")
        print("              '+' to increase threshold, '-' to decrease")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame from webcam")
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            result, processed_frame = self.process_frame(frame)

            if result is not None:
                processed_frame = self.draw_results(processed_frame, result)
                
                # Print periodic updates
                if self.frame_count % 30 == 0:
                    print(
                        f"Frame {self.frame_count}: {result.get('state_label', 'N/A')} | "
                        f"EAR: {result.get('avg_ear', 0.0):.3f} | "
                        f"Blinks: {result.get('total_blinks', 0)} | "
                        f"Rate: {result.get('blink_rate', 0.0):.1f}/min"
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

            cv2.imshow("Fusion Drowsiness Detection", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.total_blinks = 0
                self.blink_state = "OPEN"
                self.blink_timestamps.clear()
                self.last_minute_blinks.clear()
                self.ear_history.clear()
                print("All buffers reset")
            elif key == ord("+"):
                self.ear_threshold += 0.01
                self.ear_threshold = min(0.3, self.ear_threshold)
                print(f"Threshold increased to: {self.ear_threshold:.3f}")
            elif key == ord("-"):
                self.ear_threshold -= 0.01
                self.ear_threshold = max(0.15, self.ear_threshold)
                print(f"Threshold decreased to: {self.ear_threshold:.3f}")

        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Blinks Detected: {self.total_blinks}")
        print(f"Final Blink Rate: {self.calculate_blink_rate():.1f}/min")
        print(f"Final EAR Threshold: {self.ear_threshold:.3f}")
        print("Webcam stopped")


def main():
    print("=" * 60)
    print("REAL-TIME FUSION DETECTOR WITH FIXED BLINK DETECTION")
    print("=" * 60)

    try:
        detector = RealTimeFusionDetector()
        detector.run_webcam()
    except KeyboardInterrupt:
        print("\n Stopped by user")
    except Exception as e:
        print(f" Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()