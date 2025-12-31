"""
NEO-COGNITION - WORKING BLINK DETECTION
Fixed blink counter and rate calculation
"""
import csv
import os
from datetime import datetime
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

CSV_PATH = "logs/neo_cognition_session.csv"
CSV_LOG_INTERVAL = 1.0  # seconds


# ==================== CONFIG ====================
EAR_THRESHOLD = 0.20  # Lowered for better sensitivity
EYE_OPEN_THRESHOLD = 0.20
MIN_BLINK_DURATION = 0.1  # 100ms minimum
MAX_BLINK_DURATION = 0.4  # 400ms maximum
WINDOW_NAME = "NeoCognition"

# Correct eye landmarks
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

# ==================== EAR FUNCTION ====================
def calculate_ear(landmarks, eye, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    p1, p2, p3 = pt(eye["p1"]), pt(eye["p2"]), pt(eye["p3"])
    p4, p5, p6 = pt(eye["p4"]), pt(eye["p5"]), pt(eye["p6"])

    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

# ==================== WORKING FACE PROCESSOR ====================
class WorkingFaceProcessor:
    def __init__(self, ear_threshold):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Store threshold as instance variable
        self.ear_threshold = ear_threshold
        
        # Blink tracking 
        self.total_blinks = 0
        self.blink_state = "OPEN"  # OPEN or CLOSED
        self.blink_start_time = 0
        self.blink_timestamps = []  # List of all blink times
        
        # For rate calculation
        self.last_minute_blinks = deque(maxlen=100)
        
        # EAR history for smoothing
        self.ear_history = deque(maxlen=10)  # Shorter buffer for responsiveness
        self.last_features = {
            "avg_ear": 0.25, 
            "eye_open": 1.0, 
            "ear_std": 0.0,
            "blink_state": "OPEN", 
            "ear_raw": 0.25  
        }
        
        # Debug
        self.debug_info = []

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        face_detected = False
        blink_detected_now = False  # Blink detected in THIS frame

        if not res.multi_face_landmarks:
            return face_detected, blink_detected_now, self.last_features

        face_detected = True
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Calculate EAR
        le = calculate_ear(lm, LEFT_EYE_EAR, w, h)
        re = calculate_ear(lm, RIGHT_EYE_EAR, w, h)
        ear = (le + re) / 2.0

        # Store for history
        self.ear_history.append(ear)
        
        # Calculate smoothed EAR (but keep it responsive)
        if len(self.ear_history) >= 3:
            ear_smooth = np.median(self.ear_history)  # Use median for robustness
        else:
            ear_smooth = ear
        
        # Eye open probability
        if ear_smooth > self.ear_threshold:
            eye_open = np.clip((ear_smooth - 0.15) / 0.15, 0.0, 1.0)
        else:
            eye_open = np.clip(ear_smooth / self.ear_threshold, 0.0, 1.0)

        # ===== ROBUST BLINK DETECTION =====
        close_threshold = self.ear_threshold
        open_threshold = close_threshold + 0.008  # Slight hysteresis
        
        current_time = time.time()
        
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
            
            # Reset state regardless
            self.blink_state = "OPEN"
        
        # Store features
        self.last_features = {
            "avg_ear": ear_smooth,
            "eye_open": eye_open,
            "ear_std": np.std(self.ear_history) if len(self.ear_history) > 1 else 0.0,
            "blink_state": self.blink_state,
            "ear_raw": ear  # For debugging
        }

        return face_detected, blink_detected_now, self.last_features

    def calculate_blink_rate(self):
        """Calculate blinks per minute from last 60 seconds"""
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
                return max(0.0, min(60.0, rate))  # Cap at 60 blinks/min
        
        # Fallback: overall rate
        if len(self.blink_timestamps) >= 2:
            total_time = current_time - min(self.blink_timestamps)
            if total_time > 0:
                rate = (len(self.blink_timestamps) / total_time) * 60.0
                return max(0.0, min(60.0, rate))
        
        return 0.0

# ==================== REALISTIC COGNITIVE ESTIMATOR ====================
class RealisticEstimator:
    def __init__(self):
        self.start = time.time()
        self.buffers = {
            "c": deque(maxlen=8),
            "a": deque(maxlen=8),
            "f": deque(maxlen=8),
            "s": deque(maxlen=12),
        }
        
        self.last_values = {
            "cognitive": 0.0,
            "attention": 0.0,
            "fatigue": 0.0,
            "stress": 0.0,
            "session_time": 0.0
        }


    def estimate(self, blink_rate, eye_open, ear_std, face):
        if not face:
            self.last_values["session_time"] = time.time() - self.start
            return self.last_values

        t = time.time() - self.start
        
        # Realistic formulas
        c = 0.3 + min(0.2, t / 1800) + min(0.2, blink_rate / 60)
        c = np.clip(c, 0.25, 0.75)
        
        a = 0.6 + (eye_open * 0.3) - min(0.3, blink_rate / 50)
        a = np.clip(a, 0.4, 0.9)
        
        f = min(0.5, t / 2400) + (1.0 - eye_open) * 0.2
        f = np.clip(f, 0.1, 0.6)
        
        # Realistic stress (15-55%)
        s = 0.25  # Baseline
        
        if blink_rate > 20:
            s += min(0.1, (blink_rate - 20) / 100)
        
        if a < 0.6:
            s += (0.6 - a) * 0.1
        
        if ear_std > 0.02:
            s += min(0.08, ear_std * 3)
        
        if eye_open < 0.5:
            s += (0.5 - eye_open) * 0.15
        
        s = np.clip(s, 0.15, 0.55)
        
        # Update buffers
        for key, val in zip("cafs", [c, a, f, s]):
            self.buffers[key].append(val)
        
        self.last_values = {
            "cognitive": np.mean(self.buffers["c"]) * 100,
            "attention": np.mean(self.buffers["a"]) * 100,
            "fatigue": np.mean(self.buffers["f"]) * 100,
            "stress": np.mean(self.buffers["s"]) * 100,
            "session_time": t
        }
        
        return self.last_values

# ==================== CLEAN DISPLAY ====================
def draw_working(frame, feat, states, face, blink_now, total_blinks, blink_rate, fps, ear_threshold):
    h, w = frame.shape[:2]
    
    # Light overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    
    y = 40
    
    # Title
    cv2.putText(frame, "NEO-COGNITION", (20, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    y += 50
    
    # Eye metrics (use .get() with defaults for safety)
    avg_ear = feat.get('avg_ear', 0.25)
    ear_color = (0, 255, 0) if avg_ear > ear_threshold else (0, 0, 255)
    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)
    y += 35
    
    eye_open = feat.get('eye_open', 1.0)
    cv2.putText(frame, f"Eye Open: {eye_open*100:.1f}%", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    y += 35
    
    # ===== BLINK INFO - WORKING DISPLAY =====
    # Show total blinks and rate
    blink_text = f"Blinks: {total_blinks}"
    cv2.putText(frame, blink_text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2)
    y += 35
    
    rate_text = f"Rate: {blink_rate:.1f}/min"
    cv2.putText(frame, rate_text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2)
    y += 35
    
    # Blink indicator (use .get() for safety)
    if blink_now:
        cv2.putText(frame, "BLINK!", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        y += 35
    elif 'blink_state' in feat and feat['blink_state'] == "CLOSED":
        cv2.putText(frame, "EYE CLOSED", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        y += 35
    
    y += 35
    
    # Cognitive states
    cv2.putText(frame, "Cognitive States:", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
    y += 30
    
    state_info = [
        (f"Load: {states.get('cognitive', 0):.1f}%", (255, 255, 255)),
        (f"Attention: {states.get('attention', 0):.1f}%", (0, 200, 255)),
        (f"Fatigue: {states.get('fatigue', 0):.1f}%", (255, 150, 0)),
        (f"Stress: {states.get('stress', 0):.1f}%", (255, 50, 50))
    ]
    
    for text, color in state_info:
        cv2.putText(frame, text, (40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 30
    
    # Right panel
    status = "TRACKING" if face else " NO FACE"
    status_color = (0, 255, 0) if face else (0, 0, 255)
    
    cv2.putText(frame, status, (w - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 220, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Session time
    mins, secs = divmod(int(states.get('session_time', 0)), 60)
    cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (w - 220, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Threshold info
    cv2.putText(frame, f"Threshold: {ear_threshold:.3f}", (w - 220, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
    
    # Instructions
    cv2.putText(frame, "Q: Quit | R: Reset | D: Debug", (w - 250, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame

def init_csv():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "ear",
                "eye_open",
                "total_blinks",
                "blink_rate",
                "cognitive",
                "attention",
                "fatigue",
                "stress"
            ])


def log_to_csv(feat, states, blink_rate, total_blinks):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            round(feat["avg_ear"], 4),
            round(feat["eye_open"], 3),
            total_blinks,
            round(blink_rate, 2),
            round(states["cognitive"], 1),
            round(states["attention"], 1),
            round(states["fatigue"], 1),
            round(states["stress"], 1)
        ])


# ==================== MAIN ====================
def main():
    init_csv()
    last_csv_log_time = time.time()

    # Use a mutable container for threshold so we can modify it
    current_threshold = EAR_THRESHOLD
    
    print("\n🚀 NEO-COGNITION - WORKING BLINK DETECTION")
    print("=" * 60)
    print("BLINK DETECTION SETTINGS:")
    print(f"• EAR Threshold: {current_threshold}")
    print(f"• Min Blink Duration: {MIN_BLINK_DURATION}s")
    print(f"• Max Blink Duration: {MAX_BLINK_DURATION}s")
    print("=" * 60)
    print("TEST INSTRUCTIONS:")
    print("1. Blink NORMALLY (not too fast)")
    print("2. Watch EAR value - should drop below threshold")
    print("3. Look for 'BLINK!' indicator")
    print("4. Press 'D' for debug info")
    print("5. Press '+' or '-' to adjust threshold")
    print("=" * 60)
    
    proc = WorkingFaceProcessor(current_threshold)
    est = RealisticEstimator()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_buf = deque(maxlen=30)
    last = time.time()
    
    last_blink_count = 0
    
    print("\nStarting monitoring...")
    print("Blink intentionally to test detection!")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1)
        
        face, blink_now, feat = proc.process(frame)
        
        now = time.time()
        fps_buf.append(now - last)
        last = now
        fps = 1 / np.mean(fps_buf) if fps_buf else 0
        
        # Calculate blink rate
        blink_rate = proc.calculate_blink_rate()
        
        # Check if blink count increased
        if proc.total_blinks > last_blink_count:
            print(f"Blink detected! Total: {proc.total_blinks}")
            last_blink_count = proc.total_blinks
        
        # Estimate states
        states = est.estimate(blink_rate, feat["eye_open"], feat["ear_std"], face)
        
        if time.time() - last_csv_log_time >= CSV_LOG_INTERVAL:
            log_to_csv(
                feat=feat,
                states=states,
                blink_rate=blink_rate,
                total_blinks=proc.total_blinks
            )
            last_csv_log_time = time.time()

        # Draw
        frame = draw_working(frame, feat, states, face, blink_now, 
                           proc.total_blinks, blink_rate, fps, current_threshold)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("🔄 Session reset")
            proc = WorkingFaceProcessor(current_threshold)
            est = RealisticEstimator()
            fps_buf.clear()
            last = time.time()
            last_blink_count = 0
        elif key == ord('d'):
            # Debug info
            print(f"\n🔍 DEBUG INFO:")
            print(f"   EAR: {feat['avg_ear']:.3f} (raw: {feat.get('ear_raw', 0):.3f})")
            print(f"   Threshold: {current_threshold}")
            print(f"   Eye State: {feat['blink_state']}")
            print(f"   Eye Open Prob: {feat['eye_open']:.2f}")
            print(f"   Total Blinks: {proc.total_blinks}")
            print(f"   Blink Rate: {blink_rate:.1f}/min")
            print(f"   Blink Timestamps: {len(proc.blink_timestamps)}")
            
            # Suggest threshold adjustment if needed
            if feat['avg_ear'] < 0.15:
                print(f"   ⚠️  EAR very low. Try INCREASING threshold with '+'")
            elif feat['avg_ear'] > 0.35:
                print(f"   ⚠️  EAR high. Try DECREASING threshold with '-'")
        elif key == ord('+'):
            # Increase threshold
            current_threshold += 0.01
            current_threshold = min(0.3, current_threshold)
            print(f"📈 Threshold increased to: {current_threshold:.3f}")
            # Update processor with new threshold
            proc.ear_threshold = current_threshold
        elif key == ord('-'):
            # Decrease threshold
            current_threshold -= 0.01
            current_threshold = max(0.15, current_threshold)
            print(f"📉 Threshold decreased to: {current_threshold:.3f}")
            # Update processor with new threshold
            proc.ear_threshold = current_threshold
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
        # Final stats
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    duration = time.time() - est.start
    
    # Calculate session average rate
    if duration > 0:
        session_rate = (proc.total_blinks / duration) * 60.0
    else:
        session_rate = 0.0
    
    print(f"\n Duration: {duration:.1f}s")
    print(f"Total Blinks Detected: {proc.total_blinks}")
    print(f"Session Average: {session_rate:.1f}/min")
    print(f"Final 60s Rate: {blink_rate:.1f}/min")
    
    if proc.ear_history:
        print(f"EAR Statistics:")
        print(f"   • Average: {np.mean(proc.ear_history):.3f}")
        print(f"   • Minimum: {np.min(proc.ear_history):.3f}")
        print(f"   • Maximum: {np.max(proc.ear_history):.3f}")

    print(f"\nFinal Cognitive States:")
    print(f"   • Load: {states['cognitive']:.1f}%")
    print(f"   • Attention: {states['attention']:.1f}%")
    print(f"   • Fatigue: {states['fatigue']:.1f}%")
    print(f"   • Stress: {states['stress']:.1f}%")
    
    # Blink rate interpretation
        # Blink rate interpretation (use session average)
    if session_rate < 8:
        blink_feedback = "Low blink rate overall (normal: 15-20/min)"
    elif session_rate < 25:
        blink_feedback = "Normal blink rate overall"
    else:
        blink_feedback = "High blink rate overall"
    
    print(f"\n Overall Feedback: {blink_feedback}")
    print("\nSession ended")


if __name__ == "__main__":
    main()