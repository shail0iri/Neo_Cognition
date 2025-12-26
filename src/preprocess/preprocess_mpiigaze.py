import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import math

class MPIIGazeProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mpiigaze_dir = os.path.join(base_dir, "data", "MPIIGAZE")
        self.output_dir = os.path.join(base_dir, "outputs", "MPIIGAZE")
        os.makedirs(self.output_dir, exist_ok=True)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE_INDICES = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        self.RIGHT_EYE_INDICES = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]

    def calculate_ear(self, eye_points):
        if len(eye_points) < 6:
            return 0.25
        p1, p2, p3, p4, p5, p6 = eye_points[1], eye_points[2], eye_points[3], eye_points[4], eye_points[5], eye_points[0]
        A = np.linalg.norm(p2 - p3)
        B = np.linalg.norm(p1 - p4)
        C = np.linalg.norm(p5 - p6)
        return float(np.clip((A + B) / (2.0 * C), 0.15, 0.45)) if C != 0 else 0.25

    def estimate_gaze_direction(self, landmarks, w, h):
        try:
            left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE_INDICES])
            right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE_INDICES])

            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
            avg_gaze = (left_center + right_center) / 2.0

            gaze_x = float(avg_gaze[0] / 60.0)
            gaze_y = float(avg_gaze[1] / 60.0)

            angle = math.atan2(gaze_y, gaze_x)
            magnitude = np.linalg.norm([gaze_x, gaze_y])

            return {
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "gaze_angle": float(angle),
                "gaze_magnitude": float(magnitude)
            }
        except:
            return {"gaze_x":0.0,"gaze_y":0.0,"gaze_angle":0.0,"gaze_magnitude":0.0}

    def calculate_attention_metrics(self, gaze_history):
        if len(gaze_history) < 5:
            return {
                "attention_score": 0.5,
                "gaze_velocity": 0.0,
                "gaze_entropy": 0.0,
                "fixation_stability": 0.5
            }

        velocities = []
        for i in range(1, len(gaze_history)):
            dx = gaze_history[i]['gaze_x'] - gaze_history[i-1]['gaze_x']
            dy = gaze_history[i]['gaze_y'] - gaze_history[i-1]['gaze_y']
            velocities.append(math.sqrt(dx**2 + dy**2))

        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        gaze_points = np.array([[g['gaze_x'], g['gaze_y']] for g in gaze_history])

        gx_std = np.std(gaze_points[:,0])
        gy_std = np.std(gaze_points[:,1])
        gaze_spread = float(np.sqrt(gx_std**2 + gy_std**2))

        velocity_norm = 1.0 / (1.0 + avg_velocity * 3.5)
        spread_norm = 1.0 / (1.0 + gaze_spread * 2.5)

        raw_score = 0.55 * velocity_norm + 0.45 * spread_norm

        sigmoid = 1 / (1 + np.exp(-6 * (raw_score - 0.45)))
        attention_score = 0.15 + sigmoid * 0.8
        attention_score = float(np.clip(attention_score, 0.15, 0.95))

        fixation_stability = velocity_norm

        return {
            "attention_score": attention_score,
            "gaze_velocity": avg_velocity,
            "gaze_entropy": gaze_spread,
            "fixation_stability": fixation_stability
        }

    def process_dataset(self):
        print("🚀 Processing MPIIGAZE with STABLE Attention Mapping...")

        all_data = []
        gaze_history = []
        processed_count = 0
        error_count = 0

        for subject in os.listdir(self.mpiigaze_dir):
            if not subject.startswith('p'):
                continue
            subject_path = os.path.join(self.mpiigaze_dir, subject)

            for day in os.listdir(subject_path):
                if not day.startswith('day'):
                    continue
                day_path = os.path.join(subject_path, day)
                images = sorted([f for f in os.listdir(day_path) if f.lower().endswith('.jpg')])

                for img in tqdm(images, desc=f"{subject}/{day}"):
                    try:
                        frame = cv2.imread(os.path.join(day_path, img))
                        if frame is None:
                            error_count += 1
                            continue
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.face_mesh.process(rgb)
                        if not results.multi_face_landmarks:
                            error_count += 1
                            continue
                        landmarks = results.multi_face_landmarks[0].landmark
                        h, w = frame.shape[:2]

                        left_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE_INDICES])
                        right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE_INDICES])

                        left_ear = self.calculate_ear(left_pts)
                        right_ear = self.calculate_ear(right_pts)
                        avg_ear = (left_ear + right_ear) / 2

                        gaze_data = self.estimate_gaze_direction(landmarks, w, h)
                        gaze_history.append(gaze_data)
                        if len(gaze_history) > 30:
                            gaze_history.pop(0)

                        attention = self.calculate_attention_metrics(gaze_history)

                        all_data.append({
                            "subject": subject,
                            "day": day,
                            "image": img,
                            "avg_ear": avg_ear,
                            "left_ear": left_ear,
                            "right_ear": right_ear,
                            "ear_asymmetry": abs(left_ear - right_ear),
                            **gaze_data,
                            **attention,
                            "eyes_open": avg_ear > 0.3
                        })

                        processed_count += 1
                    except:
                        error_count += 1

        df = pd.DataFrame(all_data)
        output_csv = os.path.join(self.output_dir, "mpiigaze_features.csv")
        df.to_csv(output_csv, index=False)

        print("\n✅ PROCESSING COMPLETE")
        print(f"Samples: {processed_count}")
        print(f"Errors: {error_count}")
        print(f"Mean attention: {df['attention_score'].mean():.3f}")
        print(f"Std attention: {df['attention_score'].std():.3f}")
        print(f">0.7: {(df['attention_score']>0.7).mean():.2%}")
        print(f"<0.3: {(df['attention_score']<0.3).mean():.2%}")
        print("Saved:", output_csv)


def main():
    base_dir = r"C:\Users\Shail\Downloads\neo_cognition"
    MPIIGazeProcessor(base_dir).process_dataset()


if __name__ == "__main__":
    main()
