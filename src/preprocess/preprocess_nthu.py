# src/preprocess/preprocess_nthu_optimized.py
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import re

class OptimizedNTHUPreprocessor:
    def __init__(self, frame_interval=3):
        self.NTHU_PATH = Path(r"C:\Users\Shail\Downloads\neo_cognition\data\NTHU_ DDD")

        self.OUT_CSV = Path(r"C:\Users\Shail\Downloads\neo_cognition\outputs\nthu_features_optimized.csv")
        self.FRAME_INTERVAL = frame_interval
        self.EAR_THRESHOLD = 0.20
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def extract_frame_number(self, filename):
        match = re.search(r'_(\d+)_(drowsy|notdrowsy)\.jpg$', filename)
        if match:
            return int(match.group(1))
        return 0

    def group_files_by_sequence(self, file_list):
        sequences = {}
        
        for file_path in file_list:
            filename = file_path.name
            parts = filename.replace('.jpg', '').split('_')
            
            if len(parts) >= 4:
                seq_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
                frame_num = self.extract_frame_number(filename)
                
                if seq_key not in sequences:
                    sequences[seq_key] = []
                
                sequences[seq_key].append((frame_num, file_path))
        
        for key in sequences:
            sequences[key].sort(key=lambda x: x[0])
        
        return sequences

    def select_frames_from_sequence(self, sequence, interval):
        if not sequence:
            return []
        
        selected = [sequence[0][1]] 
        
        for i in range(interval, len(sequence) - 1, interval):
            selected.append(sequence[i][1])
        
        if sequence[-1][1] not in selected:
            selected.append(sequence[-1][1])
        
        return selected

    def get_optimized_file_list(self, folder_path, label):
        folder = self.NTHU_PATH / folder_path
        if not folder.exists():
            print(f"⚠️ Folder missing: {folder}")
            return []
        
        all_files = [f for f in folder.iterdir() if f.name.lower().endswith('.jpg')]
        print(f"📁 Found {len(all_files)} images in {folder_path}")
        
        sequences = self.group_files_by_sequence(all_files)
        print(f"   Grouped into {len(sequences)} sequences")
        
        selected_files = []
        for seq_key, sequence in sequences.items():
            seq_files = self.select_frames_from_sequence(sequence, self.FRAME_INTERVAL)
            selected_files.extend(seq_files)
            print(f"   {seq_key}: {len(sequence)} → {len(seq_files)} frames")
        
        print(f"🎯 Selected {len(selected_files)}/{len(all_files)} images ({len(selected_files)/len(all_files):.1%})")
        return selected_files

    def calculate_ear(self, lm, eye):
        pts = np.array([[lm.landmark[i].x, lm.landmark[i].y] for i in eye])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def parse_filename(self, filename):
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) < 5:
            return {}
        return {
            "participant_id": parts[0],
            "condition": parts[1],
            "behavior": parts[2],
            "frame_id": parts[3],
            "name_label": parts[4]
        }

    def extract_visual_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {"face_detected": False}
        
        lm = results.multi_face_landmarks[0]
        left = self.calculate_ear(lm, self.LEFT_EYE)
        right = self.calculate_ear(lm, self.RIGHT_EYE)
        avg = (left + right) / 2
        
        return {
            "face_detected": True,
            "left_ear": left,
            "right_ear": right,
            "avg_ear": avg,
            "blink": 1 if avg < self.EAR_THRESHOLD else 0,
            "ear_asymmetry": abs(left - right)
        }

    def process_optimized_directory(self, subdir, label):
        selected_files = self.get_optimized_file_list(subdir, label)
        rows = []
        
        for i, file in enumerate(selected_files):
            if i % 500 == 0:
                print(f"🔄 {label}: {i}/{len(selected_files)}")
            
            meta = self.parse_filename(file.name)
            feats = self.extract_visual_features(str(file))
            
            if feats:
                rows.append({
                    "image_path": str(file),
                    "filename": file.name,
                    "label": label,
                    **meta,
                    **feats
                })
        
        return rows

    def run_optimized(self):
        print("🚀 Starting OPTIMIZED NTHU Processing (Keep Every 3rd Frame)")
        
        self.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        
        drowsy_data = self.process_optimized_directory("drowsy", "drowsy")
        notdrowsy_data = self.process_optimized_directory("notdrowsy", "notdrowsy")
        
        data = drowsy_data + notdrowsy_data
        df = pd.DataFrame(data)
        
        print(f"\n📊 FINAL OPTIMIZED DATASET:")
        print(f"Total images processed: {len(df)}")
        print(f"Reduction: {len(df)/66521:.1%}")
        print(f"Class balance: {df['label'].value_counts().to_dict()}")
        
        df.to_csv(self.OUT_CSV, index=False)
        print(f"💾 Saved optimized dataset: {self.OUT_CSV}")
        
        return df


def main():
    print("✅ Using Every 3rd Frame Strategy")
    processor = OptimizedNTHUPreprocessor(frame_interval=3)
    df = processor.run_optimized()
    
    if df is not None:
        print("\n🎉 OPTIMIZED DATASET READY!")
        print(df[["filename", "avg_ear", "blink", "label"]].head(10))


if __name__ == "__main__":
    main()
