import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import json
from datetime import datetime

BASE_DIR = r"C:\Users\Shail\Downloads\neo_cognition"
DATASET_DIR = os.path.join(BASE_DIR, "data", "eyeblink8")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eyeblink8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "eyeblink8_processed.csv")
METADATA_JSON = os.path.join(OUTPUT_DIR, "processing_metadata.json")
VIDEO_METADATA_JSON = os.path.join(OUTPUT_DIR, "video_metadata.json")

# ================= SETTINGS =================
FRAME_SKIP = 1                  # MAX TEMPORAL RESOLUTION
MAX_CONSECUTIVE_FAILED_FRAMES = 50
BLINK_CONTEXT_SECONDS = 3.0     # WIDE CONTEXT ✅
EAR_MAX_THRESHOLD = 1.2         # Remove unrealistic spikes
# ============================================

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249]


def calculate_ear(eye_points):
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = (
        eye_points[1], eye_points[2], eye_points[3],
        eye_points[4], eye_points[5], eye_points[0]
    )
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)

    if C == 0:
        return 0.0

    return float((A + B) / (2.0 * C))


def load_blink_labels(txt_path):
    ranges = []
    try:
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ranges.append((float(parts[0]), float(parts[1])))
    except:
        pass
    return ranges


def load_tag_metadata(tag_path):
    try:
        with open(tag_path) as f:
            return f.read().strip()
    except:
        return "N/A"


def is_within_blink_context(timestamp, blink_ranges):
    for start, end in blink_ranges:
        if (start - BLINK_CONTEXT_SECONDS) <= timestamp <= (end + BLINK_CONTEXT_SECONDS):
            return True
    return False


def process_video(video_path, txt_path, tag_path, subject_id, video_metadata):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    blink_ranges = load_blink_labels(txt_path)
    if not blink_ranges:
        print(f"⚠️ No blink labels in {video_path}")
        return []

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"📊 {video_id} | Blink events: {len(blink_ranges)}")

    video_metadata[video_id] = {
        "subject": subject_id,
        "tag_content": load_tag_metadata(tag_path)
    }

    results = []
    failed = 0

    for i in tqdm(range(total_frames), desc=video_id):
        ret, frame = cap.read()

        if not ret:
            failed += 1
            if failed >= MAX_CONSECUTIVE_FAILED_FRAMES:
                break
            continue

        failed = 0

        if i % FRAME_SKIP != 0:
            continue

        timestamp = i / fps
        blink = 1 if any(s <= timestamp <= e for s, e in blink_ranges) else 0

        if not (blink or is_within_blink_context(timestamp, blink_ranges)):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_eye = np.array([[lm[x].x * w, lm[x].y * h] for x in LEFT_EYE])
        right_eye = np.array([[lm[x].x * w, lm[x].y * h] for x in RIGHT_EYE])

        le = calculate_ear(left_eye)
        re = calculate_ear(right_eye)
        avg = (le + re) / 2

        # ✅ Filter unrealistic EAR spikes
        if avg > EAR_MAX_THRESHOLD or avg <= 0:
            continue

        results.append({
            "subject": subject_id,
            "video": video_id,
            "frame": i + 1,
            "timestamp": round(timestamp, 4),
            "left_ear": le,
            "right_ear": re,
            "avg_ear": avg,
            "blink": blink,
            "face_detected": True
        })

    cap.release()
    return results


def main():
    print("🚀 PROCESSING EYEBLINK8 — FINAL CLEANED HYBRID MODE")
    print("=" * 60)

    all_data = []
    video_metadata = {}

    subjects = ['1', '2', '3', '4', '8', '9', '10', '11']

    for subject in subjects:
        folder = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(folder):
            continue

        for video in os.listdir(folder):
            if video.endswith(".avi"):
                base = video.replace(".avi", "")
                all_data.extend(process_video(
                    os.path.join(folder, video),
                    os.path.join(folder, base + ".txt"),
                    os.path.join(folder, base + ".tag"),
                    subject,
                    video_metadata
                ))

    df = pd.DataFrame(all_data)

    # ✅ Remove bad videos (zero blink or invalid EAR)
    initial_videos = df["video"].nunique()
    df = df.groupby("video").filter(
        lambda x: x["blink"].sum() > 0 and x["avg_ear"].mean() > 0
    )

    print(f"🧹 Removed {initial_videos - df['video'].nunique()} bad videos")

    df.to_csv(OUTPUT_CSV, index=False)

    with open(METADATA_JSON, "w") as f:
        json.dump({
            "processed_at": datetime.now().isoformat(),
            "frames": len(df),
            "blinks": int(df["blink"].sum()),
            "videos": int(df["video"].nunique())
        }, f, indent=2)

    with open(VIDEO_METADATA_JSON, "w") as f:
        json.dump(video_metadata, f, indent=2)

    print("\n✅ FINAL DATASET READY")
    print(f"Frames: {len(df)}")
    print(f"Blinks: {df['blink'].sum()}")
    print(f"Videos: {df['video'].nunique()}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
