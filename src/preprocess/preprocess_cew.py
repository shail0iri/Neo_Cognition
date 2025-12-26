# src/preprocess/preprocess_cew.py
import os
import cv2
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class CEWFullPreprocessor:
    def __init__(self):
        self.CEW_PATH = Path(r"C:\Users\Shail\Downloads\neo_cognition\data\CEW")

        self.OUT_CSV = Path(r"C:\Users\Shail\Downloads\neo_cognition\outputs\cew_processed_dataset.csv")

        self.PROCESSED_DIR = Path(r"C:\Users\Shail\Downloads\neo_cognition\outputs\CEW_processed")

        self.IMG_SIZE = (80, 80)
        self.RANDOM_SEED = 42

        # Ensure directories exist
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


    # ------------------ IMAGE PROCESSING ------------------
    def enhanced_preprocess(self, image_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            resized = cv2.resize(enhanced, self.IMG_SIZE, interpolation=cv2.INTER_AREA)

            normalized = resized.astype(np.float32) / 255.0

            return normalized

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None


    # ------------------ AUGMENTATION ------------------
    def smart_augment(self, image):
        aug = image.copy()

        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)

        if random.random() < 0.4:
            factor = 0.7 + random.random() * 0.6
            aug = np.clip(aug * factor, 0, 1)

        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            h, w = aug.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if random.random() < 0.25:
            zoom = random.uniform(0.9, 1.1)
            h, w = aug.shape
            zh, zw = int(h * zoom), int(w * zoom)
            zoomed = cv2.resize(aug, (zw, zh))

            sy = max(0, (zh - h) // 2)
            sx = max(0, (zw - w) // 2)
            aug = zoomed[sy:sy + h, sx:sx + w]

        return aug


    # ------------------ DATASET COLLECTION ------------------
    def gather_dataset(self):
        print("📥 Gathering CEW dataset...")

        folder_map = {
            "openlefteyes": ("open", "left"),
            "openrighteyes": ("open", "right"),
            "closedlefteyes": ("closed", "left"),
            "closedrighteyes": ("closed", "right"),
        }

        rows = []

        for folder, (state, side) in folder_map.items():
            f_path = self.CEW_PATH / folder
            if not f_path.exists():
                print(f"⚠️ Missing: {folder}")
                continue

            images = [f for f in f_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]

            print(f"   {folder}: {len(images)} images")

            for img in images:
                rows.append({
                    "image_path": str(img),
                    "eye_state": state,
                    "eye_side": side,
                    "folder": folder,
                    "filename": img.name
                })

        return pd.DataFrame(rows)


    # ------------------ SPLIT DATA ------------------
    def create_balanced_splits(self, df):
        print("🎯 Creating balanced splits...")

        train_val, test = train_test_split(
            df,
            test_size=0.15,
            stratify=df["eye_state"],
            random_state=self.RANDOM_SEED
        )

        train, val = train_test_split(
            train_val,
            test_size=0.176,
            stratify=train_val["eye_state"],
            random_state=self.RANDOM_SEED
        )

        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test


    # ------------------ PROCESS SPLITS ------------------
    def process_and_save_split(self, df, split, augment=False):
        print(f"🔧 Processing {split} set...")

        split_dir = self.PROCESSED_DIR / split
        split_dir.mkdir(exist_ok=True)

        records = []

        for idx, row in df.iterrows():

            img = self.enhanced_preprocess(row["image_path"])
            if img is None:
                continue

            # Save original
            out_file = split_dir / f"{row['eye_state']}_{row['eye_side']}_{idx}_orig.npy"
            np.save(out_file, img)

            records.append({
                "file_path": str(out_file),
                "eye_state": row["eye_state"],
                "eye_side": row["eye_side"],
                "original_path": row["image_path"],
                "split": split,
                "augmented": False,
                "label": 1 if row["eye_state"] == "open" else 0
            })

            # Augment (train only)
            if augment:
                for a in range(2):
                    aug_img = self.smart_augment(img)
                    aug_file = split_dir / f"{row['eye_state']}_{row['eye_side']}_{idx}_aug{a}.npy"
                    np.save(aug_file, aug_img)

                    records.append({
                        "file_path": str(aug_file),
                        "eye_state": row["eye_state"],
                        "eye_side": row["eye_side"],
                        "original_path": row["image_path"],
                        "split": split,
                        "augmented": True,
                        "label": 1 if row["eye_state"] == "open" else 0
                    })

        return records


    # ------------------ MASTER RUN ------------------
    def run_full_processing(self):
        print("🚀 Starting CEW preprocessing...")
        df = self.gather_dataset()

        if df.empty:
            print("❌ No CEW images found!")
            return None

        train, val, test = self.create_balanced_splits(df)

        train_records = self.process_and_save_split(train, "train", augment=True)
        val_records = self.process_and_save_split(val, "val", augment=False)
        test_records = self.process_and_save_split(test, "test", augment=False)

        all_records = train_records + val_records + test_records
        out_df = pd.DataFrame(all_records)

        out_df.to_csv(self.OUT_CSV, index=False)

        print(f"💾 CSV saved to: {self.OUT_CSV}")
        print(f"💾 Processed .npy saved at: {self.PROCESSED_DIR}")

        print("🎉 CEW PREPROCESSING COMPLETED!")
        return out_df


# ------------------ RUN SCRIPT ------------------
def main():
    CEWFullPreprocessor().run_full_processing()


if __name__ == "__main__":
    main()