import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = r"C:\Users\Shail\Downloads\neo_cognition\outputs\eyeblink8\eyeblink8_processed.csv"

print("📊 Loading Eyeblink8 dataset...")
df = pd.read_csv(CSV_PATH)

print("\n✅ Dataset Loaded")
print("Shape:", df.shape)

print("\n================ BASIC INFO ================")
print(df.info())

print("\n================ SAMPLE ROWS ================")
print(df.head())

print("\n================ NULL CHECK ================")
print(df.isnull().sum())

print("\n================ BLINK DISTRIBUTION ================")
print(df['blink'].value_counts())

print("\n================ EAR STATS ================")
print(df[['left_ear','right_ear','avg_ear']].describe())

# ================= PLOTS =================

plt.figure(figsize=(6,4))
df['blink'].value_counts().plot(kind='bar', title="Blink Distribution")
plt.xlabel("Blink (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df['avg_ear'], bins=50)
plt.title("Average EAR Distribution")
plt.xlabel("EAR Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Blink vs EAR
blink_ear = df[df['blink'] == 1]['avg_ear']
non_blink_ear = df[df['blink'] == 0]['avg_ear']

plt.figure(figsize=(8,4))
plt.hist(non_blink_ear, bins=50, alpha=0.6, label="No Blink")
plt.hist(blink_ear, bins=50, alpha=0.8, label="Blink")
plt.legend()
plt.title("EAR Comparison: Blink vs Non-Blink")
plt.xlabel("EAR")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\n================ VIDEO LEVEL SUMMARY ================")
video_summary = df.groupby("video").agg({
    "blink": "sum",
    "avg_ear": "mean",
    "frame": "count"
}).rename(columns={"frame": "total_frames"})

print(video_summary)
print("\n✅ EDA Completed")