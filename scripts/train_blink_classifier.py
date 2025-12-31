import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from pathlib import Path

# ================= PATH CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "outputs" / "eyeblink8" / "eyeblink8_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "blink_detection"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("🚀 TRAINING BLINK CLASSIFIER (ROBUST VERSION)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"📊 Raw Dataset: {len(df)} frames | Blinks: {df['blink'].sum()}")

# ================= CLEAN BAD DATA =================
# Remove any frames where face detection failed / EAR is zero
df = df[df["avg_ear"] > 0]
print(f"✅ After Cleaning: {len(df)} frames | Blinks: {df['blink'].sum()}")

# ================= FEATURE ENGINEERING =================
def create_features(df_in: pd.DataFrame) -> pd.DataFrame:
    features_df = df_in.copy()

    # Basic EAR relationships
    features_df["ear_diff"] = features_df["left_ear"] - features_df["right_ear"]
    features_df["ear_ratio"] = features_df["left_ear"] / (features_df["right_ear"] + 1e-8)

    base_cols = ["avg_ear", "left_ear", "right_ear", "ear_diff"]

    # Compute temporal features per video in time order
    for video_id in features_df["video"].unique():
        video_mask = features_df["video"] == video_id
        video_data = features_df.loc[video_mask].sort_values("timestamp")

        idx = video_data.index  # preserve index alignment

        for col in base_cols:
            rolling_mean = (
                video_data[col].rolling(window=3, min_periods=1).mean()
            )
            velocity = video_data[col].diff().fillna(0)

            features_df.loc[idx, f"{col}_rolling_mean_3"] = rolling_mean.values
            features_df.loc[idx, f"{col}_velocity"] = velocity.values

    return features_df


df = create_features(df)

feature_columns = [
    "avg_ear",
    "left_ear",
    "right_ear",
    "ear_diff",
    "ear_ratio",
    "avg_ear_rolling_mean_3",
    "left_ear_rolling_mean_3",
    "right_ear_rolling_mean_3",
    "avg_ear_velocity",
]

# Clean infinities / NaNs from feature engineering
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

X = df[feature_columns]
y = df["blink"]

print(f"📈 Blink Ratio: {y.mean():.2%}")

# ================= VIDEO-LEVEL SPLIT (BLINK-AWARE) =================
# Ensure both train and test have at least some blink frames
video_blinks = df.groupby("video")["blink"].sum()
pos_videos = video_blinks[video_blinks > 0].index.tolist()
neg_videos = video_blinks[video_blinks == 0].index.tolist()

train_videos = []
test_videos = []

# Split videos that contain blinks
if len(pos_videos) > 1:
    tr_pos, te_pos = train_test_split(
        pos_videos, test_size=0.3, random_state=42
    )
    train_videos.extend(tr_pos)
    test_videos.extend(te_pos)
else:
    # Edge case: only one video with blinks
    train_videos.extend(pos_videos)

# Split non-blink videos (if any)
if len(neg_videos) > 1:
    tr_neg, te_neg = train_test_split(
        neg_videos, test_size=0.3, random_state=42
    )
    train_videos.extend(tr_neg)
    test_videos.extend(te_neg)
elif len(neg_videos) == 1:
    # Put the single non-blink video into train
    train_videos.extend(neg_videos)

train_videos = list(sorted(set(train_videos)))
test_videos = list(sorted(set(test_videos)))

print(f"🎬 Train videos: {len(train_videos)} | Test videos: {len(test_videos)}")
print(f"   Train: {train_videos}")
print(f"   Test : {test_videos}")

train_df = df[df["video"].isin(train_videos)]
test_df = df[df["video"].isin(test_videos)]

X_train = train_df[feature_columns]
y_train = train_df["blink"]
X_test = test_df[feature_columns]
y_test = test_df["blink"]

print(f"🔹 Train frames: {len(X_train)} | Blinks: {y_train.sum()}")
print(f"🔹 Test frames : {len(X_test)} | Blinks: {y_test.sum()}")

# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODELS =================
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "SVM": SVC(
        class_weight="balanced", probability=True, random_state=42
    ),
}

results = {}

print("\n🤖 TRAINING MODELS")
print("=" * 50)

for name, model in models.items():
    print(f"\n▶ Training {name}")

    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Robust AUC computation
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5  # fallback if only one class present in y_test

    # ROC-based optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    if len(thresholds) > 0:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5

    preds = (probs > optimal_threshold).astype(int)
    acc = np.mean(preds == y_test)

    results[name] = {
        "model": model,
        "auc": auc,
        "accuracy": acc,
        "threshold": float(optimal_threshold),
    }

    print(f"AUC: {auc:.3f} | Accuracy: {acc:.3f} | Optimal Threshold: {optimal_threshold:.4f}")

# ================= SELECT BEST =================
best_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_name]["model"]
BEST_THRESHOLD = results[best_name]["threshold"]

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"✅ Optimal Threshold: {BEST_THRESHOLD:.4f}")

# ================= FINAL EVAL =================
final_probs = best_model.predict_proba(X_test_scaled)[:, 1]
final_preds = (final_probs > BEST_THRESHOLD).astype(int)

print("\n📋 Classification Report")
print(classification_report(y_test, final_preds, zero_division=0))

# (Optional) Confusion matrix plot
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, final_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ================= SAVE EVERYTHING =================
joblib.dump(best_model, MODEL_DIR / "blink_classifier.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_columns, MODEL_DIR / "features.pkl")
joblib.dump(BEST_THRESHOLD, MODEL_DIR / "threshold.pkl")


print("\n✅ Model + Threshold saved")

# ================= PREDICTION FUNCTION =================
def predict_blink(features_dict):
    """
    Predict blink given a single feature dict with the same keys as feature_columns.
    Missing features will be filled with 0 (safe default).
    """
    df_input = pd.DataFrame([features_dict])

    for col in feature_columns:
        if col not in df_input:
            df_input[col] = 0

    scaled = scaler.transform(df_input[feature_columns])
    prob = best_model.predict_proba(scaled)[0, 1]
    pred = int(prob > BEST_THRESHOLD)

    return {
        "blink_probability": float(prob),
        "prediction": pred,
        "threshold": float(BEST_THRESHOLD),
        "is_blink": bool(pred),
    }
print("\n🚀 Prediction Function Ready")