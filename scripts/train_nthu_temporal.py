# =============================================================
# ✅ NTHU TEMPORAL MODEL TRAINING - FIXED VERSION
# Solves:
# - invalid literal for int() with 'drowsy'
# - robust label encoding
# - safe preprocessing for temporal ML
# Works with: nthu_features_optimized.csv
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

print("✅ NTHU TEMPORAL TRAINING STARTED")

# ================= PATH CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "outputs" / "nthu_features_optimized.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

# ================= LOAD DATA =================
print("📥 Loading NTHU features...")

df = pd.read_csv(CSV_PATH)
print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

# ================= LABEL FIX =================
print("🔄 Converting labels to numeric...")

print("Unique label values before:", df[LABEL_COL].unique())

label_mapping = {
    "drowsy": 1,
    "notdrowsy": 0,
    "alert": 0,
    "normal": 0,
    "1": 1,
    "0": 0
}

df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map(label_mapping)

if df[LABEL_COL].isnull().any():
    bad = df[df[LABEL_COL].isnull()][LABEL_COL]
    raise ValueError(f"❌ Unmapped labels found: {bad.unique()}")

print("✅ Labels after conversion:")
print(df[LABEL_COL].value_counts())

# ================= FEATURE SELECTION =================

possible_path_cols = [
    "video_path", "image_path", "filename", "frame_path",
    "file_path", "name", "source"
]
df = df.drop(columns=[c for c in possible_path_cols if c in df.columns], errors="ignore")

safe_numeric_cols = []
for col in df.columns:
    if col == LABEL_COL:
        continue
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.isnull().sum() > df[col].isnull().sum():
        print(f"🚫 Dropping non-numeric contaminated column -> {col}")
    else:
        safe_numeric_cols.append(col)
        df[col] = converted

if not safe_numeric_cols:
    raise ValueError("❌ No safe numeric columns found!")

print(f"✅ Safe numeric features selected ({len(safe_numeric_cols)}):")
print(safe_numeric_cols)

X = df[safe_numeric_cols]
y = df[LABEL_COL].astype(int).values

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Drop any remaining non-numeric columns
string_like_cols = [col for col in df.columns if df[col].dtype == "object"]
df = df.drop(columns=string_like_cols, errors="ignore")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != LABEL_COL]

print(f"✅ Using {len(feature_cols)} numeric features:")
print(feature_cols)

X = df[feature_cols].fillna(df[feature_cols].median())
y = df[LABEL_COL].astype(int).values

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ================= SCALE =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, MODELS_DIR / "nthu_scaler.pkl")

# ================= RANDOM FOREST =================
print("🌳 Training RandomForest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("\n📊 RandomForest Report")
print(classification_report(y_test, rf_preds))

joblib.dump(rf, MODELS_DIR / "nthu_rf_model.pkl")

# ================= XGBOOST =================
print("🚀 Training XGBoost...")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_preds = (xgb_model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

print("\n📊 XGBoost Report")
print(classification_report(y_test, xgb_preds))

joblib.dump(xgb_model, MODELS_DIR / "nthu_xgb_model.pkl")

# ================= CONFUSION MATRIX =================
print("\n🧠 Confusion Matrix (XGBoost)")
print(confusion_matrix(y_test, xgb_preds))

print("✅ NTHU TEMPORAL MODEL TRAINING COMPLETE")
print("Models saved in:", MODELS_DIR)
