import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(base_dir: Path):
    csv_path = base_dir / "outputs" / "MPIIGAZE" / "mpiigaze_features.csv"
    df = pd.read_csv(csv_path)

    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "avg_ear", "left_ear", "right_ear", "ear_asymmetry",
        "gaze_x", "gaze_y", "gaze_angle", "gaze_magnitude",
        "gaze_velocity", "gaze_entropy", "fixation_stability"
    ]

    df = df.dropna(subset=feature_cols + ["attention_score"])

    X = df[feature_cols].values
    y = df["attention_score"].values

    print(f"📊 Dataset after cleaning: {len(df)} samples")
    print(f"🎯 Attention range: {y.min():.3f} → {y.max():.3f}")

    return X, y, feature_cols, df


def train_attention_model(base_dir: Path):
    print("\n🧠 Training MPIIGAZE Attention Model (XGBoost)")
    print("=" * 60)

    X, y, feature_cols, df = load_data(base_dir)

    print(f"Training samples: {len(y)}")
    print(f"Attention mean: {y.mean():.3f}, std: {y.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    print("🔄 Training model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 PERFORMANCE")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # ========== SAVE OUTPUTS ==========
    models_dir = base_dir / "outputs" / "models_mpiigaze"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "attention_xgb.pkl")
    joblib.dump(scaler, models_dir / "attention_scaler.pkl")

    with open(models_dir / "attention_features.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    with open(models_dir / "training_metadata.json", "w") as f:
        json.dump(
            {
                "r2": float(r2),
                "rmse": float(rmse),
                "mse": float(mse),
                "top_features": feature_importance.head(10).to_dict("records")
            },
            f,
            indent=2
        )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=feature_importance["importance"],
        y=feature_importance["feature"]
    )
    plt.title("Feature Importance - Attention Model")
    plt.tight_layout()
    plt.savefig(models_dir / "feature_importance.png", dpi=300)
    plt.close()

    print("\n✅ MODEL SAVED SUCCESSFULLY")
    print(models_dir)


def main():
    train_attention_model(PROJECT_ROOT)


if __name__ == "__main__":
    main()
