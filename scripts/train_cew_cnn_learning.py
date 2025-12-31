# =============================================================
# 🚀 CEW CNN FINAL STABLE TRAINING SCRIPT (Keras 3 Compatible)
# - No 'options' errors
# - No legacy save issues
# - Works with your CEW_processed + cew_processed_dataset.csv
# - Minimal RAM usage, stable accuracy learning
# =============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

print("✅ CEW CNN FINAL STABLE TRAINING STARTED")

from pathlib import Path

# ================= PATH CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "outputs" / "CEW" / "cew_processed_dataset.csv"
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "CEW_processed"
MODELS_DIR = PROJECT_ROOT / "models" / "cew"
REPORTS_DIR = PROJECT_ROOT / "reports" / "cew"

IMG_SIZE = 80
BATCH_SIZE = 16   # low RAM safe
EPOCHS = 20

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ================= LOAD CSV =================
print("📥 Loading dataset...")

df = pd.read_csv(CSV_PATH)

# Ensure paths are valid
existing_mask = df['file_path'].apply(lambda x: os.path.exists(str(x)))
df = df[existing_mask].reset_index(drop=True)

print("✅ Dataset loaded:")
print(df[['split','label']].value_counts())

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

# ================= DATA PIPELINE =================

def load_image(path, label):
    def _read(p):
        img = np.load(p.decode())
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32')
        if img.max() > 1.5:
            img /= 255.0
        return img

    img = tf.numpy_function(_read, [path], tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img, -1)
    return img, label


def make_dataset(dataframe, training=False):
    paths = dataframe['file_path'].values.astype(str)
    labels = dataframe['label'].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1200)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ================= MODEL =================

def build_model():
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.Rescaling(1.0),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

# ================= CALLBACKS =================
best_model_path = MODELS_DIR / "cew_best.keras"
final_model_path = MODELS_DIR / "cew_final.keras"


callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]


# ================= TRAIN =================
print("🚀 Training started...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save(final_model_path)
print("💾 Model saved")

# ================= EVALUATION =================
print("🧪 Testing...")
loss, acc = model.evaluate(test_ds)
print(f"Accuracy: {acc:.4f}")

# Predictions
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = np.concatenate([model.predict(x) for x, _ in test_ds])
y_pred = (y_prob > 0.5).astype(int)

print("\n📊 Classification Report")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CEW Confusion Matrix")
plt.savefig(REPORTS_DIR / "confusion_matrix.png")
plt.show()

# ================= CURVES =================
history_df = pd.DataFrame(history.history)
history_df.to_csv(REPORTS_DIR / "training_history.csv", index=False)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig(REPORTS_DIR /  "accuracy_curve.png")
plt.show()

print("✅ TRAINING COMPLETED SUCCESSFULLY")
