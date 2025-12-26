# src/fusion/train_fusion_model.py
"""
Train the multimodal fusion model on outputs/fusion/fusion_dataset.csv

Outputs:
  - models/fusion/fusion_model.pt
  - outputs/fusion/fusion_scaler.json  (mean/std per column)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from fusion_model import MultimodalFusionNet


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FUSION_CSV = os.path.join(ROOT, "outputs", "fusion", "fusion_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models", "fusion")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------- Load fusion CSV ----------------
df = pd.read_csv(FUSION_CSV)
print("Loaded fusion:", df.shape)

# Remove id-like columns
id_cols = [c for c in ['dataset','participant_id','block_id'] if c in df.columns]
feature_cols = [c for c in df.columns if c not in id_cols]

# For training, we need targets. Since fusion rows come from multiple datasets,
# we may not have ground-truth cognitive labels. For demo, we will:
# - If 'cognitive_label' present in CSV use it
# - Else we create synthetic pseudo-labels (NOT ideal). Replace with real labels later.
target_col = None
for t in ['cognitive_load', 'cognitive', 'label_cognitive']:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    print("⚠️ No target column found in fusion CSV. Creating synthetic target for demo.")
    # synthetic target: normalize sum of select features -> pseudo cognitive load
    s = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    pseudo = (s.mean(axis=1) - s.mean().mean()) / (s.mean().std() + 1e-9)
    pseudo = 0.5 + 0.5 * np.tanh(pseudo / 2.0)  # 0-1
    df['__target__'] = pseudo
    target_col = '__target__'

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED)

# Build numeric matrix per modality
# Identify modality prefixes: blink_, nthu_, gaze_, cew_, clas_
modalities = {}
for col in feature_cols:
    if col.startswith("blink_"):
        modalities.setdefault("blink", []).append(col)
    elif col.startswith("nthu_"):
        modalities.setdefault("nthu", []).append(col)
    elif col.startswith("gaze_"):
        modalities.setdefault("gaze", []).append(col)
    elif col.startswith("cew_"):
        modalities.setdefault("cew", []).append(col)
    elif col.startswith("clas_"):
        modalities.setdefault("clas", []).append(col)
    else:
        modalities.setdefault("other", []).append(col)

print("Detected modalities and dims:")
modality_dims = {k: len(v) for k,v in modalities.items() if len(v)>0}
print(modality_dims)

# Scale numeric columns globally (store scaler)
num_cols = []
for v in modalities.values():
    num_cols += v
num_cols = [c for c in num_cols if c in df.columns]

scaler = StandardScaler()
scaler.fit(train_df[num_cols].fillna(0).to_numpy())

# save scaler params
scaler_info = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "cols": num_cols}
with open(os.path.join(MODEL_DIR, "fusion_scaler.json"), "w") as f:
    json.dump(scaler_info, f)

# Dataset class
class FusionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.x = scaler.transform(self.df[num_cols].fillna(0).to_numpy()).astype(np.float32)
        self.y = self.df[target_col].astype(np.float32).to_numpy()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.x[i]
        # split into modality tensors
        xdict = {}
        pointer = 0
        for k, cols in modalities.items():
            if len(cols)==0: continue
            d = len(cols)
            xdict[k] = torch.from_numpy(row[pointer:pointer+d]).float()
            pointer += d
        y = torch.tensor(self.y[i]).float()
        return xdict, y

train_ds = FusionDataset(train_df)
val_ds = FusionDataset(val_df)
test_ds = FusionDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# Build model
model = MultimodalFusionNet(modality_dims={k: len(v) for k,v in modalities.items() if len(v)>0})
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

# Training loop (regression on target 0-1)
best_val = 1e9
patience = 8
cur_pat = 0
EPOCHS = 100

for epoch in range(1, EPOCHS+1):
    model.train()
    train_losses = []
    for xdict, y in train_loader:
        # move to device and reshape per modality
        x_in = {}
        for k, t in xdict.items():
            x_in[k] = t.to(DEVICE)
        y = y.to(DEVICE)
        out = model(x_in)
        pred = out['cognitive']
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)

    # val
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xdict, y in val_loader:
            x_in = {k: t.to(DEVICE) for k,t in xdict.items()}
            y = y.to(DEVICE)
            out = model(x_in)
            pred = out['cognitive']
            loss = criterion(pred, y)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}")

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        cur_pat = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fusion_model_best.pt"))
    else:
        cur_pat += 1
    if cur_pat >= patience:
        print("Early stopping")
        break

# test metrics
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fusion_model_best.pt")))
model.eval()
preds = []
gts = []
with torch.no_grad():
    for xdict, y in test_loader:
        x_in = {k: t.to(DEVICE) for k,t in xdict.items()}
        out = model(x_in)
        pred = out['cognitive'].cpu().numpy()
        preds.append(pred)
        gts.append(y.numpy())
preds = np.concatenate(preds)
gts = np.concatenate(gts)
mse = np.mean((preds - gts)**2)
print("Test MSE:", mse)

# Save final model (state dict already saved)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fusion_model_final.pt"))
print("Saved models to", MODEL_DIR)
