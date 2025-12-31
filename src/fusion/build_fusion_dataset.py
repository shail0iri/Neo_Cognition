import os
import sys
import json
import numpy as np
import pandas as pd

# ================= PATH CONFIG =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PATHS = {
    "eyeblink8": os.path.join(ROOT, "outputs", "eyeblink8", "eyeblink8_processed.csv"),
    "nthu":      os.path.join(ROOT, "outputs", "NTHU", "nthu_features_optimized.csv"),
    "mpiigaze":  os.path.join(ROOT, "outputs", "MPIIGAZE", "mpiigaze_features.csv"),
    "cew":       os.path.join(ROOT, "outputs", "CEW", "cew_processed_dataset.csv"),
    "clas_features": os.path.join(ROOT, "data", "raw" , "CLAS", "processed", "clas_features.csv"),
    "clas_timeseries": os.path.join(ROOT, "data", "raw" , "CLAS", "processed", "clas_timeseries.csv"),
}

OUT_DIR = os.path.join(ROOT, "outputs", "fusion")
os.makedirs(OUT_DIR, exist_ok=True)

NUMERIC_AGG_FUNCS = {
    "mean": np.nanmean,
    "std":  np.nanstd,
    "min":  np.nanmin,
    "max":  np.nanmax,
    "25p":  lambda x: np.nanpercentile(x, 25),
    "50p":  lambda x: np.nanpercentile(x, 50),
    "75p":  lambda x: np.nanpercentile(x, 75),
    "count": lambda x: np.count_nonzero(~np.isnan(x)),
    "nan_ratio": lambda x: float(np.isnan(x).sum()) / max(1, len(x))
}

# ---------------- helpers ----------------
def safe_load_csv(path):
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"📥 Loaded {os.path.basename(path)} -> {len(df):,} rows, {len(df.columns)} cols")
        return df
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def numeric_columns(df):
    # Return numeric columns only (float/int)
    return df.select_dtypes(include=[np.number]).columns.tolist()

def aggregate_df_to_row(df, prefix, id_info=None):
    """
    Aggregate a dataframe to a single row embedding.
    - prefix: string to prefix all aggregated column names
    - id_info: optional dict to include in the resulting row (e.g. {"dataset":"Eyeblink8"})
    """
    row = {}
    if id_info:
        row.update(id_info)

    num_cols = numeric_columns(df)
    if not num_cols:
        # No numeric columns => fallback: counts
        row[f"{prefix}_num_columns"] = 0
        return row

    for col in num_cols:
        vals = df[col].to_numpy(dtype=float)
        for agg_name, func in NUMERIC_AGG_FUNCS.items():
            try:
                val = func(vals)
                col_name = f"{prefix}_{col}__{agg_name}"
                # convert numpy types to native python floats/ints for JSON/CSV friendliness
                if isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                row[col_name] = val
            except Exception:
                row[f"{prefix}_{col}__{agg_name}"] = np.nan

    # also include number of numeric cols
    row[f"{prefix}_num_numeric_cols"] = len(num_cols)
    row[f"{prefix}_nrows"] = len(df)
    return row

# ---------------- loaders & modality-specific handling ----------------

def build_eyeblink_embedding(path):
    df = safe_load_csv(path)
    if df is None:
        return None
    # rename columns to safe names if necessary (optional)
    emb = aggregate_df_to_row(df, prefix="blink", id_info={"dataset": "eyeblink8"})
    return emb

def build_nthu_embedding(path):
    df = safe_load_csv(path)
    if df is None:
        return None
    emb = aggregate_df_to_row(df, prefix="nthu", id_info={"dataset": "nthu"})
    return emb

def build_mpiigaze_embedding(path):
    df = safe_load_csv(path)
    if df is None:
        return None
    emb = aggregate_df_to_row(df, prefix="gaze", id_info={"dataset": "mpiigaze"})
    return emb

def build_cew_embedding(path):
    df = safe_load_csv(path)
    if df is None:
        return None
    emb = aggregate_df_to_row(df, prefix="cew", id_info={"dataset": "cew"})
    return emb

def build_clas_block_embeddings(path):
    """
    Return:
      - list of per-block dict rows (participant_id, block_id, prefixed features)
      - a global aggregated dict (participant_id='CLAS_all', block_id=-1)
    """
    df = safe_load_csv(path)
    if df is None:
        return None, None

    # Expect participant_id and block_id columns
    if 'participant_id' not in df.columns or 'block_id' not in df.columns:
        print("⚠️ clas_features missing participant_id/block_id, attempting to infer")
        # try to infer common columns - fallback will group by whatever exists
        if 'participant' in df.columns:
            df = df.rename(columns={'participant': 'participant_id'})
        else:
            # create synthetic participant_id if missing
            df['participant_id'] = 'CLAS_anon'

        if 'block' in df.columns:
            df = df.rename(columns={'block': 'block_id'})
        else:
            df['block_id'] = df.index

    rows = []
    grouped = df.groupby(['participant_id', 'block_id'], sort=False)
    for (pid, bid), g in grouped:
        id_info = {"dataset": "clas", "participant_id": pid, "block_id": int(bid)}
        emb = aggregate_df_to_row(g, prefix="clas", id_info=id_info)
        rows.append(emb)

    # Global aggregation across all blocks
    global_emb = aggregate_df_to_row(df, prefix="clas", id_info={"dataset": "clas_global", "participant_id": "CLAS_ALL", "block_id": -1})
    return rows, global_emb

# ---------------- main builder ----------------

def build_fusion_dataset(paths=DEFAULT_PATHS, out_dir=OUT_DIR):
    all_rows = []

    # 1) Eyeblink8 aggregated row
    eb = build_eyeblink_embedding(paths['eyeblink8'])
    if eb:
        all_rows.append(eb)
        pd.DataFrame([eb]).to_csv(os.path.join(out_dir, "blink_aggregated.csv"), index=False)

    # 2) NTHU aggregated row
    nt = build_nthu_embedding(paths['nthu'])
    if nt:
        all_rows.append(nt)
        pd.DataFrame([nt]).to_csv(os.path.join(out_dir, "nthu_aggregated.csv"), index=False)

    # 3) MPIIGAZE aggregated row
    mg = build_mpiigaze_embedding(paths['mpiigaze'])
    if mg:
        all_rows.append(mg)
        pd.DataFrame([mg]).to_csv(os.path.join(out_dir, "mpiigaze_aggregated.csv"), index=False)

    # 4) CEW aggregated row
    cw = build_cew_embedding(paths['cew'])
    if cw:
        all_rows.append(cw)
        pd.DataFrame([cw]).to_csv(os.path.join(out_dir, "cew_aggregated.csv"), index=False)

    # 5) CLAS per-block + global
    clas_block_rows, clas_global = build_clas_block_embeddings(paths['clas_features'])
    if clas_block_rows:
        # write per-block rows csv
        pd.DataFrame(clas_block_rows).to_csv(os.path.join(out_dir, "clas_block_embeddings.csv"), index=False)
        all_rows.extend(clas_block_rows)
    if clas_global:
        pd.DataFrame([clas_global]).to_csv(os.path.join(out_dir, "clas_global_embedding.csv"), index=False)
        all_rows.append(clas_global)

    # Final consolidation: normalize all rows into same set of columns
    if not all_rows:
        raise RuntimeError("❌ No embeddings were built. Check input paths.")

    fusion_df = pd.DataFrame(all_rows).fillna(np.nan)

    # sort columns: put id-info first
    id_cols = [c for c in ['dataset', 'participant_id', 'block_id'] if c in fusion_df.columns]
    other_cols = [c for c in fusion_df.columns if c not in id_cols]
    fusion_df = fusion_df[id_cols + other_cols]

    outpath = os.path.join(out_dir, "fusion_dataset.csv")
    fusion_df.to_csv(outpath, index=False)
    print(f"💾 Fusion dataset saved → {outpath}")
    print(f"ℹ️ Fusion rows: {len(fusion_df)} | features: {len(fusion_df.columns) - len(id_cols)}")
    return fusion_df

# ---------------- CLI ----------------
if __name__ == "__main__":
    print("🚀 Building fusion dataset (Option C: CLAS blocks + global + other datasets aggregated)...")
    fusion_df = build_fusion_dataset()
    # print basic summary
    print(fusion_df.head(5).T)
    print("✅ Done.")
