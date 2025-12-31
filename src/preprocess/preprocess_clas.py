import os
import re
import pandas as pd


# ================= PATH CONFIG =================
BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "CLAS")
)

PARTICIPANT_DIR = os.path.join(BASE, "Participants")
ANSWERS_DIR = os.path.join(BASE, "Answers")
BLOCK_DETAILS_DIR = os.path.join(BASE, "Block_details")
DOCUMENTATION_DIR = os.path.join(BASE, "Documentation")

OUT = os.path.join(BASE, "processed")
os.makedirs(OUT, exist_ok=True)



def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None


def extract_block_number(filename):
    m = re.match(r"(\d+)_", filename)
    return int(m.group(1)) if m else None


def detect_stream_type(filename):
    fname = filename.lower()
    if "ecg" in fname:
        return "ECG"
    if "gsr" in fname or "ppg" in fname:
        return "GSR_PPG"
    return "UNKNOWN"


# =========================================================
# STEP 1 — PARTICIPANT PHYSIOLOGICAL DATA
# =========================================================

def load_participant_signals():
    print("📥 Loading ECG + GSR/PPG streams from Participants/...")

    frames = []

    for participant in os.listdir(PARTICIPANT_DIR):
        p_dir = os.path.join(PARTICIPANT_DIR, participant)
        by_block = os.path.join(p_dir, "by_block")

        if not os.path.isdir(by_block):
            continue

        for fname in os.listdir(by_block):
            if not fname.endswith(".csv"):
                continue

            block_id = extract_block_number(fname)
            if block_id is None:
                continue

            df = safe_load_csv(os.path.join(by_block, fname))
            if df is None:
                continue

            df["participant_id"] = participant
            df["block_id"] = block_id
            df["stream_type"] = detect_stream_type(fname)
            df["source_file"] = fname

            task_match = re.findall(r"_(\w+)\.csv$", fname)
            df["task_name"] = task_match[0] if task_match else "unknown"

            frames.append(df)

    if not frames:
        raise RuntimeError("❌ No physiological CSV files found in Participants/")

    return pd.concat(frames, ignore_index=True)


# =========================================================
# STEP 2 — BLOCK DETAILS (ENSURE block_id EXISTS)
# =========================================================

def load_block_metadata():
    print("📥 Loading block details...")

    frames = []

    for fname in os.listdir(BLOCK_DETAILS_DIR):
        if not fname.endswith(".csv"):
            continue

        df = safe_load_csv(os.path.join(BLOCK_DETAILS_DIR, fname))
        if df is None:
            continue

        # extract participant ID
        m = re.match(r"Part(\d+)_Block_Details\.csv", fname)
        if not m:
            print(f"⚠️ Can't parse participant from {fname}")
            continue

        participant_id = f"Part{m.group(1)}"
        df["participant_id"] = participant_id

        # detect block id column
        possible_block_cols = [
            col for col in df.columns
            if "block" in col.lower() and "id" in col.lower()
        ]

        if possible_block_cols:
            df.rename(columns={possible_block_cols[0]: "block_id"}, inplace=True)
        else:
            # assign sequential block numbers (CLAS format)
            df["block_id"] = df.index

        frames.append(df)

    if not frames:
        raise RuntimeError("❌ No block detail CSVs found")

    return pd.concat(frames, ignore_index=True)


# =========================================================
# STEP 3 — ANSWERS (ENSURE participant_id + block_id)
# =========================================================

def load_answers():
    print("📥 Loading answer sheets...")

    frames = []

    for fname in os.listdir(ANSWERS_DIR):
        if not fname.endswith(".csv"):
            continue

        df = safe_load_csv(os.path.join(ANSWERS_DIR, fname))
        if df is None:
            continue

        # extract participant ID
        m = re.match(r"Part(\d+)_c_i_answers\.csv", fname)
        if not m:
            print(f"⚠️ Can't parse participant from {fname}")
            continue

        participant_id = f"Part{m.group(1)}"
        df["participant_id"] = participant_id

        # assign block numbers (answers correspond to blocks 0–38)
        df["block_id"] = df.index

        frames.append(df)

    if not frames:
        raise RuntimeError("❌ No answer CSVs found in Answers/")

    return pd.concat(frames, ignore_index=True)


# =========================================================
# STEP 4 — MERGE ALL INTO UNIFIED DATASET
# =========================================================

def build_unified_dataset():
    print("🔗 Building unified CLAS behavioural dataset...")

    signals = load_participant_signals()
    blocks = load_block_metadata()
    answers = load_answers()

    print("🔗 Merging: signals + block metadata...")
    merged = signals.merge(blocks, on=["participant_id", "block_id"], how="left")

    print("🔗 Merging: (signals + blocks) + answers...")
    merged = merged.merge(answers, on=["participant_id", "block_id"], how="left")

    outpath = os.path.join(OUT, "clas_timeseries.csv")
    merged.to_csv(outpath, index=False)

    print(f"💾 Unified timeseries saved → {outpath}")
    return merged


# =========================================================
# STEP 5 — FEATURE EXTRACTION
# =========================================================

def extract_features(df):
    print("⚙️ Extracting block-level features...")

    ecg_cols = [c for c in df.columns if "ecg" in c.lower()]
    gsr_cols = [c for c in df.columns if "gsr" in c.lower()]
    ppg_cols = [c for c in df.columns if "ppg" in c.lower()]

    if ecg_cols:
        df.rename(columns={ecg_cols[0]: "ECG"}, inplace=True)
    if gsr_cols:
        df.rename(columns={gsr_cols[0]: "GSR"}, inplace=True)
    if ppg_cols:
        df.rename(columns={ppg_cols[0]: "PPG"}, inplace=True)

    agg = df.groupby(["participant_id", "block_id"]).agg({
        "ECG": ["mean", "std"],
        "GSR": ["mean", "std"],
        "PPG": ["mean", "std"]
    })

    agg.columns = ["_".join(col) for col in agg.columns]
    agg.reset_index(inplace=True)

    outpath = os.path.join(OUT, "clas_features.csv")
    agg.to_csv(outpath, index=False)

    print(f"💾 Features saved → {outpath}")
    return agg


if __name__ == "__main__":
    print("\n🚀 Starting CLAS Preprocessing...\n")

    unified = build_unified_dataset()
    extract_features(unified)

    print("\n🎉 CLAS preprocessing completed successfully!\n")
