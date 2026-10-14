"""
Phase 2 — Dataset acquisition.

Downloads the IIIT-H Hindi-English code-mixed sentiment dataset
("IIITH_Codemixed.txt") and does a basic sanity check (row count,
label distribution). See README.md for why this dataset was chosen
over raw SAIL 2017 / random Kaggle dumps.

Usage:
    python src/download_data.py
"""

import os
import urllib.request
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/drimpossible/"
    "Sub-word-LSTM/master/Data/IIITH_Codemixed.txt"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(DATA_DIR, "IIITH_Codemixed.txt")

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"Already present: {OUT_PATH}")
        return
    print(f"Downloading from {RAW_URL} ...")
    urllib.request.urlretrieve(RAW_URL, OUT_PATH)
    print(f"Saved to {OUT_PATH}")


def load_raw(path: str = OUT_PATH) -> pd.DataFrame:
    """
    File format: id \\t text \\t label \\t label   (label is duplicated
    in the source file; second column is dropped).
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue  # skip malformed lines instead of crashing the load
            row_id, text, label = parts[0], parts[1], parts[2]
            try:
                label = int(label)
            except ValueError:
                continue
            rows.append({"id": row_id, "text": text, "label": label})
    df = pd.DataFrame(rows)
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df


def sanity_check(df: pd.DataFrame):
    print(f"\nTotal rows loaded: {len(df)}")
    print("\nLabel distribution:")
    dist = df["label_name"].value_counts()
    for name, count in dist.items():
        print(f"  {name:10s}: {count:5d}  ({count / len(df):.1%})")
    print("\nSample rows:")
    print(df.sample(5, random_state=42)[["text", "label_name"]].to_string(index=False))
    n_dupes = df["text"].duplicated().sum()
    print(f"\nDuplicate text rows: {n_dupes}")
    n_empty = df["text"].str.strip().eq("").sum()
    print(f"Empty text rows: {n_empty}")


if __name__ == "__main__":
    download()
    df = load_raw()
    sanity_check(df)

    # NOTE (stretch goal): to add a second dataset for cross-domain
    # robustness testing in Phase 6, drop a second loader function here,
    # e.g. load_hf_dataset("<hf-dataset-id>") or load_kaggle_csv(path),
    # and keep it OUT of the main train/val/test split — use it only as
    # an extra held-out generalization test set.
