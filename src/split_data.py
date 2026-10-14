"""
Phase 3 — Stratified 70/15/15 train/val/test split.

Stratified on `label` so the neutral-heavy class imbalance (see README)
is preserved proportionally across all three splits, rather than the test
set randomly ending up short on the minority "negative" class.

Usage:
    python src/split_data.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import CLEANED_PATH, PROCESSED_DIR

RANDOM_STATE = 42


def split(df: pd.DataFrame):
    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE
    )
    return train_df, val_df, test_df


def report(name, split_df, full_df):
    dist = split_df["label_name"].value_counts(normalize=True).sort_index()
    print(f"{name:6s} n={len(split_df):4d}  ", dict(round(dist, 3)))


if __name__ == "__main__":
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"{CLEANED_PATH} not found — run `python src/preprocessing.py` first."
        )
    df = pd.read_csv(CLEANED_PATH)
    train_df, val_df, test_df = split(df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report(name, split_df, df)
        split_df.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)

    print(f"\nSaved train.csv / val.csv / test.csv -> {PROCESSED_DIR}")
