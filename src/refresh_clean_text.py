"""
Phase 4.5 fix — recompute `text_clean` / `lang_tags` with the fixed
preprocessing (emoticon placeholder bug) WITHOUT re-splitting.

Train/val/test membership is preserved row-for-row (same ids, same
order), so results before and after the fix are directly comparable and
nothing downstream has to change. Only the cleaned text changes: the
~3% of comments with emoticons now keep "<3", ":)", ":(" ... instead of
a single meaningless "EMO0" token.

Usage:
    python src/refresh_clean_text.py
"""

import os

import pandas as pd

from preprocessing import clean_text, tag_language_per_token, PROCESSED_DIR


def main():
    for name in ("cleaned", "train", "val", "test"):
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df = pd.read_csv(path)
        old = df["text_clean"].astype(str).copy()
        df["text_clean"] = df["text"].astype(str).apply(clean_text)
        df["lang_tags"] = df["text_clean"].apply(tag_language_per_token)
        changed = (old != df["text_clean"]).sum()
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"{name:>8}: {len(df)} rows, text_clean changed in {changed}")


if __name__ == "__main__":
    main()
