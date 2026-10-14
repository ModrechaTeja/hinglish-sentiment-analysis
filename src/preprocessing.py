"""
Phase 3 — Preprocessing for Hindi-English code-mixed (Hinglish) text.

Design choices (documented so they're easy to defend in the write-up):

1. We do NOT force-transliterate everything to one script. Devanagari
   ("अच्छा") is preserved as-is; only Romanized spelling *variants* of the
   same word ("acha"/"achha"/"accha") are folded together via a lookup
   table. Destroying the script-mixing signal would remove exactly the
   phenomenon Phase 6's error analysis is supposed to study.
2. Emojis are converted to text tags (":red_heart:" style via `emoji`
   library) rather than stripped — they carry real sentiment signal.
   Text emoticons (":)", ":(", "xD") are preserved as-is for the same
   reason.
3. URLs, @mentions, and excess punctuation/whitespace are stripped
   because they add noise without sentiment signal; "!" and "?" repeats
   are collapsed to at most 2 (keeps emphasis, drops noise).
4. Per-token language ID is optional (flagged) — a lightweight heuristic
   dictionary lookup, not a full LID model, to keep the pipeline light.
   Turn it on if you want the extra "depth" feature from the roadmap.

Usage:
    python src/preprocessing.py
"""

import os
import re
import emoji
import pandas as pd

from download_data import load_raw, OUT_PATH  # Phase 2 loader

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEANED_PATH = os.path.join(PROCESSED_DIR, "cleaned.csv")

# --------------------------------------------------------------------------
# 1. Transliteration / spelling normalization
# --------------------------------------------------------------------------
# Small, extensible lookup of common Romanized-Hindi spelling variants seen
# in the dataset (and Hinglish generally). Extend this as you do error
# analysis in Phase 6 and spot more variant clusters.
NORMALIZATION_MAP = {
    # good/nice
    "acha": "achha", "aacha": "achha", "accha": "achha", "acchha": "achha",
    # is/are (hai variants)
    "h": "hai", "hy": "hai", "hai": "hai",
    # not
    "nai": "nahi", "nahin": "nahi", "nhi": "nahi", "nahi": "nahi",
    # very
    "bhot": "bahut", "bahot": "bahut", "bohot": "bahut", "bhut": "bahut",
    # you (respectful)
    "aap": "aap", "ap": "aap",
    # please
    "plz": "please", "pls": "please", "plzz": "please",
    # thanks
    "thnx": "thanks", "thx": "thanks", "tysm": "thanks",
    # great
    "gr8": "great", "grt": "great",
    # love
    "luv": "love",
    # why
    "kyu": "kyun", "kyun": "kyun", "kyon": "kyun",
}


def normalize_transliteration(text: str) -> str:
    tokens = text.split()
    normed = [NORMALIZATION_MAP.get(tok.lower(), tok) for tok in tokens]
    return " ".join(normed)


# --------------------------------------------------------------------------
# 2. Emoji / emoticon handling
# --------------------------------------------------------------------------
EMOTICON_PATTERN = re.compile(
    r"(:\)|:\(|:D|:P|:p|;\)|:-\)|:-\(|xD|XD|<3|:'\(|:O|:o)"
)


def handle_emoji(text: str) -> str:
    # Emoji -> readable tag, e.g. "😂" -> " :face_with_tears_of_joy: "
    text = emoji.demojize(text, delimiters=(" :", ": "))
    # Text emoticons are left untouched by demojize; nothing further needed,
    # they survive noise_removal() below because we whitelist them there.
    return text


# --------------------------------------------------------------------------
# 3. Noise removal
# --------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
MULTI_PUNCT_PATTERN = re.compile(r"([!?])\1{2,}")   # !!!! -> !!
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")
# keep letters (incl. Devanagari), digits, basic punctuation, emoticons, colons (for emoji tags)
ALLOWED_CHARS_PATTERN = re.compile(
    r"[^a-zA-Z0-9\u0900-\u097F\s.,!?':;\-]"
)


def remove_noise(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_SYMBOL_PATTERN.sub("", text)      # "#great" -> "great"
    text = MULTI_PUNCT_PATTERN.sub(r"\1\1", text)    # collapse runs of !/? to 2
    # protect emoticons before the character whitelist strips symbols like ()
    protected = {}
    def _protect(m):
        # Placeholder must consist ONLY of whitelisted chars (letters/digits),
        # otherwise ALLOWED_CHARS_PATTERN mangles it and the emoticon is never
        # restored. (Bug fixed in Phase 4.5: the old "__EMO0__" key lost its
        # underscores, so every emoticon collapsed into the same "EMO0" token.)
        key = f"qqemo{len(protected)}qq"
        protected[key] = m.group(0)
        return key
    text = EMOTICON_PATTERN.sub(_protect, text)
    text = ALLOWED_CHARS_PATTERN.sub(" ", text)
    for key, val in protected.items():
        text = text.replace(key, val)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# 4. (Optional) crude per-token language ID
# --------------------------------------------------------------------------
# Heuristic only: a short seed list of very common Hindi (Romanized) and
# English function words. Good enough to tag "likely hi / likely en /
# unknown" for an exploratory plot; NOT a substitute for a real LID model.
_HINDI_SEED = {
    "hai", "nahi", "bahut", "achha", "kya", "kyun", "hum", "tum", "aap",
    "mai", "mera", "tera", "uska", "yeh", "woh", "kar", "karo", "raha",
    "rahi", "the", "thi", "yaar", "bhai", "sir", "ji", "jaa", "aaj",
}
_ENGLISH_SEED = {
    "the", "is", "are", "was", "were", "movie", "good", "bad", "great",
    "love", "like", "please", "thanks", "very", "not", "this", "that",
}


def tag_language_per_token(text: str) -> str:
    tags = []
    for tok in text.split():
        low = tok.lower()
        if re.search(r"[\u0900-\u097F]", tok):
            tags.append("hi_dev")
        elif low in _HINDI_SEED:
            tags.append("hi_rom")
        elif low in _ENGLISH_SEED:
            tags.append("en")
        else:
            tags.append("unk")
    return " ".join(tags)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = handle_emoji(text)
    text = remove_noise(text)
    text = normalize_transliteration(text)
    return text


def run(add_lang_tags: bool = True) -> pd.DataFrame:
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run `python src/download_data.py` first."
        )
    df = load_raw(OUT_PATH)
    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    before = len(df)
    df = df[df["text_clean"].str.strip().ne("")].drop_duplicates(subset="text_clean")
    print(f"Dropped {before - len(df)} empty/duplicate rows after cleaning "
          f"({before} -> {len(df)}).")

    if add_lang_tags:
        df["lang_tags"] = df["text_clean"].apply(tag_language_per_token)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False, encoding="utf-8")
    print(f"Saved cleaned data -> {CLEANED_PATH}")
    return df


if __name__ == "__main__":
    df = run()
    print("\nBefore/after examples:\n")
    raw_df = load_raw(OUT_PATH)
    for i in [0, 5, 20]:
        print("RAW  :", raw_df.iloc[i]["text"])
        print("CLEAN:", clean_text(raw_df.iloc[i]["text"]))
        print()
