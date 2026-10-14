"""
Phase 7 — Error analysis: WHERE and WHY do models fail on code-mixed text?

Uses the saved predictions from Phases 4.5/5 (no retraining). Compares the
best classical model (chosen on validation) against each transformer's
selected run (see evaluate.py for the selection protocol).

Analyses
  1. Word-level language ID (approximate) -> Code-Mixing Index (CMI) per
     comment (Das & Gambäck, 2014), and performance by CMI bucket.
  2. Linguistic slices: length, out-of-vocabulary rate (a proxy for
     non-standard romanized spelling), elongated words ("dhannnsu"),
     negation, rhetorical questions, emoji.
  3. Confusion-type breakdown (which class pairs get confused).
  4. Classical-vs-transformer disagreement (who fixes whose errors).
  5. High-confidence errors (likely sarcasm / label noise; annotator
     kappa for this dataset is only 0.64).
  6. Tokenizer fertility (sub-word pieces per word) for mBERT / MuRIL /
     IndicBERT, split by Hindi vs English words. Needs the HF tokenizers
     (internet) -> pass --fertility (run it on Colab).
  7. A manual-annotation sheet of sampled errors to label by hand.

Language ID note: a dictionary/frequency heuristic, NOT a trained LID
model. A token is English if it is a frequent English word (wordfreq
Zipf >= 4.0) and not in a list of common romanized-Hindi words that collide
with English spellings ("main", "to", "me", "se", "par", ...); remaining
alphabetic tokens are treated as romanized Hindi. Good enough for
bucketing, stated as approximate in the report.

Usage
  python src/error_analysis.py              # everything except fertility
  python src/error_analysis.py --fertility  # + tokenizer fertility (Colab)
"""

import argparse
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from eval_utils import RESULTS_DIR, FIGURES_DIR, LABEL_NAMES, load_splits
from evaluate import (
    load_preds, select_transformer_runs, ordered_models,
    C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, INK, INK_MUTED, GRID, color_for,
)

OUT_DIR = RESULTS_DIR / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SERIES = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]

# Romanized-Hindi words that are also (frequent) English spellings, plus very
# common Hindi function words — always tagged Hindi.
HINDI_WORDS = set("""
main mai me mein to se ke ki ka ko ho hai hain he hi na nahi nahin nhi mat par pe bhi koi
jo kar kare karo karna kiya raha rahi rahe the thi tha thay ye yeh yah wo woh vo aur or ab
sab log bat baat aap ap tum tu hum ham mera meri mere tera teri tere apna apni uska uski
kya kyu kyun kab kaun kaise kitna bahut bohot bhot achha acha accha bhai ji sir jee
ne jab tab phir fir sath saath dil pyar pyaar jaan ek do teen hua hue hui gaya gayi diya
liye lie kuch kuchh bas sirf abhi aaj kal din raat desh sahi galat bura bure hamara hamare
""".split())
ENGLISH_SLANG = set("u ur r plz pls thx thnx gud nyc gr8 lol omg bro sis pic pics fb msg".split())
NEGATIONS = {"nahi", "nahin", "nhi", "na", "mat", "not", "no", "never", "dont", "don't",
             "cant", "can't", "isnt", "isn't", "nai", "naa", "bina"}
TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|EMO\d+|[^\sA-Za-z\d]+")


def word_lang(tok: str, zipf) -> str:
    low = tok.lower()
    if not re.fullmatch(r"[a-z']+", low) or re.fullmatch(r"emo\d+", low):
        return "other"
    if low in HINDI_WORDS:
        return "hi"
    if low in ENGLISH_SLANG or zipf(low, "en") >= 4.0:
        return "en"
    return "hi"


def cmi(langs) -> float:
    n_hi, n_en = langs.count("hi"), langs.count("en")
    n = n_hi + n_en
    return 0.0 if n == 0 else 100.0 * (1 - max(n_hi, n_en) / n)


def add_features(test: pd.DataFrame, train_vocab: set) -> pd.DataFrame:
    from wordfreq import zipf_frequency
    df = test.copy()
    toks = df.text_clean.apply(lambda t: TOKEN_RE.findall(t))
    words = toks.apply(lambda ts: [t.lower() for t in ts if re.fullmatch(r"[A-Za-z']+", t)])
    langs = toks.apply(lambda ts: [word_lang(t, zipf_frequency) for t in ts])
    df["n_words"] = words.apply(len)
    df["lang_seq"] = langs.apply(" ".join)
    df["frac_english"] = langs.apply(lambda l: l.count("en") / max(1, l.count("en") + l.count("hi")))
    df["cmi"] = langs.apply(cmi)
    df["oov_rate"] = words.apply(lambda w: np.mean([x not in train_vocab for x in w]) if w else 0.0)
    df["has_negation"] = words.apply(lambda w: any(x in NEGATIONS for x in w))
    df["has_elongation"] = df.text.apply(lambda t: bool(re.search(r"([A-Za-z])\1\1", t)))
    df["has_question"] = df.text.str.contains(r"\?", regex=True)
    df["has_emoji"] = df.text.str.contains(r"<3|[:;]-?[()DPpO*]|:'\(|xD|XD", regex=True)
    return df


def slice_definitions(df):
    q = df.cmi
    return [
        ("Length", "short (<=6 words)", df.n_words <= 6),
        ("Length", "medium (7-15)", df.n_words.between(7, 15)),
        ("Length", "long (>15)", df.n_words > 15),
        ("Code-mixing (CMI)", "monolingual (CMI=0)", q == 0),
        ("Code-mixing (CMI)", "light (0<CMI<=25)", (q > 0) & (q <= 25)),
        ("Code-mixing (CMI)", "heavy (CMI>25)", q > 25),
        ("Unseen words (OOV)", "none", df.oov_rate == 0),
        ("Unseen words (OOV)", "some (<=20%)", (df.oov_rate > 0) & (df.oov_rate <= 0.2)),
        ("Unseen words (OOV)", "many (>20%)", df.oov_rate > 0.2),
        ("Phenomenon", "negation", df.has_negation),
        ("Phenomenon", "elongated word", df.has_elongation),
        ("Phenomenon", "question mark", df.has_question),
        ("Phenomenon", "emoticon", df.has_emoji),
        ("All", "all test", pd.Series(True, index=df.index)),
    ]


def slice_table(df, model_preds):
    rows = []
    for group, name, mask in slice_definitions(df):
        sub = df[mask]
        row = {"group": group, "slice": name, "n": int(mask.sum()),
               "pct_negative": (sub.label == 0).mean() if len(sub) else np.nan}
        for m, pred in model_preds.items():
            p = pred[mask.values]
            row[f"{m} acc"] = (p == sub.label.values).mean() if len(sub) else np.nan
            row[f"{m} macro-F1"] = (f1_score(sub.label, p, average="macro", zero_division=0)
                                    if len(sub) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_slices(st: pd.DataFrame, models):
    st = st[st.group != "All"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(st) * max(1, len(models) / 2) + 1.5))
    k = len(models)
    h = 0.8 / k
    y = np.arange(len(st))
    for i, m in enumerate(models):
        vals = st[f"{m} acc"].values
        pos = y - 0.4 + h * (i + 0.5)
        ax.barh(pos, vals, height=h, color=C_BLUE if i == 0 else color_for(m, i), label=m, edgecolor="white", linewidth=1.5)
        for yp, v in zip(pos, vals):
            ax.text(v + 0.005, yp, f"{v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [f"{g}: {s}  (n={n})" for g, s, n in zip(st.group, st.slice, st.n)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test accuracy within slice")
    ax.set_title("Accuracy by linguistic slice")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_slices.png", dpi=200)
    plt.close(fig)


def confusion_types(y, pred):
    names = {(0, 1): "Neg<->Neu", (1, 0): "Neg<->Neu", (1, 2): "Neu<->Pos",
             (2, 1): "Neu<->Pos", (0, 2): "Neg<->Pos", (2, 0): "Neg<->Pos"}
    errs = [names[(a, b)] for a, b in zip(y, pred) if a != b]
    s = pd.Series(errs).value_counts()
    return {k: int(s.get(k, 0)) for k in ("Neu<->Pos", "Neg<->Neu", "Neg<->Pos")} | {"total_errors": len(errs)}


def fertility(df):
    """Sub-word pieces per word, by tokenizer and by (approximate) word language."""
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency
    tokzs = {"mBERT": "bert-base-multilingual-cased", "MuRIL": "google/muril-base-cased",
             "IndicBERT": "ai4bharat/indic-bert"}
    words = [(w, word_lang(w, zipf_frequency)) for t in df.text_clean
             for w in re.findall(r"[A-Za-z']+", t)]
    rows = []
    for name, hf in tokzs.items():
        try:
            kw = {"keep_accents": True} if "indic" in hf else {}
            tok = AutoTokenizer.from_pretrained(hf, **kw)
        except Exception as e:
            print(f"[skip] {name} tokenizer not available: {e}")
            continue
        unk = tok.unk_token
        for lang in ("hi", "en", "all"):
            ws = [w for w, l in words if lang == "all" or l == lang]
            pieces = [tok.tokenize(w) for w in ws]
            n = np.array([len(p) for p in pieces])
            rows.append({"tokenizer": name, "word_lang": lang, "n_words": len(ws),
                         "pieces_per_word": n.mean(), "pct_split_3plus": (n >= 3).mean() * 100,
                         "pct_unchanged": (n == 1).mean() * 100,
                         "pct_unk": np.mean([unk in p for p in pieces]) * 100})
        ex = ["bahut", "bohot", "dhannnsu", "pyaar", "mubarak", "zabardast", "awesome"]
        print(f"{name:>9}: " + " | ".join(f"{w}->{tok.tokenize(w)}" for w in ex))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fertility", action="store_true", help="tokenizer fertility (needs internet)")
    ap.add_argument("--n_manual", type=int, default=100)
    args = ap.parse_args()

    train, _, test = load_splits()
    train_vocab = {w.lower() for t in train.text_clean for w in re.findall(r"[A-Za-z']+", t)}
    df = add_features(test, train_vocab)

    base = pd.read_csv(RESULTS_DIR / "baseline_results.csv")
    best_cls = base.loc[base.validation_macro_f1.idxmax(), "model"]
    preds = {best_cls: load_preds(best_cls)}
    runs_path = RESULTS_DIR / "transformer_runs.csv"
    sel_info = {}
    if runs_path.exists():
        final = select_transformer_runs(pd.read_csv(runs_path))
        for m in ordered_models(final.model.unique()):
            g = final[final.model == m]
            sel = g.loc[g.validation_macro_f1.idxmax()]
            sel_info[m] = sel
            preds[m] = load_preds(sel.run_id)
    for m, p in preds.items():
        assert (p.id.values == df.id.values).all(), f"{m}: predictions not aligned with test split"
    pred_arr = {m: p.pred.values for m, p in preds.items()}
    models = list(preds)
    best_tf = max(sel_info, key=lambda m: sel_info[m].validation_macro_f1) if sel_info else None

    md = ["# Phase 7 — Error analysis\n",
          f"Best classical (by val): **{best_cls}**." +
          (f" Best transformer (by val): **{best_tf}** ({sel_info[best_tf].run_id})." if best_tf else ""),
          "\nLanguage ID is an approximate frequency-dictionary heuristic (see script docstring).\n"]

    # ---- 1. corpus code-mixing profile ----
    md += ["## Code-mixing profile of the test set\n",
           f"- Mean CMI: {df.cmi.mean():.1f}; monolingual (CMI=0): {(df.cmi == 0).mean():.1%}; "
           f"heavy (CMI>25): {(df.cmi > 25).mean():.1%}",
           f"- Mean share of English words per comment: {df.frac_english.mean():.1%}",
           f"- Comments with ≥1 word unseen in training: {(df.oov_rate > 0).mean():.1%} "
           f"(mean OOV rate {df.oov_rate.mean():.1%}) — a proxy for spelling variation\n"]

    # ---- 2. slices ----
    st = slice_table(df, pred_arr)
    st.to_csv(OUT_DIR / "slice_performance.csv", index=False)
    fig_slices(st, models)
    cols = ["group", "slice", "n"] + [f"{m} acc" for m in models]
    md += ["## Accuracy by slice\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in st.iterrows():
        md.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")

    # ---- 3. confusion types ----
    ct = pd.DataFrame([{"model": m, **confusion_types(df.label.values, p)} for m, p in pred_arr.items()])
    ct.to_csv(OUT_DIR / "confusion_types.csv", index=False)
    md += ["\n## Which classes get confused (count of test errors)\n",
           "| Model | Neu↔Pos | Neg↔Neu | Neg↔Pos | Total errors |", "|---|---|---|---|---|"]
    md += [f"| {r.model} | {r['Neu<->Pos']} | {r['Neg<->Neu']} | {r['Neg<->Pos']} | {r.total_errors} |"
           for _, r in ct.iterrows()]

    # ---- 4. disagreement ----
    if best_tf:
        c_ok = pred_arr[best_cls] == df.label.values
        t_ok = pred_arr[best_tf] == df.label.values
        quad = {"both correct": c_ok & t_ok, "only transformer correct": ~c_ok & t_ok,
                "only classical correct": c_ok & ~t_ok, "both wrong": ~c_ok & ~t_ok}
        dis = df[["id", "text", "label"]].copy()
        dis["label"] = dis.label.map(dict(enumerate(LABEL_NAMES)))
        dis["classical_pred"] = pd.Series(pred_arr[best_cls]).map(dict(enumerate(LABEL_NAMES))).values
        dis["transformer_pred"] = pd.Series(pred_arr[best_tf]).map(dict(enumerate(LABEL_NAMES))).values
        dis["outcome"] = np.select(list(quad.values()), list(quad.keys()), default="")
        dis["cmi"], dis["oov_rate"] = df.cmi.round(1), df.oov_rate.round(2)
        dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
        md += [f"\n## {best_cls} vs {best_tf}: who gets what right\n",
               "| Outcome | n | % | mean OOV rate | mean CMI |", "|---|---|---|---|---|"]
        for k, m in quad.items():
            md.append(f"| {k} | {m.sum()} | {m.mean():.1%} | {df.oov_rate[m].mean():.1%} | {df.cmi[m].mean():.1f} |")
        md.append("\nExamples (see `error_analysis/disagreements.csv` for all):\n")
        for k in ("only transformer correct", "only classical correct", "both wrong"):
            ex = dis[dis.outcome == k].head(4)
            md.append(f"**{k}:**")
            md += [f"- \"{r.text[:120]}\" — gold {r.label}; classical {r.classical_pred}; "
                   f"transformer {r.transformer_pred}" for r in ex.itertuples()]
            md.append("")

        # ---- 5. confident errors ----
        tp = preds[best_tf]
        prob_cols = [c for c in tp.columns if c.startswith("prob_")]
        if prob_cols:
            conf = tp[prob_cols].max(axis=1)
            ce = tp.assign(confidence=conf)[~tp.correct].sort_values("confidence", ascending=False).head(30)
            ce = ce.assign(gold=ce.label.map(dict(enumerate(LABEL_NAMES))),
                           predicted=ce.pred.map(dict(enumerate(LABEL_NAMES))))
            ce[["id", "text", "gold", "predicted", "confidence"]].to_csv(
                OUT_DIR / "confident_errors.csv", index=False)
            md += [f"## {best_tf}: most confident errors (sarcasm / label-noise candidates)\n"]
            md += [f"- ({r.confidence:.2f}) \"{r.text[:120]}\" — gold {r.gold}, predicted {r.predicted}"
                   for r in ce.head(8).itertuples()]

        # ---- 7. manual annotation sheet ----
        errs = dis[dis.outcome.isin(["both wrong", "only classical correct"])]
        n = min(args.n_manual, len(errs))
        shuffled = errs.sample(frac=1, random_state=0)
        quota = (shuffled.label.value_counts(normalize=True) * n).round().clip(lower=1).astype(int)
        sheet = pd.concat([shuffled[shuffled.label == lab].head(q) for lab, q in quota.items()]).head(n)
        sheet = sheet.assign(category="", notes="")
        sheet.to_csv(OUT_DIR / "manual_annotation_sheet.csv", index=False)
        md += ["\n## Manual annotation\n",
               f"`error_analysis/manual_annotation_sheet.csv` holds {len(sheet)} sampled {best_tf} errors. "
               "Suggested `category` values: sarcasm/irony, spelling-variant, negation, "
               "mixed/implicit sentiment, needs-context (named entity / event), label-noise, other."]

    # ---- 6. tokenizer fertility ----
    if args.fertility:
        fe = fertility(df)
        if len(fe):
            fe.to_csv(OUT_DIR / "tokenizer_fertility.csv", index=False)
            md += ["\n## Tokenizer fertility (sub-word pieces per word, test set)\n",
                   "| Tokenizer | Words | Pieces/word | % split into ≥3 | % kept whole | % [UNK] |",
                   "|---|---|---|---|---|---|"]
            md += [f"| {r.tokenizer} | {r.word_lang} ({r.n_words}) | {r.pieces_per_word:.2f} | "
                   f"{r.pct_split_3plus:.1f} | {r.pct_unchanged:.1f} | {r.pct_unk:.2f} |"
                   for r in fe.itertuples()]

    (RESULTS_DIR / "phase7_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {RESULTS_DIR / 'phase7_summary.md'}, {OUT_DIR}/, {FIGURES_DIR / 'error_slices.png'}")


if __name__ == "__main__":
    main()
