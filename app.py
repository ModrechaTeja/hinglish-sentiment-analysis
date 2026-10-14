"""
Phase 8 — Demo web app: Hinglish sentiment, classical vs transformer.

Type a Hindi-English code-mixed comment and see the predictions from
  * the best classical model (char 2-5-gram TF-IDF + Linear SVM), and
  * the best fine-tuned transformer (chosen on validation macro-F1),
side by side, with the cleaned text the models actually saw.

Usage
  python app.py                                   # auto-finds outputs/models/*_best
  python app.py --model_dir /path/to/best_model   # e.g. the folder copied to Drive
  python app.py --model_dir ... --share           # public link (needed on Colab)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402

MODELS_DIR = ROOT / "outputs" / "models"
RESULTS_DIR = ROOT / "outputs" / "results"
LABELS = ["Negative", "Neutral", "Positive"]

EXAMPLES = [
    "Trailer dhannnsu hai bhai",
    "Aisa PM naa hua hai aur naa hee hoga",
    "abe kutte tere se kon baat karega",
    "bhai ap muslim bina kesi hit karoge ye films",
    "Salman bhai i knw ur a big celebrity par hakk toh hume bhi milna chaye apse baat krne ka !",
    "movie bilkul bakwas thi, paisa barbaad",
    "Happy birthday bhai jaan, bahut saara pyaar <3",
    "Modi ji kal rally kahan hogi?",
]


def find_best_transformer_dir():
    best, best_f1 = None, -1
    for meta_path in glob.glob(str(MODELS_DIR / "*_best" / "run_meta.json")):
        meta = json.loads(Path(meta_path).read_text())
        if meta["val_macro_f1"] > best_f1:
            best, best_f1 = Path(meta_path).parent, meta["val_macro_f1"]
    return best


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Classical:
    def __init__(self):
        self.vec = joblib.load(MODELS_DIR / "tfidf_char_vectorizer.joblib")
        self.clf = joblib.load(MODELS_DIR / "char_linear_svm.joblib")
        self.name = "Char n-gram TF-IDF + Linear SVM"

    def predict(self, cleaned):
        scores = self.clf.decision_function(self.vec.transform([cleaned]))[0]
        # SVM margins are not probabilities; softmax only makes them comparable/visual.
        return dict(zip(LABELS, softmax(scores).tolist()))


class Transformer:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        meta_path = self.dir / "run_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.text_col = self.meta.get("text_col", "text_clean")
        self.max_len = self.meta.get("max_len", 128)
        self.name = self.meta.get("display", self.dir.name)

    @torch.no_grad()
    def predict(self, raw, cleaned):
        text = cleaned if self.text_col == "text_clean" else raw
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits[0].float(), dim=-1).cpu().numpy()
        return dict(zip(LABELS, probs.tolist()))


def scores_line(model_name):
    path = RESULTS_DIR / "final_comparison.csv"
    if not path.exists():
        return ""
    t = pd.read_csv(path)
    row = t[t.model == model_name]
    if row.empty:
        return ""
    r = row.iloc[0]
    std = r.get("test_macro_f1_std")
    s = f"{r.test_macro_f1:.3f}" + (f" ± {std:.3f}" if pd.notna(std) else "")
    return f"  (test macro-F1 {s})"


def build_app(classical, transformer):
    tf_name = transformer.name if transformer else "Transformer (not loaded)"

    def analyse(text):
        text = (text or "").strip()
        if not text:
            return {}, {}, "", ""
        cleaned = clean_text(text)
        c = classical.predict(cleaned)
        t = transformer.predict(text, cleaned) if transformer else {}
        c_lab = max(c, key=c.get)
        if t:
            t_lab = max(t, key=t.get)
            verdict = (f"**Both models say {c_lab}.**" if c_lab == t_lab else
                       f"**The models disagree:** classical → {c_lab}, {tf_name} → {t_lab}.")
        else:
            verdict = f"Classical → **{c_lab}** (no transformer checkpoint loaded)."
        return c, t, verdict, f"Cleaned text the models saw: `{cleaned}`"

    with gr.Blocks(title="Hinglish Sentiment") as demo:
        gr.Markdown(
            "# Hinglish Sentiment Analysis\n"
            "Hindi-English code-mixed sentiment: a classical char n-gram model vs a fine-tuned "
            "multilingual transformer, trained on the IIIT-H Hi-En code-mixed dataset "
            "(Facebook comments; Negative / Neutral / Positive)."
        )
        with gr.Row():
            inp = gr.Textbox(label="Comment (Roman-script Hinglish)", lines=3,
                             placeholder="e.g. Trailer dhannnsu hai bhai")
        btn = gr.Button("Analyse", variant="primary")
        verdict = gr.Markdown()
        with gr.Row():
            c_out = gr.Label(num_top_classes=3,
                             label=f"Classical: {classical.name}{scores_line(classical_row_name)}")
            t_out = gr.Label(num_top_classes=3,
                             label=f"Transformer: {tf_name}{scores_line(tf_name)}")
        cleaned = gr.Markdown(label="Cleaned text")
        gr.Examples(EXAMPLES, inputs=inp)
        gr.Markdown(
            "<small>Classical scores are a softmax over SVM margins (relative, not calibrated "
            "probabilities). Cleaning converts emoji to text and normalises common spelling variants.</small>"
        )
        btn.click(analyse, inp, [c_out, t_out, verdict, cleaned])
        inp.submit(analyse, inp, [c_out, t_out, verdict, cleaned])
    return demo


classical_row_name = "Char Linear SVM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="fine-tuned transformer folder")
    ap.add_argument("--share", action="store_true", help="public Gradio link (Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    classical = Classical()
    model_dir = args.model_dir or os.environ.get("HINGLISH_MODEL_DIR") or find_best_transformer_dir()
    transformer = None
    if model_dir:
        print(f"Loading transformer from {model_dir}")
        transformer = Transformer(model_dir)
    else:
        print("[warn] no transformer checkpoint found; running classical model only")

    build_app(classical, transformer).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
