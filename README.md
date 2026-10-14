# Hinglish Sentiment Analysis: Traditional ML vs. Multilingual Transformers

Comparing TF-IDF + classical ML against mBERT / MuRIL / IndicBERT on
Hindi-English code-mixed (Hinglish) sentiment classification, with a focus
on **where** each approach fails (transliteration variants, sarcasm,
script-mixing).

## Dataset decision (Phase 2)

**Chosen dataset: IIIT-H Hindi-English Code-Mixed Sentiment Dataset
("IIITH_Codemixed")**

| Option considered | Verdict |
|---|---|
| SAIL 2017 (ICON shared task) | The official GitHub repo (`brajagopalcse/SAIL_CodeMixed-ICON-2017`) only hosts the **evaluation scripts**, not the raw tweets — the actual train/test data has to be requested from the shared-task organizers by email, with no guaranteed/quick turnaround. Not practical on a course-project timeline. |
| **IIIT-H Codemixed (chosen)** | Directly downloadable, no request needed. 3,879 Hindi-English code-mixed Facebook comments (from Salman Khan / Narendra Modi public pages), 3-class labels (negative/neutral/positive), inter-annotator-agreed. Introduced in Joshi et al., COLING 2016 ("Towards Sub-Word Level Compositions for Sentiment Analysis of Hi-En Code-Mixed Text") and reused since in the GLUECoS benchmark and the HIT (ACL 2021) codemixed-representation paper — so you have published baselines to cite and compare against. |
| Random Kaggle/HF Twitter dumps | Mostly small (hundreds of rows), unlicensed/unverified, no annotator-agreement info, not tied to a paper you can cite as ground truth. Kept as an optional *augmentation* source, not primary. |

This satisfies your "option 1 or 3" framing: it has the academic rigor and
citability of the SAIL-era benchmarks (option 1), while being as
practically downloadable as a GitHub/Kaggle dataset (option 3).

Source file: `raw.githubusercontent.com/drimpossible/Sub-word-LSTM/master/Data/IIITH_Codemixed.txt`
Format: `id \t text \t label \t label` (label repeated; 0=negative, 1=neutral, 2=positive)

Class distribution (checked directly):

| Label | Count | % |
|---|---|---|
| Negative (0) | 570 | 14.7% |
| Neutral (1) | 1,957 | 50.4% |
| Positive (2) | 1,352 | 34.9% |

→ **Imbalanced, neutral-heavy.** This is exactly why the roadmap calls for
macro-F1 (not accuracy) as your headline metric, and it's a natural thread
to carry into your error analysis in Phase 6.

**On size (3,879 rows):** this is workable for TF-IDF/SVM baselines and
for *fine-tuning* (not pretraining) mBERT/MuRIL/IndicBERT, but it's on the
smaller side for a transformer to shine. Two options, noted for Phase 3/5:
1. Ship the project on IIITH alone (fully defensible, matches the
   literature it comes from).
2. Optionally pull in one more dataset for augmentation/robustness testing
   — `L3Cube-HingCorpus`/GLUECoS sentiment split or a Kaggle Hinglish
   Twitter sentiment set — and treat it as an extra "generalization" test
   set in Phase 6 (does a model trained on Facebook comments hold up on
   tweets?). This is an easy way to make your error analysis section
   stronger and is flagged as a stretch goal in `src/download_data.py`.

## Project structure

```
hinglish-sentiment-analysis/
├── README.md
├── app.py                       # Phase 8: Gradio demo
├── requirements.txt
├── data/
│   ├── raw/                     # untouched source files
│   │   └── IIITH_Codemixed.txt
│   └── processed/                # outputs of src/preprocessing.py + split_data.py
│       ├── cleaned.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── download_data.py          # Phase 2: fetch + sanity-check raw data
│   ├── preprocessing.py          # Phase 3: cleaning/normalization pipeline
│   ├── split_data.py             # Phase 3: stratified 70/15/15 split
│   ├── baseline_models.py        # Phase 4: TF-IDF + SVM/NB/LR/RF
│   ├── eval_utils.py             # shared metrics / prediction-saving helpers
│   ├── transformer_models.py     # Phase 5: fine-tune mBERT/MuRIL/IndicBERT
│   ├── refresh_clean_text.py     # Phase 4.5: re-clean text after emoticon fix (same split)
│   ├── evaluate.py               # Phase 6: comparison, significance tests, figures
│   └── error_analysis.py         # Phase 7: slices, CMI, disagreements, fertility
├── notebooks/
│   ├── phase5_transformers_colab.ipynb  # Phases 5–7 on Colab GPU
│   └── phase8_demo_colab.ipynb          # Phase 8 demo on Colab
├── outputs/
│   ├── figures/                  # confusion matrices, class-distribution plots
│   └── models/                   # saved model checkpoints
└── reports/                      # write-up drafts (Phase 7)
```

## How to run (so far)

```bash
pip install -r requirements.txt

# Phase 2
python src/download_data.py

# Phase 3
python src/preprocessing.py
python src/split_data.py

# Phase 4 / 4.5 (classical baselines, word + char TF-IDF)
python src/baseline_models.py

# Phase 5 (GPU) — run notebooks/phase5_transformers_colab.ipynb on Colab, or:
python src/transformer_models.py --model muril --seeds 42 --lrs 2e-5 3e-5 5e-5
python src/transformer_models.py --model muril --seeds 43 44 --lrs 3e-5
```

```bash
# Phase 6–7 (after Phase 5; the Colab notebook runs these too)
python src/evaluate.py                    # comparison table, McNemar + bootstrap, figures
python src/error_analysis.py --fertility  # slices, CMI, disagreements, tokenizer fertility
```

Summaries for the report: `outputs/results/phase6_summary.md` and
`outputs/results/phase7_summary.md`. The demo app (Phase 8) is next.

## Phase 4 results (class-imbalance-aware baselines)

TF-IDF (1–2 grams) + four classical classifiers, all trained with
`class_weight="balanced"` (Naive Bayes via balanced `sample_weight`,
since `MultinomialNB` has no `class_weight` param) to counter the
50/35/15 neutral/positive/negative split — see "Dataset decision" above.
Without this, every model over-predicted "neutral" and the negative
class in particular suffered (Linear SVM's negative-class recall was
~42% unweighted vs. 50%+ balanced; Naive Bayes' test macro-F1 rose from
0.515 to 0.642 after weighting). Full numbers: `outputs/results/baseline_results.csv`.

| Model | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Naive Bayes | 0.663 | **0.642** |
| Logistic Regression | 0.656 | 0.631 |
| Linear SVM | 0.653 | 0.614 |
| Random Forest | 0.639 | 0.609 |

Per-class precision/recall/F1 for every model is in
`outputs/results/<model>_classification_report.csv`, and both raw-count
and row-normalized confusion matrices are in `outputs/figures/`. Fitted
models + the TF-IDF vectorizer are pickled to `outputs/models/` so Phase
6's error analysis can reload them without retraining.

## Phase 4.5 — character n-gram baselines

Added TF-IDF over character 2–5-grams (`char_wb`). This is the classical
counterpart of the base paper's sub-word idea: char n-grams share
features across romanized spelling variants (*bahut / bohot / bhot*).
C was chosen on the **validation** set (LR: C=4, SVM: C=0.25).

| Model | Features | Test Acc | Test Macro-F1 | F1 Neg | F1 Neu | F1 Pos |
|---|---|---|---|---|---|---|
| **Char Linear SVM** | char 2–5 | **0.696** | **0.673** | 0.608 | 0.737 | 0.675 |
| Char Logistic Regression | char 2–5 | 0.679 | 0.659 | 0.599 | 0.717 | 0.662 |
| Naive Bayes | word 1–2 | 0.660 | 0.639 | 0.540 | 0.704 | 0.672 |
| Logistic Regression | word 1–2 | 0.651 | 0.625 | 0.524 | 0.693 | 0.658 |
| Linear SVM | word 1–2 | 0.653 | 0.616 | 0.512 | 0.706 | 0.631 |
| Random Forest | word 1–2 | 0.620 | 0.597 | 0.516 | 0.675 | 0.600 |

For reference, the base paper (Joshi et al., 2016, Table 5) reports
MNB unigram 0.614 F1 and Subword-LSTM 0.658 F1 on this dataset (their
split is 64/16/20, so the comparison is approximate). Both char models
match or beat the paper's best system, and this is the bar the
transformers must clear.

**Preprocessing bug fixed (Phase 4.5).** In `remove_noise()` the emoticon
placeholder `__EMO0__` lost its underscores to the character whitelist, so
every emoticon (`<3`, `:)`, `:(` ...) collapsed into one token `EMO0`.
The placeholder is now whitelist-safe, and `src/refresh_clean_text.py`
recomputed `text_clean` for the existing splits **without re-splitting**
(same ids in train/val/test). 145 of 3,878 comments changed. The effect on
the classical baselines is within noise (±0.01 macro-F1), because the
word tokenizer drops symbols anyway, but transformers now see the real
emoticons. The table above is after the fix.

Every model now also saves per-example predictions to
`outputs/predictions/<model>_{val,test}.csv` (shared format via
`src/eval_utils.py`), which Phase 6–7 use.

## Phase 5 — transformer fine-tuning (`src/transformer_models.py`)

| Alias | Model | HF id |
|---|---|---|
| `mbert` | mBERT | `bert-base-multilingual-cased` |
| `muril` | MuRIL | `google/muril-base-cased` |
| `indicbert` | IndicBERT | `ai4bharat/indic-bert` |

Setup: class-weighted cross-entropy (same balancing as the baselines),
AdamW + linear warmup (10%), max_len 128, batch 16, up to 10 epochs with
early stopping (patience 3) on validation macro-F1, fp16 on GPU.
(Run 1 used 5 epochs / patience 2: MuRIL was still improving at epoch 5,
train loss 0.55–0.83, so run 2 trains longer. Run-1 results are kept in
`outputs/archive_run1_5epochs/`.)
IndicBERT (`ai4bharat/indic-bert`) is a gated model: accept its terms on
Hugging Face and log in (`huggingface_hub.login()`) before training.
Protocol: LR sweep {3e-5, 5e-5} with seed 42 (run 1 also tried 2e-5; never best) → best LR per model by
validation macro-F1 → seeds 43, 44 at that LR → report mean ± std over
3 seeds. Optional ablation: `--text_col text` (raw) vs `text_clean`.

Outputs: `outputs/results/transformer_runs.csv` (one row per run),
`outputs/results/logs/*_history.csv` (per-epoch curves),
`outputs/predictions/`, `outputs/figures/`, and the best checkpoint per
model (by validation F1) in `outputs/models/<alias>_text_clean_best/`.

## Phase 8 — Demo app (`app.py`, Gradio)

Type a Roman-script Hinglish comment and see the char n-gram SVM and the
best fine-tuned transformer side by side, with class scores, whether they
agree, and the cleaned text they saw.

```bash
pip install gradio
python app.py                                  # auto-finds outputs/models/*_best
python app.py --model_dir path/to/best_model --share
```

On Colab: `notebooks/phase8_demo_colab.ipynb` loads the best checkpoint that
the Phase 5–7 notebook copied to `MyDrive/hinglish_outputs_run2/best_model`
and prints a public link.
