# Phase 6 — Model comparison

Test set: 582 examples. Model selection on validation macro-F1 only. Transformers: mean ± std over seeds of the final config; 95% CI is a bootstrap over test examples for the selected run (classical: the single run).

| Model | Family | Config | Seeds | Val macro-F1 | Test acc | Test macro-F1 | 95% CI | F1 Neg | F1 Neu | F1 Pos |
|---|---|---|---|---|---|---|---|---|---|---|
| Char Linear SVM | classical | classical (char tf-idf) | 1 | 0.7210 | 0.6959 | **0.6732** | [0.630, 0.714] | 0.608 | 0.737 | 0.675 |
| Char Logistic Regression | classical | classical (char tf-idf) | 1 | 0.7169 | 0.6787 | **0.6591** | [0.616, 0.699] | 0.599 | 0.717 | 0.662 |
| Naive Bayes | classical | classical (word tf-idf) | 1 | 0.6775 | 0.6598 | **0.6387** | [0.596, 0.679] | 0.540 | 0.704 | 0.672 |
| mBERT | transformer | lr=3e-05, text_clean | 3 | 0.6670 ± 0.0196 | 0.6707 ± 0.0144 | **0.6310 ± 0.0115** | [0.579, 0.667] | 0.536 | 0.735 | 0.622 |
| MuRIL | transformer | lr=3e-05, text_clean | 3 | 0.7033 ± 0.0180 | 0.6535 ± 0.0235 | **0.6270 ± 0.0251** | [0.608, 0.695] | 0.553 | 0.699 | 0.629 |
| Logistic Regression | classical | classical (word tf-idf) | 1 | 0.6673 | 0.6512 | **0.6250** | [0.582, 0.667] | 0.524 | 0.693 | 0.658 |
| Linear SVM | classical | classical (word tf-idf) | 1 | 0.6831 | 0.6529 | **0.6162** | [0.571, 0.659] | 0.512 | 0.706 | 0.630 |
| Random Forest | classical | classical (word tf-idf) | 1 | 0.6259 | 0.6203 | **0.5966** | [0.554, 0.638] | 0.516 | 0.675 | 0.599 |

## Base paper (Joshi et al., 2016, Table 5) — same dataset, 64/16/20 split

| Method | Accuracy | F1 |
|---|---|---|
| NBSVM (Unigram) | 0.5915 | 0.5335 |
| MNB (Unigram) | 0.6675 | 0.6143 |
| MNB (Tf-Idf) | 0.6353 | 0.4783 |
| SVM (Unigram) | 0.5760 | 0.5232 |
| Lexicon Lookup (Sharma et al.) | 0.5115 | 0.2520 |
| Char-LSTM | 0.5980 | 0.5110 |
| Subword-LSTM | 0.6970 | 0.6580 |

## Significance vs best classical model (Char Linear SVM)

| Comparison | Only classical correct | Only transformer correct | McNemar p | Δ macro-F1 [95% CI] | Bootstrap p | Seeds sig. (p<.05) |
|---|---|---|---|---|---|---|
| mBERT vs Char Linear SVM | 72 | 52 | 0.0876 | -0.0488 [-0.0926, -0.0070] | 0.9883 | 0/3 |
| MuRIL vs Char Linear SVM | 61 | 52 | 0.4519 | -0.0199 [-0.0607, +0.0208] | 0.8318 | 2/3 |

## Ablation — cleaned vs raw input text

| model | lr | input | n_seeds | val_macro_f1 | test_macro_f1 | test_f1_negative |
|---|---|---|---|---|---|---|
| MuRIL | 3e-05 | text_clean | 3 | 0.7033 ± 0.0180 | 0.6270 ± 0.0251 | 0.5528 ± 0.0357 |
| MuRIL | 3e-05 | text (raw) | 3 | 0.6985 ± 0.0291 | 0.6349 ± 0.0291 | 0.5461 ± 0.0438 |
