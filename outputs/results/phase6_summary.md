# Phase 6 — Model comparison

Test set: 582 examples. Model selection on validation macro-F1 only. Transformers: mean ± std over seeds of the final config; 95% CI is a bootstrap over test examples for the selected run (classical: the single run).

| Model | Family | Config | Seeds | Val macro-F1 | Test acc | Test macro-F1 | 95% CI | F1 Neg | F1 Neu | F1 Pos |
|---|---|---|---|---|---|---|---|---|---|---|
| Char Linear SVM | classical | classical (char tf-idf) | 1 | 0.7210 | 0.6959 | **0.6732** | [0.630, 0.714] | 0.608 | 0.737 | 0.675 |
| Char Logistic Regression | classical | classical (char tf-idf) | 1 | 0.7169 | 0.6787 | **0.6591** | [0.616, 0.699] | 0.599 | 0.717 | 0.662 |
| Naive Bayes | classical | classical (word tf-idf) | 1 | 0.6775 | 0.6598 | **0.6387** | [0.596, 0.679] | 0.540 | 0.704 | 0.672 |
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
