# Phase 7 — Error analysis

Best classical (by val): **Char Linear SVM**.

Language ID is an approximate frequency-dictionary heuristic (see script docstring).

## Code-mixing profile of the test set

- Mean CMI: 16.5; monolingual (CMI=0): 28.7%; heavy (CMI>25): 25.9%
- Mean share of English words per comment: 17.2%
- Comments with ≥1 word unseen in training: 66.8% (mean OOV rate 14.2%) — a proxy for spelling variation

## Accuracy by slice

| group | slice | n | Char Linear SVM acc |
|---|---|---|---|
| Length | short (<=6 words) | 189 | 0.778 |
| Length | medium (7-15) | 276 | 0.641 |
| Length | long (>15) | 117 | 0.692 |
| Code-mixing (CMI) | monolingual (CMI=0) | 167 | 0.701 |
| Code-mixing (CMI) | light (0<CMI<=25) | 264 | 0.678 |
| Code-mixing (CMI) | heavy (CMI>25) | 151 | 0.722 |
| Unseen words (OOV) | none | 193 | 0.777 |
| Unseen words (OOV) | some (<=20%) | 235 | 0.651 |
| Unseen words (OOV) | many (>20%) | 154 | 0.662 |
| Phenomenon | negation | 116 | 0.638 |
| Phenomenon | elongated word | 55 | 0.727 |
| Phenomenon | question mark | 33 | 0.697 |
| Phenomenon | emoticon | 23 | 0.652 |
| All | all test | 582 | 0.696 |

## Which classes get confused (count of test errors)

| Model | Neu↔Pos | Neg↔Neu | Neg↔Pos | Total errors |
|---|---|---|---|---|
| Char Linear SVM | 106 | 49 | 22 | 177 |
