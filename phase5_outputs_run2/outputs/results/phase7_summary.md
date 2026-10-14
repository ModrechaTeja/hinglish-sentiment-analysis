# Phase 7 — Error analysis

Best classical (by val): **Char Linear SVM**. Best transformer (by val): **MuRIL** (muril_lr3e-05_seed42_text_clean).

Language ID is an approximate frequency-dictionary heuristic (see script docstring).

## Code-mixing profile of the test set

- Mean CMI: 16.5; monolingual (CMI=0): 28.7%; heavy (CMI>25): 25.9%
- Mean share of English words per comment: 17.2%
- Comments with ≥1 word unseen in training: 66.8% (mean OOV rate 14.2%) — a proxy for spelling variation

## Accuracy by slice

| group | slice | n | Char Linear SVM acc | mBERT acc | MuRIL acc | IndicBERT acc |
|---|---|---|---|---|---|---|
| Length | short (<=6 words) | 189 | 0.778 | 0.730 | 0.762 | 0.709 |
| Length | medium (7-15) | 276 | 0.641 | 0.678 | 0.649 | 0.565 |
| Length | long (>15) | 117 | 0.692 | 0.590 | 0.632 | 0.547 |
| Code-mixing (CMI) | monolingual (CMI=0) | 167 | 0.701 | 0.665 | 0.671 | 0.599 |
| Code-mixing (CMI) | light (0<CMI<=25) | 264 | 0.678 | 0.663 | 0.655 | 0.572 |
| Code-mixing (CMI) | heavy (CMI>25) | 151 | 0.722 | 0.715 | 0.742 | 0.682 |
| Unseen words (OOV) | none | 193 | 0.777 | 0.767 | 0.777 | 0.699 |
| Unseen words (OOV) | some (<=20%) | 235 | 0.651 | 0.630 | 0.643 | 0.549 |
| Unseen words (OOV) | many (>20%) | 154 | 0.662 | 0.636 | 0.623 | 0.584 |
| Phenomenon | negation | 116 | 0.638 | 0.629 | 0.586 | 0.534 |
| Phenomenon | elongated word | 55 | 0.727 | 0.691 | 0.691 | 0.655 |
| Phenomenon | question mark | 33 | 0.697 | 0.667 | 0.727 | 0.606 |
| Phenomenon | emoticon | 23 | 0.652 | 0.652 | 0.696 | 0.652 |
| All | all test | 582 | 0.696 | 0.677 | 0.682 | 0.608 |

## Which classes get confused (count of test errors)

| Model | Neu↔Pos | Neg↔Neu | Neg↔Pos | Total errors |
|---|---|---|---|---|
| Char Linear SVM | 106 | 49 | 22 | 177 |
| mBERT | 102 | 55 | 31 | 188 |
| MuRIL | 111 | 55 | 19 | 185 |
| IndicBERT | 126 | 66 | 36 | 228 |

## Char Linear SVM vs MuRIL: who gets what right

| Outcome | n | % | mean OOV rate | mean CMI |
|---|---|---|---|---|
| both correct | 342 | 58.8% | 12.8% | 17.1 |
| only transformer correct | 55 | 9.5% | 14.9% | 18.2 |
| only classical correct | 63 | 10.8% | 15.0% | 16.1 |
| both wrong | 122 | 21.0% | 17.4% | 14.4 |

Examples (see `error_analysis/disagreements.csv` for all):

**only transformer correct:**
- "suna ha ap sub comments parhtay hain :* <3" — gold Neutral; classical Positive; transformer Neutral
- "Modije mera request hai ke ap apna whatsapp no de take mai live pic apko dekha saku" — gold Neutral; classical Positive; transformer Neutral
- "friends tum sab salman khan ku thaknah chati hu. ...???" — gold Neutral; classical Positive; transformer Neutral
- "Main ne tu kbi in romurs py yakeen kiya hi ni tha Bhai jaan" — gold Negative; classical Neutral; transformer Negative

**only classical correct:**
- "Aaaaaaa jaaaaaoooooo bhaiiiiiiiiiiiiiiiiiii knha busy hooooo ham wait kr rhe h" — gold Neutral; classical Neutral; transformer Positive
- "Javab nahi hai" — gold Neutral; classical Neutral; transformer Negative
- "nic song . bhi jan" — gold Positive; classical Positive; transformer Neutral
- "Advance mai eid mobarak bhai" — gold Neutral; classical Neutral; transformer Positive

**both wrong:**
- "Sala kutta kamena amara time west kara tia salman" — gold Positive; classical Negative; transformer Negative
- "Sllam bhai jaan EIDmubarakh." — gold Neutral; classical Positive; transformer Positive
- "ye tumse jalne walo k kaam h bhai" — gold Negative; classical Positive; transformer Neutral
- "jo jo gali diya hai sabko sunna hoga ab" — gold Negative; classical Neutral; transformer Neutral

## MuRIL: most confident errors (sarcasm / label-noise candidates)

- (0.93) "akele modi kya ker sakte hai.........akela aadmi pahad nhi khod sakta" — gold Neutral, predicted Negative
- (0.93) "paye kuppi puthe vichil" — gold Neutral, predicted Negative
- (0.93) "Koi bhi yeh chutiyo Salman Khan picture nahi dekhne musalmano be care ful picture ko flop karne ha" — gold Positive, predicted Negative
- (0.93) "abe salman khan yaha logo ko bewakuf bana raha he aur log ban b rahe he per me nahi banoonga samjha be bajrangi" — gold Neutral, predicted Negative
- (0.93) "Sb kh rhe h fake id fake id . Aap ko proove krna chahiye aap hi ho." — gold Positive, predicted Negative
- (0.93) "Jo salman se pyar karte h wo islaam se pyar nhi karte" — gold Positive, predicted Negative
- (0.93) "Sala kutta kamena amara time west kara tia salman" — gold Positive, predicted Negative
- (0.93) "Sallman ki photos uplod krke fb ko khrab mt kro" — gold Neutral, predicted Negative

## Manual annotation

`error_analysis/manual_annotation_sheet.csv` holds 100 sampled MuRIL errors. Suggested `category` values: sarcasm/irony, spelling-variant, negation, mixed/implicit sentiment, needs-context (named entity / event), label-noise, other.

## Tokenizer fertility (sub-word pieces per word, test set)

| Tokenizer | Words | Pieces/word | % split into ≥3 | % kept whole | % [UNK] |
|---|---|---|---|---|---|
| mBERT | hi (5202) | 1.80 | 12.8 | 37.1 | 0.00 |
| mBERT | en (1162) | 1.18 | 3.8 | 86.0 | 0.00 |
| mBERT | all (6364) | 1.69 | 11.1 | 46.0 | 0.00 |
| MuRIL | hi (5202) | 1.48 | 8.4 | 63.8 | 0.00 |
| MuRIL | en (1162) | 1.05 | 0.4 | 95.1 | 0.00 |
| MuRIL | all (6364) | 1.40 | 6.9 | 69.5 | 0.00 |
| IndicBERT | hi (5202) | 1.91 | 16.4 | 30.8 | 0.00 |
| IndicBERT | en (1162) | 1.15 | 0.7 | 85.9 | 0.00 |
| IndicBERT | all (6364) | 1.77 | 13.5 | 40.9 | 0.00 |
