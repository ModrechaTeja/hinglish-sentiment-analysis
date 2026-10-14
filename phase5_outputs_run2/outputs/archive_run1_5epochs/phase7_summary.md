# Phase 7 — Error analysis

Best classical (by val): **Char Linear SVM**. Best transformer (by val): **MuRIL** (muril_lr3e-05_seed42_text_clean).

Language ID is an approximate frequency-dictionary heuristic (see script docstring).

## Code-mixing profile of the test set

- Mean CMI: 16.5; monolingual (CMI=0): 28.7%; heavy (CMI>25): 25.9%
- Mean share of English words per comment: 17.2%
- Comments with ≥1 word unseen in training: 66.8% (mean OOV rate 14.2%) — a proxy for spelling variation

## Accuracy by slice

| group | slice | n | Char Linear SVM acc | mBERT acc | MuRIL acc |
|---|---|---|---|---|---|
| Length | short (<=6 words) | 189 | 0.778 | 0.783 | 0.767 |
| Length | medium (7-15) | 276 | 0.641 | 0.594 | 0.627 |
| Length | long (>15) | 117 | 0.692 | 0.624 | 0.667 |
| Code-mixing (CMI) | monolingual (CMI=0) | 167 | 0.701 | 0.671 | 0.689 |
| Code-mixing (CMI) | light (0<CMI<=25) | 264 | 0.678 | 0.621 | 0.636 |
| Code-mixing (CMI) | heavy (CMI>25) | 151 | 0.722 | 0.722 | 0.748 |
| Unseen words (OOV) | none | 193 | 0.777 | 0.767 | 0.777 |
| Unseen words (OOV) | some (<=20%) | 235 | 0.651 | 0.604 | 0.638 |
| Unseen words (OOV) | many (>20%) | 154 | 0.662 | 0.617 | 0.623 |
| Phenomenon | negation | 116 | 0.638 | 0.586 | 0.595 |
| Phenomenon | elongated word | 55 | 0.727 | 0.691 | 0.727 |
| Phenomenon | question mark | 33 | 0.697 | 0.758 | 0.758 |
| Phenomenon | emoticon | 23 | 0.652 | 0.652 | 0.652 |
| All | all test | 582 | 0.696 | 0.662 | 0.680 |

## Which classes get confused (count of test errors)

| Model | Neu↔Pos | Neg↔Neu | Neg↔Pos | Total errors |
|---|---|---|---|---|
| Char Linear SVM | 106 | 49 | 22 | 177 |
| mBERT | 117 | 51 | 29 | 197 |
| MuRIL | 114 | 51 | 21 | 186 |

## Char Linear SVM vs MuRIL: who gets what right

| Outcome | n | % | mean OOV rate | mean CMI |
|---|---|---|---|---|
| both correct | 344 | 59.1% | 12.6% | 17.0 |
| only transformer correct | 52 | 8.9% | 15.2% | 19.5 |
| only classical correct | 61 | 10.5% | 16.0% | 16.7 |
| both wrong | 125 | 21.5% | 17.2% | 13.9 |

Examples (see `error_analysis/disagreements.csv` for all):

**only transformer correct:**
- "suna ha ap sub comments parhtay hain :* <3" — gold Neutral; classical Positive; transformer Neutral
- "Modije mera request hai ke ap apna whatsapp no de take mai live pic apko dekha saku" — gold Neutral; classical Positive; transformer Neutral
- "friends tum sab salman khan ku thaknah chati hu. ...???" — gold Neutral; classical Positive; transformer Neutral
- "Aapne aisa q kha salman ki aap bina muslmn ki apni pix hit kra doge" — gold Negative; classical Neutral; transformer Negative

**only classical correct:**
- "Salman bhai i knw ur a big celebrity par hakk toh hume bhi milna chaye apse baat krne ka !" — gold Neutral; classical Neutral; transformer Positive
- "bhai ap muslim bina kesi hit karoge ye films. ." — gold Negative; classical Negative; transformer Positive
- "Javab nahi hai" — gold Neutral; classical Neutral; transformer Negative
- "nic song . bhi jan" — gold Positive; classical Positive; transformer Neutral

**both wrong:**
- "Sala kutta kamena amara time west kara tia salman" — gold Positive; classical Negative; transformer Negative
- "Main ne tu kbi in romurs py yakeen kiya hi ni tha Bhai jaan" — gold Negative; classical Neutral; transformer Neutral
- "Sllam bhai jaan EIDmubarakh." — gold Neutral; classical Positive; transformer Positive
- "ye tumse jalne walo k kaam h bhai" — gold Negative; classical Positive; transformer Neutral

## MuRIL: most confident errors (sarcasm / label-noise candidates)

- (0.81) "nai ata kamina salman khan sabko pagal banara lolz" — gold Neutral, predicted Negative
- (0.81) "Koi bhi yeh chutiyo Salman Khan picture nahi dekhne musalmano be care ful picture ko flop karne ha" — gold Positive, predicted Negative
- (0.81) "abe salman khan yaha logo ko bewakuf bana raha he aur log ban b rahe he per me nahi banoonga samjha be bajrangi" — gold Neutral, predicted Negative
- (0.81) "Kuch to bto koi" — gold Neutral, predicted Negative
- (0.81) "Shadi kr lo wrna firr na hogi" — gold Neutral, predicted Negative
- (0.81) "akele modi kya ker sakte hai.........akela aadmi pahad nhi khod sakta" — gold Neutral, predicted Negative
- (0.81) "Sala kutta kamena amara time west kara tia salman" — gold Positive, predicted Negative
- (0.81) "paye kuppi puthe vichil" — gold Neutral, predicted Negative

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
