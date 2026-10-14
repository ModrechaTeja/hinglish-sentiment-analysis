"""
Phase 5 — Fine-tuning multilingual Transformers (mBERT, MuRIL, IndicBERT)
for Hindi-English code-mixed sentiment classification.

Design choices (easy to defend in the write-up):
  * Same train/val/test split and same text column as the classical
    baselines, so the comparison is apples-to-apples.
  * Class-weighted cross-entropy (weights = n / (k * count_c)) — the
    transformer analogue of class_weight="balanced" used in Phase 4.
  * Model selection on VALIDATION macro-F1 only (early stopping + best
    epoch + learning-rate choice). The test set is touched once per run.
  * Multiple seeds per configuration -> report mean ± std, because with a
    ~580-example test set a single run can swing by 2-3 F1 points.
  * Plain PyTorch training loop (no HF Trainer) so it works unchanged
    across transformers 4.x / 5.x versions on Colab.

Outputs:
  outputs/results/transformer_runs.csv            one row per run (val + test metrics)
  outputs/results/logs/<run>_history.csv          per-epoch train loss / val F1
  outputs/predictions/<run>_{val,test}.csv        per-example predictions + probs
  outputs/figures/<run>_confusion_matrix*.png
  outputs/models/<alias>_<text_col>_best/         best run (by VAL macro-F1) per model

Usage (Colab, GPU):
  python src/transformer_models.py --model muril --seeds 42 --lrs 2e-5 3e-5 5e-5
  python src/transformer_models.py --model muril --seeds 43 44 --lrs 3e-5
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from eval_utils import (
    load_splits, compute_metrics, print_metrics, save_predictions,
    save_confusion_and_report, results_row, slugify,
    RESULTS_DIR, MODELS_DIR, LABEL_NAMES,
)

MODEL_ALIASES = {
    "mbert": ("mBERT", "bert-base-multilingual-cased"),
    "muril": ("MuRIL", "google/muril-base-cased"),
    "indicbert": ("IndicBERT", "ai4bharat/indic-bert"),
}

RUNS_CSV = RESULTS_DIR / "transformer_runs.csv"
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(hf_name: str):
    """IndicBERT (ALBERT + SentencePiece) must keep accents/diacritics."""
    kwargs = {"keep_accents": True} if "indic-bert" in hf_name else {}
    try:
        return AutoTokenizer.from_pretrained(hf_name, **kwargs)
    except Exception as e:  # fall back to the slow tokenizer if fast conversion fails
        print(f"[warn] fast tokenizer failed ({e}); retrying with use_fast=False")
        return AutoTokenizer.from_pretrained(hf_name, use_fast=False, **kwargs)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def make_collate(tokenizer):
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.long)
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return collate


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    all_probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(all_probs)
    return probs.argmax(axis=1), probs


def append_run_row(row: dict):
    df_new = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[old["run_id"] != row["run_id"]]  # re-running replaces the old row
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(RUNS_CSV, index=False)


def maybe_save_best(model, tokenizer, alias, text_col, val_f1, meta):
    """Keep only the best run (by VALIDATION macro-F1) per model+text_col on disk."""
    out_dir = MODELS_DIR / f"{alias}_{text_col}_best"
    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev["val_macro_f1"] >= val_f1:
            return False
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta_path.write_text(json.dumps({**meta, "val_macro_f1": val_f1}, indent=2))
    return True


# ----------------------------------------------------------------------------
# One training run
# ----------------------------------------------------------------------------
def run_one(args, alias, display, hf_name, lr, seed, splits, device):
    train, val, test = splits
    set_seed(seed)
    run_id = f"{alias}_lr{lr:g}_seed{seed}_{args.text_col}"
    print("\n" + "=" * 70 + f"\nRUN {run_id}   ({hf_name})\n" + "=" * 70)

    tokenizer = load_tokenizer(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=3,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    ).to(device)

    collate = make_collate(tokenizer)
    ds = {name: TextDataset(df[args.text_col], df["label"], tokenizer, args.max_len)
          for name, df in (("train", train), ("val", val), ("test", test))}
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=g)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size * 2, collate_fn=collate)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size * 2, collate_fn=collate)

    # Class-balanced loss weights from the TRAIN split only.
    counts = np.bincount(train["label"], minlength=3)
    weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights if args.class_weight else None)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps)

    use_amp = device.type == "cuda" and not args.no_fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, best_epoch, bad_epochs, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val_pred, _ = predict(model, val_loader, device, use_amp)
        val_m = compute_metrics(val["label"], val_pred)
        train_loss = running / len(train_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_macro_f1={val_m['macro_f1']:.4f}  "
              f"val_acc={val_m['accuracy']:.4f}  ({time.time() - t0:.0f}s)")

        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_m["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no val improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(LOG_DIR / f"{slugify(run_id)}_history.csv", index=False)

    val_pred, val_probs = predict(model, val_loader, device, use_amp)
    test_pred, test_probs = predict(model, test_loader, device, use_amp)
    val_m, test_m = compute_metrics(val["label"], val_pred), compute_metrics(test["label"], test_pred)
    print_metrics(val_m, f"{run_id} | Validation (best epoch {best_epoch})")
    print_metrics(test_m, f"{run_id} | Test")

    save_predictions(val, val_pred, run_id, "val", val_probs)
    save_predictions(test, test_pred, run_id, "test", test_probs)
    save_confusion_and_report(test["label"], test_pred, run_id)

    row = results_row(
        display, "transformer", val_m, test_m,
        run_id=run_id, alias=alias, hf_name=hf_name, lr=lr, seed=seed,
        text_col=args.text_col, max_len=args.max_len, batch_size=args.batch_size,
        best_epoch=best_epoch, train_seconds=round(time.time() - t0, 1),
    )
    append_run_row(row)

    meta = {"run_id": run_id, "alias": alias, "display": display, "hf_name": hf_name,
            "lr": lr, "seed": seed, "text_col": args.text_col, "max_len": args.max_len,
            "best_epoch": best_epoch}
    if not args.no_save and maybe_save_best(model, tokenizer, alias, args.text_col, val_m["macro_f1"], meta):
        print(f"saved as current best {display} checkpoint (val macro-F1 {val_m['macro_f1']:.4f})")

    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="mbert | muril | indicbert, or any HF model id / local path")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-5])
    p.add_argument("--text_col", default="text_clean", choices=["text_clean", "text"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--no_class_weight", dest="class_weight", action="store_false")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--no_save", action="store_true", help="don't save model checkpoints")
    p.add_argument("--limit", type=int, default=0, help="debug: subsample each split to N rows")
    return p.parse_args()


def main():
    args = parse_args()
    if args.model in MODEL_ALIASES:
        display, hf_name = MODEL_ALIASES[args.model]
        alias = args.model
    else:
        hf_name = args.model
        alias = slugify(Path(hf_name).name)
        display = Path(hf_name).name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train, val, test = load_splits()
    if args.limit:
        train, val, test = (df.sample(frac=1, random_state=0).groupby("label").head(args.limit // 3)
                            for df in (train, val, test))
    splits = (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))

    rows = [run_one(args, alias, display, hf_name, lr, seed, splits, device)
            for lr in args.lrs for seed in args.seeds]

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(pd.DataFrame(rows)[["run_id", "best_epoch", "validation_macro_f1", "test_macro_f1", "test_accuracy"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
