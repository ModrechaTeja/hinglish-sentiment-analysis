"""
Phase 6 — Unified evaluation & model comparison.

Reads what Phases 4/4.5/5 wrote and produces the tables + figures for the
report. No model is retrained here.

Protocol (all selection on VALIDATION, test is report-only):
  * Transformers: per model, the LR with the best seed-42 validation
    macro-F1 is the final config; we report mean ± std over its seeds.
    The "selected run" (best val macro-F1 among those seeds) is the saved
    checkpoint used for significance tests, confusion matrices and the demo.
  * Classical: the best classical model is chosen by validation macro-F1.
  * Significance: exact McNemar test (are the error sets different?) and a
    paired bootstrap of the macro-F1 difference (10,000 resamples).

Outputs:
  outputs/results/final_comparison.csv
  outputs/results/significance_tests.csv
  outputs/results/text_ablation.csv          (if raw-text runs exist)
  outputs/results/phase6_summary.md          (copy-paste into the report)
  outputs/figures/macro_f1_comparison.png
  outputs/figures/per_class_f1.png
  outputs/figures/training_curves.png
  outputs/figures/confusion_matrices_grid.png

Usage:
  python src/evaluate.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from eval_utils import (
    RESULTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, LABELS, LABEL_NAMES, slugify,
)

N_BOOT = 10_000
RNG = np.random.default_rng(0)

# Base paper, Joshi et al. (COLING 2016), Table 5 — same dataset, 64/16/20 split.
PAPER_RESULTS = [
    ("NBSVM (Unigram)", 0.5915, 0.5335),
    ("MNB (Unigram)", 0.6675, 0.6143),
    ("MNB (Tf-Idf)", 0.6353, 0.4783),
    ("SVM (Unigram)", 0.5760, 0.5232),
    ("Lexicon Lookup (Sharma et al.)", 0.5115, 0.2520),
    ("Char-LSTM", 0.5980, 0.5110),
    ("Subword-LSTM", 0.6970, 0.6580),
]
PAPER_BEST = 0.658

# Categorical slots (fixed order), validated for CVD separation.
# Contrast vs white is < 3:1 for slots 3-4, so every bar carries a value label.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES_COLORS = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
TRANSFORMER_ORDER = ["mBERT", "MuRIL", "IndicBERT"]
# Colour follows the model, never its rank: same colour in every figure.
MODEL_COLOR = {"mBERT": C_ORANGE, "MuRIL": C_AQUA, "IndicBERT": C_YELLOW}


def color_for(name: str, fallback_idx: int = 0) -> str:
    return MODEL_COLOR.get(name, SERIES_COLORS[fallback_idx % len(SERIES_COLORS)])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
})


# ----------------------------------------------------------------------------
# Loading & selection
# ----------------------------------------------------------------------------
def load_preds(run_name: str, split: str = "test") -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / f"{slugify(run_name)}_{split}.csv")


def select_transformer_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Final config per model = LR with best seed-42 val macro-F1 (text_clean)."""
    runs = runs[runs.text_col == "text_clean"]
    finals = []
    for alias, g in runs.groupby("alias"):
        s42 = g[g.seed == 42]
        if len(s42):
            best_lr = s42.loc[s42.validation_macro_f1.idxmax(), "lr"]
        else:
            best_lr = g.groupby("lr").validation_macro_f1.mean().idxmax()
        finals.append(g[g.lr == best_lr])
    return pd.concat(finals) if finals else runs.iloc[0:0]


def ordered_models(names):
    known = [m for m in TRANSFORMER_ORDER if m in names]
    return known + sorted(set(names) - set(known))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def bootstrap_f1_ci(y, p, n=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    scores = np.array([f1_score(y[i], p[i], average="macro", zero_division=0) for i in idx])
    return np.percentile(scores, [2.5, 97.5])


def paired_bootstrap(y, pa, pb, n=N_BOOT):
    """Macro-F1(b) - macro-F1(a): mean diff, 95% CI, one-sided p (b <= a)."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    diffs = np.array([
        f1_score(y[i], pb[i], average="macro", zero_division=0)
        - f1_score(y[i], pa[i], average="macro", zero_division=0) for i in idx
    ])
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs <= 0).mean()


def mcnemar_exact(y, pa, pb):
    ca, cb = (np.asarray(pa) == np.asarray(y)), (np.asarray(pb) == np.asarray(y))
    b_only, a_only = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = binomtest(b_only, b_only + a_only, 0.5).pvalue if b_only + a_only else 1.0
    return a_only, b_only, p


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_macro_f1(table: pd.DataFrame):
    t = table.sort_values("test_macro_f1")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    colors = [C_ORANGE if f == "transformer" else C_BLUE for f in t.family]
    ax.barh(t.model, t.test_macro_f1, color=colors, height=0.6, edgecolor="white", linewidth=2)
    std = t.test_macro_f1_std.fillna(0).values
    has = std > 0  # error bars only for multi-seed (transformer) rows
    ax.errorbar(t.test_macro_f1.values[has], np.arange(len(t))[has], xerr=std[has], fmt="none",
                ecolor=INK, elinewidth=1, capsize=3)
    for y, (v, s) in enumerate(zip(t.test_macro_f1, t.test_macro_f1_std.fillna(0))):
        label = f"{v:.3f}" + (f" ± {s:.3f}" if s > 0 else "")
        ax.text(v + s + 0.006, y, label, va="center", fontsize=9, color=INK)
    ax.axvline(PAPER_BEST, color=INK_MUTED, ls="--", lw=1)
    lo = max(0.0, t.test_macro_f1.min() - 0.08)
    ax.set_xlim(lo, min(1.0, t.test_macro_f1.max() + 0.09))
    ax.set_xlabel("Test macro-F1  (transformers: mean ± std over seeds)")
    ax.set_title("Test macro-F1 by model")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=C_BLUE, label="Classical (TF-IDF)"),
                       Patch(color=C_ORANGE, label="Transformer"),
                       Line2D([], [], color=INK_MUTED, ls="--", lw=1,
                              label=f"Base paper Subword-LSTM ({PAPER_BEST})")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "macro_f1_comparison.png", dpi=200)
    plt.close(fig)


def fig_per_class(rows):
    """rows: list of (name, [f1_neg, f1_neu, f1_pos])."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    k = len(rows)
    width = 0.8 / k
    x = np.arange(3)
    for i, (name, vals) in enumerate(rows):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width, color=C_BLUE if i == 0 else color_for(name, i), label=name,
               edgecolor="white", linewidth=2)
        for xp, v in zip(pos, vals):
            ax.text(xp, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, LABEL_NAMES)
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1: best classical vs transformers")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=k, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1.png", dpi=200)
    plt.close(fig)


def fig_training_curves(selected: pd.DataFrame):
    logs = RESULTS_DIR / "logs"
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, name in enumerate(ordered_models(selected.model.unique())):
        run_id = selected[selected.model == name].iloc[0].run_id
        path = logs / f"{slugify(run_id)}_history.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path)
        ax.plot(h.epoch, h.val_macro_f1, color=color_for(name, i + 1), lw=2,
                marker="o", markersize=5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 per epoch (selected runs)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=200)
    plt.close(fig)


def fig_confusion_grid(items, y_true):
    """items: list of (title, y_pred)."""
    fig, axes = plt.subplots(1, len(items), figsize=(3.3 * len(items), 3.9))
    axes = np.atleast_1d(axes)
    short = ["Neg", "Neu", "Pos"]
    for ax, (title, pred) in zip(axes, items):
        cm = confusion_matrix(y_true, pred, labels=LABELS, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set_xlabel("Predicted")
        ax.set_title(title, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True")
    fig.suptitle("Row-normalized test confusion matrices (diagonal = recall)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_grid.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def fmt(m, s=None):
    return f"{m:.4f}" if s is None or pd.isna(s) else f"{m:.4f} ± {s:.4f}"


def main():
    base = pd.read_csv(RESULTS_DIR / "baseline_results.csv")
    runs_path = RESULTS_DIR / "transformer_runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    if runs.empty:
        print("[warn] no transformer runs found — comparing classical models only.")

    # ---- best classical, chosen on validation ----
    best_cls = base.loc[base.validation_macro_f1.idxmax()]
    cls_test = load_preds(best_cls.model)
    y = cls_test.label.values
    print(f"Best classical (by val macro-F1): {best_cls.model} "
          f"(val {best_cls.validation_macro_f1:.4f}, test {best_cls.test_macro_f1:.4f})")

    # ---- comparison table ----
    metric_cols = ["validation_macro_f1", "test_accuracy", "test_macro_f1",
                   "test_f1_negative", "test_f1_neutral", "test_f1_positive"]
    rows = []
    for _, r in base.iterrows():
        p = load_preds(r.model)
        lo, hi = bootstrap_f1_ci(p.label, p.pred)
        rows.append({"model": r.model, "family": r.family.split(" ")[0], "config": r.family,
                     "n_seeds": 1, **{c: r[c] for c in metric_cols},
                     **{f"{c}_std": np.nan for c in metric_cols},
                     "test_macro_f1_ci_low": lo, "test_macro_f1_ci_high": hi,
                     "selected_run": r.model})

    final_runs = select_transformer_runs(runs) if not runs.empty else pd.DataFrame()
    selected = []
    for name in ordered_models(final_runs.model.unique()) if len(final_runs) else []:
        g = final_runs[final_runs.model == name]
        sel = g.loc[g.validation_macro_f1.idxmax()]
        selected.append(sel)
        p = load_preds(sel.run_id)
        lo, hi = bootstrap_f1_ci(p.label, p.pred)
        rows.append({"model": name, "family": "transformer",
                     "config": f"lr={sel.lr:g}, {sel.text_col}", "n_seeds": len(g),
                     **{c: g[c].mean() for c in metric_cols},
                     **{f"{c}_std": g[c].std() if len(g) > 1 else np.nan for c in metric_cols},
                     "test_macro_f1_ci_low": lo, "test_macro_f1_ci_high": hi,
                     "selected_run": sel.run_id})
    selected = pd.DataFrame(selected)

    table = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    table.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)

    # ---- significance: best classical vs each transformer's selected run ----
    sig = []
    for _, sel in selected.iterrows():
        tp = load_preds(sel.run_id)
        assert (tp.id.values == cls_test.id.values).all(), "prediction files are not aligned"
        a_only, b_only, p_mc = mcnemar_exact(y, cls_test.pred, tp.pred)
        d, dlo, dhi, p_bs = paired_bootstrap(y, cls_test.pred, tp.pred)
        # every seed of the final config, so one lucky seed can't carry the claim
        seeds = final_runs[final_runs.model == sel.model]
        seed_p = [mcnemar_exact(y, cls_test.pred, load_preds(r.run_id).pred)[2]
                  for _, r in seeds.iterrows()]
        sig.append({
            "comparison": f"{sel.model} vs {best_cls.model}",
            "transformer_run": sel.run_id,
            "only_classical_correct": a_only, "only_transformer_correct": b_only,
            "mcnemar_p": p_mc,
            "macro_f1_diff": d, "diff_ci_low": dlo, "diff_ci_high": dhi,
            "bootstrap_p_transformer_not_better": p_bs,
            "seeds_significant_at_0.05": f"{sum(p < 0.05 for p in seed_p)}/{len(seed_p)}",
        })
    sig = pd.DataFrame(sig)
    sig.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)

    # ---- optional ablation: raw text vs cleaned text ----
    abl = pd.DataFrame()
    if not runs.empty and (runs.text_col == "text").any():
        raw = runs[runs.text_col == "text"]
        out = []
        for (alias, lr), g in raw.groupby(["alias", "lr"]):
            clean = runs[(runs.alias == alias) & (runs.lr == lr) & (runs.text_col == "text_clean")
                         & runs.seed.isin(g.seed)]
            for label, df in (("text_clean", clean), ("text (raw)", g)):
                out.append({"model": g.model.iloc[0], "lr": lr, "input": label, "n_seeds": len(df),
                            "val_macro_f1": fmt(df.validation_macro_f1.mean(), df.validation_macro_f1.std()),
                            "test_macro_f1": fmt(df.test_macro_f1.mean(), df.test_macro_f1.std()),
                            "test_f1_negative": fmt(df.test_f1_negative.mean(), df.test_f1_negative.std())})
        abl = pd.DataFrame(out)
        abl.to_csv(RESULTS_DIR / "text_ablation.csv", index=False)

    # ---- figures ----
    fig_macro_f1(table)
    per_class = [(f"{best_cls.model} (classical)",
                  [best_cls.test_f1_negative, best_cls.test_f1_neutral, best_cls.test_f1_positive])]
    for _, r in table[table.family == "transformer"].iterrows():
        per_class.append((r.model, [r.test_f1_negative, r.test_f1_neutral, r.test_f1_positive]))
    per_class = [per_class[0]] + sorted(per_class[1:], key=lambda x: TRANSFORMER_ORDER.index(x[0])
                                        if x[0] in TRANSFORMER_ORDER else 99)
    fig_per_class(per_class[:4])
    if len(selected):
        fig_training_curves(selected)
    cm_items = [(best_cls.model, cls_test.pred.values)]
    cm_items += [(f"{s.model} (seed {s.seed})", load_preds(s.run_id).pred.values)
                 for _, s in selected.iterrows()]
    fig_confusion_grid(cm_items, y)

    # ---- markdown summary for the report ----
    md = ["# Phase 6 — Model comparison\n",
          "Test set: %d examples. Model selection on validation macro-F1 only. "
          "Transformers: mean ± std over seeds of the final config; 95%% CI is a "
          "bootstrap over test examples for the selected run (classical: the single run).\n" % len(y),
          "| Model | Family | Config | Seeds | Val macro-F1 | Test acc | Test macro-F1 | 95% CI | F1 Neg | F1 Neu | F1 Pos |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r.model} | {r.family} | {r.config} | {r.n_seeds} | "
                  f"{fmt(r.validation_macro_f1, r.validation_macro_f1_std)} | "
                  f"{fmt(r.test_accuracy, r.test_accuracy_std)} | "
                  f"**{fmt(r.test_macro_f1, r.test_macro_f1_std)}** | "
                  f"[{r.test_macro_f1_ci_low:.3f}, {r.test_macro_f1_ci_high:.3f}] | "
                  f"{r.test_f1_negative:.3f} | {r.test_f1_neutral:.3f} | {r.test_f1_positive:.3f} |")
    md += ["\n## Base paper (Joshi et al., 2016, Table 5) — same dataset, 64/16/20 split\n",
           "| Method | Accuracy | F1 |", "|---|---|---|"]
    md += [f"| {n} | {a:.4f} | {f:.4f} |" for n, a, f in PAPER_RESULTS]
    if len(sig):
        md += [f"\n## Significance vs best classical model ({best_cls.model})\n",
               "| Comparison | Only classical correct | Only transformer correct | McNemar p | "
               "Δ macro-F1 [95% CI] | Bootstrap p | Seeds sig. (p<.05) |",
               "|---|---|---|---|---|---|---|"]
        for _, s in sig.iterrows():
            md.append(f"| {s.comparison} | {s.only_classical_correct} | {s.only_transformer_correct} | "
                      f"{s.mcnemar_p:.4f} | {s.macro_f1_diff:+.4f} [{s.diff_ci_low:+.4f}, {s.diff_ci_high:+.4f}] | "
                      f"{s.bootstrap_p_transformer_not_better:.4f} | {s['seeds_significant_at_0.05']} |")
    if len(abl):
        md += ["\n## Ablation — cleaned vs raw input text\n",
               "| " + " | ".join(abl.columns) + " |", "|" + "---|" * len(abl.columns)]
        md += ["| " + " | ".join(str(v) for v in r) + " |" for r in abl.itertuples(index=False)]
    (RESULTS_DIR / "phase6_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- console ----
    show = table[["model", "family", "n_seeds", "test_accuracy", "test_macro_f1",
                  "test_macro_f1_std", "test_f1_negative"]]
    print("\n" + show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(sig):
        print("\n" + sig[["comparison", "mcnemar_p", "macro_f1_diff", "diff_ci_low", "diff_ci_high",
                          "seeds_significant_at_0.05"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(abl):
        print("\n" + abl.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'phase6_summary.md'} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
