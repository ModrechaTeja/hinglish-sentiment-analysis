"""
Phase 4 (+4.5) — Classical ML baselines for Hinglish sentiment.

Two feature families:
  * Word TF-IDF (1-2 grams)  — the original Phase 4 setup.
  * Char TF-IDF (2-5 grams, word-boundary aware) — added in Phase 4.5.
    This is the classical counterpart of the base paper's sub-word idea
    (Joshi et al., COLING 2016): character n-grams are robust to the
    romanized spelling variation that dominates this data
    (bahut / bohot / bhot / bahot all share "bh", "ho", ...), so it is the
    strongest "fair" classical baseline for the transformers to beat.

All models use class-balanced training (50/35/15 neutral/positive/negative).
Every model writes per-example predictions to outputs/predictions/ so
Phase 6 (evaluation) and Phase 7 (error analysis) can reuse them.

Usage:
    python src/baseline_models.py
"""

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from eval_utils import (
    load_splits, compute_metrics, print_metrics, save_predictions,
    save_confusion_and_report, results_row, slugify,
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR,
)

TEXT_COL = "text_clean"


def build_vectorizers():
    return {
        "word": TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
        ),
        "char": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2,
            sublinear_tf=True, lowercase=True,
        ),
    }


def build_models():
    """(display name, feature family, estimator)."""
    return [
        # --- Word TF-IDF (original Phase 4) ---
        ("Logistic Regression", "word",
         LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ("Linear SVM", "word",
         LinearSVC(random_state=42, class_weight="balanced")),
        ("Naive Bayes", "word", MultinomialNB()),
        ("Random Forest", "word",
         RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                class_weight="balanced", min_samples_leaf=2)),
        # --- Char TF-IDF (Phase 4.5); C chosen on the VALIDATION set ---
        ("Char Logistic Regression", "char",
         LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced", C=4.0)),
        ("Char Linear SVM", "char",
         LinearSVC(random_state=42, class_weight="balanced", C=0.25)),
    ]


def main():
    print("Loading datasets...")
    train, val, test = load_splits()
    print(f"Train / Val / Test: {len(train)} / {len(val)} / {len(test)}")
    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    features = {}
    for fam, vec in build_vectorizers().items():
        Xtr = vec.fit_transform(train[TEXT_COL])
        features[fam] = (Xtr, vec.transform(val[TEXT_COL]), vec.transform(test[TEXT_COL]))
        joblib.dump(vec, MODELS_DIR / f"tfidf_{fam}_vectorizer.joblib")
        print(f"{fam:>4} TF-IDF vocabulary: {len(vec.vocabulary_):,}")
    # Keep the old filename working for anything that loads it.
    joblib.dump(joblib.load(MODELS_DIR / "tfidf_word_vectorizer.joblib"),
                MODELS_DIR / "tfidf_vectorizer.joblib")

    nb_weight = compute_sample_weight("balanced", y_train)
    results = []

    for name, fam, model in build_models():
        print("\n" + "=" * 60 + f"\nTraining: {name}  [{fam} TF-IDF]\n" + "=" * 60)
        X_train, X_val, X_test = features[fam]

        if isinstance(model, MultinomialNB):
            model.fit(X_train, y_train, sample_weight=nb_weight)
        else:
            model.fit(X_train, y_train)

        val_pred, test_pred = model.predict(X_val), model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")
        val_m, test_m = compute_metrics(y_val, val_pred), compute_metrics(y_test, test_pred)
        print_metrics(val_m, "Validation")
        print_metrics(test_m, "Test")

        save_predictions(val, val_pred, name, "val",
                         model.predict_proba(X_val) if has_proba else None)
        save_predictions(test, test_pred, name, "test",
                         model.predict_proba(X_test) if has_proba else None)
        save_confusion_and_report(y_test, test_pred, name)
        joblib.dump(model, MODELS_DIR / f"{slugify(name)}.joblib")

        results.append(results_row(name, f"classical ({fam} tf-idf)", val_m, test_m,
                                   seed=42, text_col=TEXT_COL))

    df = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)
    path = RESULTS_DIR / "baseline_results.csv"
    df.to_csv(path, index=False)

    print("\n" + "=" * 60 + "\nFINAL BASELINE COMPARISON (test)\n" + "=" * 60)
    print(df[["model", "test_accuracy", "test_macro_f1",
              "test_f1_negative", "test_f1_neutral", "test_f1_positive"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults -> {path}\nFigures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
