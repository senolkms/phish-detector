# train_large.py
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Kendi URL feature'ların
from utils.url_feats import URLFeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/emails_large_son.csv")
    parser.add_argument("--model_out", default="models/phish_svc_tfidf_char_word_url_son.joblib")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--word_max", type=int, default=200_000, help="TF-IDF kelime özellik üst sınırı")
    parser.add_argument("--char_max", type=int, default=200_000, help="TF-IDF karakter özellik üst sınırı")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    DATA_FP = Path(args.data)
    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FP = Path(args.model_out); MODEL_FP.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_FP.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATA_FP}. Önce scripts/prepare_data.py çalıştırın.")

    df = pd.read_csv(DATA_FP)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV 'text' ve 'label' sütunlarını içermeli.")

    texts = df["text"].astype(str).values
    y = df["label"].astype(int).values
    print(f"[INFO] Dataset shape: {df.shape} | positive={(y==1).sum()} | negative={(y==0).sum()}")

    # ------- Özellikler: Word + Char + URL -------
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=args.word_max,
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=args.char_max,
        dtype=np.float32,
    )
    url_block = Pipeline([
        ("url", URLFeats()),
        ("scale", MaxAbsScaler()),
    ])

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
        ("url", url_block),
    ])

    # ------- Sınıflandırıcı -------
    base_svc = LinearSVC(class_weight="balanced", random_state=args.random_state)
    clf = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    pipe = Pipeline([
        ("features", features),
        ("clf", clf),
    ], memory="__cache__")  # tekrar eden dönüşümleri cache'ler

    # ------- CV -------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_acc = cross_val_score(pipe, texts, y, cv=skf, scoring="accuracy")
    cv_f1  = cross_val_score(pipe, texts, y, cv=skf, scoring="f1_macro")
    print(f"[CV] Accuracy  mean={cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"[CV] F1-macro mean={cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ------- Train/Test -------
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.20, stratify=y, random_state=args.random_state
    )
    pipe.fit(X_tr, y_tr)

    # ------- Test Sonuçları -------
    y_pred = pipe.predict(X_te)
    print("\n[TEST] Classification report:\n")
    print(classification_report(y_te, y_pred, digits=4))

    proba = pipe.predict_proba(X_te)[:, 1]

    # ------- Grafikler -------
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax_cm, colorbar=False)
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout(); fig_cm.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)

    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax_roc)
    ax_roc.set_title("ROC Curve")
    fig_roc.tight_layout(); fig_roc.savefig(OUT_DIR / "roc_curve.png", dpi=160)

    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax_pr)
    ax_pr.set_title("Precision-Recall Curve")
    fig_pr.tight_layout(); fig_pr.savefig(OUT_DIR / "pr_curve.png", dpi=160)
    plt.close("all")

    # ------- Kaydet -------
    joblib.dump(pipe, MODEL_FP)
    print(f"\n[SAVED] Model: {MODEL_FP}")

    metrics = {
        "cv": {
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std": float(cv_f1.std()),
            "n_splits": int(skf.n_splits),
        },
        "test": {
            "n_test": int(len(y_te)),
            "n_pos": int((y_te == 1).sum()),
            "n_neg": int((y_te == 0).sum()),
        },
        "artifacts": {
            "confusion_matrix_png": str(OUT_DIR / "confusion_matrix.png"),
            "roc_curve_png": str(OUT_DIR / "roc_curve.png"),
            "pr_curve_png": str(OUT_DIR / "pr_curve.png"),
            "model_joblib": str(MODEL_FP),
        },
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Metrics & plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
