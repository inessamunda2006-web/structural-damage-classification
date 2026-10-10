"""Training: evaluasi holdout + (opsional) CV, lalu fit di seluruh train dan simpan model.

Contoh:
    python -m scripts.train            # holdout 80/20 + fit final
    python -m scripts.train --cv       # tambah Stratified 5-fold CV
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             cohen_kappa_score, f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from scripts.config import CLASSES, FIG_DIR, MODEL_DIR, SEED, TARGET
from scripts.models import build_pipeline
from scripts.preprocess import encode_target, load_data


def load_params():
    """Membaca hyperparameter hasil tuning dari models/best_params.json (jika ada)."""
    path = MODEL_DIR / "best_params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def scores(y_true, y_pred):
    """Menghitung accuracy (= micro-F1), macro-F1, weighted-F1, dan Quadratic Weighted Kappa."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def plot_confusion(y_true, y_pred, path):
    """Menyimpan confusion matrix (dinormalisasi per baris) ke `path`."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASSES, normalize="true",
        cmap="Blues", ax=ax, values_format=".2f", xticks_rotation=45)
    ax.set_title("Confusion Matrix (holdout, normalisasi per baris)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(pipe, path, top=20):
    """Menyimpan bar chart feature importance (gain) dari LightGBM di dalam pipeline."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe[-1].booster_.feature_importance("gain"), index=names)
    imp = (imp / imp.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="#2b6cb0")
    ax.set_xlabel("Proporsi total gain")
    ax.set_title(f"Top {top} Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(run_cv=False):
    """Menjalankan seluruh alur training dan menyimpan model ke models/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    train, _ = load_data()
    X, y = train.drop(columns=[TARGET]), encode_target(train[TARGET])
    params = load_params()
    print("Hyperparameter:", params)

    # 1) Holdout 80/20 (stratified)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = build_pipeline(**params).fit(Xa, ya)
    pred = pipe.predict(Xb)
    print("Holdout:", {k: round(v, 4) for k, v in scores(yb, pred).items()})
    print(classification_report(yb, pred, target_names=CLASSES, digits=3))
    plot_confusion(yb, pred, FIG_DIR / "confusion_matrix.png")
    plot_importance(pipe, FIG_DIR / "feature_importance.png")

    # 2) Cross-validation (opsional, lebih lambat)
    if run_cv:
        rows = []
        for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
            m = build_pipeline(**params).fit(X.iloc[a], y.iloc[a])
            rows.append(scores(y.iloc[b], m.predict(X.iloc[b])))
            print(f"fold {i + 1}:", {k: round(v, 4) for k, v in rows[-1].items()}, flush=True)
        cv = pd.DataFrame(rows)
        print("CV mean:\n", cv.mean().round(4), "\nCV std:\n", cv.std().round(4))
        cv.to_csv(MODEL_DIR / "cv_scores.csv", index=False)

    # 3) Fit final pada seluruh data train
    final = build_pipeline(**params).fit(X, y)
    joblib.dump(final, MODEL_DIR / "pipeline_model_lgbm.pkl", compress=3)
    print("Model tersimpan di", MODEL_DIR / "pipeline_model_lgbm.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="jalankan Stratified 5-fold CV")
    main(run_cv=ap.parse_args().cv)
