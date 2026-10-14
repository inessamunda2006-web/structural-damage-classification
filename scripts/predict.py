"""Prediksi test.csv dan membuat submissions/submission.csv sesuai sample_submission.csv.

Contoh:
    python -m scripts.predict              # label angka 1-5, argmax biasa (default)
    python -m scripts.predict --weighted   # pakai bobot kelas (untuk macro-F1)
    python -m scripts.predict --grade-text # label "Grade k"
"""
import argparse

import joblib
import pandas as pd

from scripts.config import DATA_DIR, ID_COL, MODEL_DIR, SUBMISSION_DIR, TARGET
from scripts.postprocess import apply_class_weights, load_weights
from scripts.preprocess import decode_target, load_data


def main(model_name="pipeline_model_lgbm.pkl", numeric=True, use_weights=False):
    """Memuat model tersimpan, memprediksi test, dan menulis submission.csv (index=False).

    Args:
        model_name: nama file model di folder models/.
        numeric: jika True (default), label ditulis sebagai angka 1-5 (format yang
            diterima evaluator Kaggle kompetisi ini); jika False, ditulis 'Grade k'.
        use_weights: jika True dan models/class_weights.json ada, pakai bobot kelas
            (dioptimalkan untuk macro-F1); jika False (default), pakai argmax biasa.
    """
    _, test = load_data()
    pipe = joblib.load(MODEL_DIR / model_name)
    proba = pipe.predict_proba(test)
    weights = load_weights() if use_weights else None
    labels = proba.argmax(1) if weights is None else apply_class_weights(proba, weights)
    print('Bobot kelas:', 'tidak dipakai' if weights is None else weights.round(3))
    pred = decode_target(labels)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")  # urutan = sample_submission
    if numeric:
        sub[TARGET] = sub[TARGET].str.replace("Grade ", "", regex=False).astype(int)
    assert len(sub) == len(sample) and sub[TARGET].notna().all()

    SUBMISSION_DIR.mkdir(exist_ok=True)
    out = SUBMISSION_DIR / "submission.csv"
    sub.to_csv(out, index=False, lineterminator="\r\n")  # sama dengan sample_submission
    print("Tersimpan:", out, sub.shape)
    print(sub[TARGET].value_counts(normalize=True).sort_index().round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade-text", action="store_true", help='tulis label "Grade k"')
    ap.add_argument("--weighted", action="store_true", help="pakai bobot kelas (macro-F1)")
    a = ap.parse_args()
    main(numeric=not a.grade_text, use_weights=a.weighted)
