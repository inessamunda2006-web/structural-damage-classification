"""Prediksi test.csv dan membuat submissions/submission.csv sesuai sample_submission.csv.

Contoh:
    python -m scripts.predict            # label "Grade k"
    python -m scripts.predict --numeric  # label 1-5
"""
import argparse

import joblib
import pandas as pd

from scripts.config import DATA_DIR, ID_COL, MODEL_DIR, SUBMISSION_DIR, TARGET
from scripts.preprocess import decode_target, load_data


def main(model_name="pipeline_model_lgbm.pkl", numeric=False):
    """Memuat model tersimpan, memprediksi test, dan menulis submission.csv (index=False).

    Args:
        model_name: nama file model di folder models/.
        numeric: jika True, label ditulis sebagai angka 1-5 (bukan 'Grade k').
    """
    _, test = load_data()
    pipe = joblib.load(MODEL_DIR / model_name)
    pred = decode_target(pipe.predict(test))

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
    ap.add_argument("--numeric", action="store_true", help="tulis label sebagai 1-5")
    main(numeric=ap.parse_args().numeric)
