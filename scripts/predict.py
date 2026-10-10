"""Prediksi test.csv dan membuat submissions/submission.csv sesuai sample_submission.csv.

Contoh:
    python -m scripts.predict
"""
import joblib
import pandas as pd

from scripts.config import DATA_DIR, ID_COL, MODEL_DIR, SUBMISSION_DIR, TARGET
from scripts.preprocess import decode_target, load_data


def main(model_name="pipeline_model_lgbm.pkl"):
    """Memuat model tersimpan, memprediksi test, dan menulis submission.csv (index=False)."""
    _, test = load_data()
    pipe = joblib.load(MODEL_DIR / model_name)
    pred = decode_target(pipe.predict(test))

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")  # urutan = sample_submission
    assert len(sub) == len(sample) and sub[TARGET].notna().all()

    SUBMISSION_DIR.mkdir(exist_ok=True)
    out = SUBMISSION_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print("Tersimpan:", out, sub.shape)
    print(sub[TARGET].value_counts(normalize=True).sort_index().round(4))


if __name__ == "__main__":
    main()
