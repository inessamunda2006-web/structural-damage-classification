"""Tuning hyperparameter LightGBM dengan RandomizedSearchCV (Stratified 3-fold)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from scripts.config import MODEL_DIR, SEED, TARGET
from scripts.models import build_pipeline
from scripts.preprocess import encode_target, load_data

SPACE = {
    "clf__num_leaves": [31, 63, 127, 255],
    "clf__learning_rate": [0.03, 0.05, 0.08],
    "clf__n_estimators": [300, 500, 800],
    "clf__min_child_samples": [20, 50, 100, 200],
    "clf__reg_lambda": [0, 1, 5, 10, 30],
    "clf__colsample_bytree": [0.5, 0.7, 0.9],
}


def main(frac=0.25, n_iter=10):
    """Menjalankan random search pada subset `frac` data train dan menyimpan parameter terbaik."""
    train, _ = load_data()
    sub = train.sample(frac=frac, random_state=SEED)
    X, y = sub.drop(columns=[TARGET]), encode_target(sub[TARGET])
    search = RandomizedSearchCV(
        build_pipeline(), SPACE, n_iter=n_iter, scoring="accuracy",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED,
        n_jobs=1, verbose=2, refit=False)
    search.fit(X, y)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print("BEST", search.best_score_, best)
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
