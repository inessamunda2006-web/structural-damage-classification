"""Pembuat pipeline lengkap (feature engineering + preprocessing + model)."""
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from scripts.config import SEED
from scripts.feature_engineering import BuildingFeatureAdder
from scripts.preprocess import build_preprocessor


def build_pipeline(**clf_params):
    """Membuat Pipeline: BuildingFeatureAdder -> ColumnTransformer -> LightGBM.

    Args:
        **clf_params: hyperparameter LGBMClassifier yang menimpa nilai default.

    Returns:
        sklearn.pipeline.Pipeline yang siap di-fit pada DataFrame mentah.
    """
    params = dict(n_estimators=400, learning_rate=0.08, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  n_jobs=2, verbose=-1)
    params.update(clf_params)
    return Pipeline([
        ("fe", BuildingFeatureAdder(use_serial=False)),
        ("pre", build_preprocessor(use_serial=False)),
        ("clf", LGBMClassifier(**params)),
    ])
