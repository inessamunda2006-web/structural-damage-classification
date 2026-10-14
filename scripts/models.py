"""Pembuat pipeline lengkap (feature engineering + preprocessing + model)."""
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from scripts.config import SEED
from scripts.feature_engineering import BuildingFeatureAdder
from scripts.neighbor_features import NeighborTargetFeatures
from scripts.preprocess import build_preprocessor


def build_pipeline(use_neighbors=True, **clf_params):
    """Membuat Pipeline: NeighborTargetFeatures -> BuildingFeatureAdder -> ColumnTransformer -> LightGBM.

    Args:
        use_neighbors: jika True, tambahkan fitur tetangga building_id (default).
        **clf_params: hyperparameter LGBMClassifier yang menimpa nilai default.

    Returns:
        sklearn.pipeline.Pipeline yang siap di-fit pada DataFrame mentah.
    """
    params = dict(n_estimators=400, learning_rate=0.08, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  n_jobs=-1, verbose=-1)
    params.update(clf_params)
    steps = [("nbr", NeighborTargetFeatures())] if use_neighbors else []
    steps += [
        ("fe", BuildingFeatureAdder(use_serial=False)),
        ("pre", build_preprocessor(use_serial=False, use_neighbors=use_neighbors)),
        ("clf", LGBMClassifier(**params)),
    ]
    return Pipeline(steps)
