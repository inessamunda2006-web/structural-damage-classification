"""Loader data dan ColumnTransformer untuk preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from scripts.neighbor_features import NBR_COLS
from scripts.config import CAT_COLS, CLASSES, DATA_DIR, GEO_COLS, NUM_COLS, SEED, TARGET

DERIVED_NUM = ["age_is_999", "height_per_floor", "volume_proxy", "area_per_floor",
               "slenderness", "log_area"]


def load_data():
    """Membaca train.csv dan test.csv dari folder data/.

    Returns:
        tuple: (train_df, test_df) berupa DataFrame.
    """
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def encode_target(y):
    """Mengubah label teks 'Grade 1'..'Grade 5' menjadi integer 0..4."""
    return y.map({c: i for i, c in enumerate(CLASSES)}).astype(int)


def decode_target(pred):
    """Mengubah integer 0..4 kembali menjadi label teks 'Grade k'."""
    return pd.Series(pred).map(dict(enumerate(CLASSES))).values


def build_preprocessor(use_serial=True, use_neighbors=True):
    """Membuat ColumnTransformer.

    - Numerik (asli + turunan + fitur tetangga bila `use_neighbors`): impute median.
    - Kategorik: OrdinalEncoder (unknown -> -1), cocok untuk model berbasis pohon.
    - Geografi (district/vdcmun/ward): angka asli + TargetEncoder multikelas
      (dengan cross-fitting internal sehingga tidak bocor ke data training).
    """
    num = (NUM_COLS + DERIVED_NUM + (NBR_COLS if use_neighbors else [])
           + (["serial_no"] if use_serial else []))
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=False), num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
            ("geo_raw", "passthrough", GEO_COLS),
            ("geo_te", TargetEncoder(target_type="multiclass", smooth="auto",
                                     cv=5), GEO_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
