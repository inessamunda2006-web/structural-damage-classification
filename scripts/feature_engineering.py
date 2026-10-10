"""Transformer feature engineering yang kompatibel dengan sklearn Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BuildingFeatureAdder(BaseEstimator, TransformerMixin):
    """Menambah fitur turunan dari atribut bangunan.

    Fitur yang dibuat:
        age_is_999      : flag usia 999 (kode placeholder), usia diganti NaN
        height_per_floor: tinggi per lantai
        volume_proxy    : plinth_area * height (proxy volume bangunan)
        area_per_floor  : plinth_area / jumlah lantai
        slenderness     : tinggi / sqrt(plinth_area), makin besar makin ramping
        log_area        : log1p(plinth_area)
        serial_no       : 4 digit terakhir building_id (urutan survei dalam ward)
    """

    def __init__(self, use_serial=True):
        self.use_serial = use_serial

    def fit(self, X, y=None):
        """Tidak ada parameter yang dipelajari; mengembalikan self."""
        return self

    def transform(self, X):
        """Mengembalikan salinan DataFrame dengan fitur turunan tambahan."""
        X = X.copy()
        is999 = X["age_building"] == 999
        X["age_is_999"] = is999.astype(int)
        X.loc[is999, "age_building"] = np.nan
        floors = X["count_floors_pre_eq"].clip(lower=1)
        area = X["plinth_area_sq_ft"].clip(lower=1)
        X["height_per_floor"] = X["height_ft_pre_eq"] / floors
        X["volume_proxy"] = area * X["height_ft_pre_eq"]
        X["area_per_floor"] = area / floors
        X["slenderness"] = X["height_ft_pre_eq"] / np.sqrt(area)
        X["log_area"] = np.log1p(area)
        if self.use_serial and "building_id" in X.columns:
            X["serial_no"] = X["building_id"] % 10000
        return X.drop(columns=["building_id"], errors="ignore")
