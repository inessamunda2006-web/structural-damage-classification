"""Fitur tetangga: damage_grade bangunan lain dengan building_id berdekatan.

building_id berurutan mengikuti urutan survei, sehingga bangunan yang ID-nya
berdekatan (satu ward, disurvei berdekatan) cenderung punya tingkat kerusakan mirip.
Karena test adalah sampel acak dari populasi yang sama dengan train, label train
di sekitar sebuah bangunan test bisa dipakai sebagai fitur (mirip KNN).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WINDOWS = (3, 10, 30)
NBR_COLS = (
    [f"nbr_mean_w{w}" for w in WINDOWS]
    + [f"nbr_p{k}_w10" for k in range(1, 6)]
    + ["nbr_prev_grade", "nbr_next_grade", "nbr_prev_gap", "nbr_next_gap"]
)


class NeighborTargetFeatures(BaseEstimator, TransformerMixin):
    """Menambah fitur rata-rata/proporsi label dari tetangga terdekat menurut building_id.

    Saat fit_transform (data train), baris itu sendiri dikeluarkan dari jendela
    tetangga (leave-one-out) agar label tidak bocor ke fiturnya sendiri. Saat transform
    (data baru), tetangga diambil dari seluruh data train yang disimpan saat fit.
    """

    def fit(self, X, y):
        """Menyimpan building_id train (terurut) dan label one-hot kumulatif."""
        self._store(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit lalu transform data train dengan skema leave-one-out."""
        self._store(X, y)
        return self._features(X, loo=True)

    def transform(self, X):
        """Menghitung fitur tetangga untuk data baru dari label train."""
        return self._features(X, loo=False)

    def _store(self, X, y):
        """Mengurutkan train berdasarkan building_id dan menyimpan cumsum label."""
        ids = X["building_id"].to_numpy()
        order = np.argsort(ids, kind="stable")
        self.ids_ = ids[order]
        yy = np.asarray(y)[order].astype(int)
        self.y_ = yy
        onehot = np.eye(5)[yy]
        self.cum_ = np.vstack([np.zeros((1, 5)), np.cumsum(onehot, axis=0)]).astype(np.int32)
        self.onehot_ = onehot.astype(np.int8)

    def _window(self, lo, hi, self_idx=None):
        """Jumlah one-hot dalam [lo, hi) per baris, dikurangi baris sendiri bila loo."""
        n = len(self.ids_)
        lo = np.clip(lo, 0, n)
        hi = np.clip(hi, 0, n)
        s = self.cum_[hi] - self.cum_[lo]
        cnt = (hi - lo).astype(float)
        if self_idx is not None:
            s = s - self.onehot_[self_idx]
            cnt = cnt - 1
        return s, np.maximum(cnt, 1)

    def _features(self, X, loo):
        """Membangun DataFrame fitur tetangga dan menggabungkannya dengan X."""
        ids = X["building_id"].to_numpy()
        n = len(self.ids_)
        grades = np.arange(1, 6)
        if loo:
            pos = np.searchsorted(self.ids_, ids)          # posisi baris sendiri
            self_idx = pos
            left, right = pos, pos + 1                      # tetangga: <pos dan >pos
        else:
            pos = np.searchsorted(self.ids_, ids)
            self_idx = None
            left, right = pos, pos                          # tetangga: <pos dan >=pos
        out = {}
        for w in WINDOWS:
            s, c = self._window(left - w, right + w, self_idx)
            out[f"nbr_mean_w{w}"] = (s @ grades) / c
        s, c = self._window(left - 10, right + 10, self_idx)
        for k in range(5):
            out[f"nbr_p{k + 1}_w10"] = s[:, k] / c
        prev_i, next_i = left - 1, right
        ok_p, ok_n = prev_i >= 0, next_i < n
        out["nbr_prev_grade"] = np.where(ok_p, self.y_[np.clip(prev_i, 0, n - 1)] + 1, np.nan)
        out["nbr_next_grade"] = np.where(ok_n, self.y_[np.clip(next_i, 0, n - 1)] + 1, np.nan)
        out["nbr_prev_gap"] = np.where(
            ok_p, np.log1p(np.abs(ids - self.ids_[np.clip(prev_i, 0, n - 1)])), np.nan)
        out["nbr_next_gap"] = np.where(
            ok_n, np.log1p(np.abs(self.ids_[np.clip(next_i, 0, n - 1)] - ids)), np.nan)
        return pd.concat([X.reset_index(drop=True),
                          pd.DataFrame(out)[NBR_COLS]], axis=1).set_axis(X.index, axis=0)
