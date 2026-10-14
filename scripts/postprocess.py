"""Post-processing prediksi: bobot per kelas untuk mengoptimalkan macro-F1."""
import json

import numpy as np
from sklearn.metrics import f1_score

from scripts.config import MODEL_DIR

WEIGHT_PATH = MODEL_DIR / "class_weights.json"


def apply_class_weights(proba, weights):
    """Mengalikan probabilitas tiap kelas dengan bobotnya lalu mengambil argmax.

    Args:
        proba: array (n_sampel, n_kelas) hasil predict_proba.
        weights: array (n_kelas,) bobot positif per kelas.

    Returns:
        array label integer 0..n_kelas-1.
    """
    return np.argmax(np.asarray(proba) * np.asarray(weights), axis=1)


def fit_class_weights(proba, y, n_rounds=3, grid=None):
    """Mencari bobot kelas yang memaksimalkan macro-F1 dengan coordinate ascent.

    Tiap putaran, bobot satu kelas divariasikan pada grid (skala log) sementara bobot
    kelas lain dipegang tetap, dan nilai terbaik disimpan.

    Args:
        proba: array (n_sampel, n_kelas) probabilitas dari data validasi (bukan data latih).
        y: label asli (integer).
        n_rounds: jumlah putaran coordinate ascent.
        grid: kandidat log-bobot; default linspace(-1, 1, 41).

    Returns:
        array (n_kelas,) bobot terbaik.
    """
    grid = np.linspace(-1.0, 1.0, 41) if grid is None else grid
    k = proba.shape[1]
    logw = np.zeros(k)
    best = f1_score(y, apply_class_weights(proba, np.exp(logw)), average="macro")
    for _ in range(n_rounds):
        for c in range(k):
            for g in grid:
                trial = logw.copy()
                trial[c] = g
                s = f1_score(y, apply_class_weights(proba, np.exp(trial)), average="macro")
                if s > best + 1e-9:
                    best, logw = s, trial
    return np.exp(logw)


def save_weights(weights, path=WEIGHT_PATH):
    """Menyimpan bobot kelas ke file JSON."""
    path.write_text(json.dumps([float(w) for w in weights], indent=2))


def load_weights(path=WEIGHT_PATH):
    """Membaca bobot kelas dari JSON; mengembalikan None bila file tidak ada."""
    return np.array(json.loads(path.read_text())) if path.exists() else None
