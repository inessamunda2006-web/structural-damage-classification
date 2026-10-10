"""Konfigurasi global: path, nama kolom, dan seed untuk seluruh proyek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
FIG_DIR = ROOT / "reports" / "figures"

SEED = 42
ID_COL = "building_id"
TARGET = "damage_grade"

# Urutan kelas dipakai untuk encode/decode label (ordinal: Grade 1 < ... < Grade 5)
CLASSES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]

GEO_COLS = ["district_id", "vdcmun_id", "ward_id"]
NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]
