# ============================================================
# src/config.py
# ============================================================
"""
Configuration and constants for the Leukemia Risk Prediction project.
"""
from pathlib import Path
from typing import List

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data files
CLINICAL_TRAIN_PATH = DATA_DIR / "clinical_train.csv"
CLINICAL_TEST_PATH = DATA_DIR / "clinical_test.csv"
MOLECULAR_TRAIN_PATH = DATA_DIR / "molecular_train.csv"
MOLECULAR_TEST_PATH = DATA_DIR / "molecular_test.csv"
TARGET_TRAIN_PATH = DATA_DIR / "target_train.csv"

# =========================================================
# Column names
# =========================================================
ID_COL = "ID"
TARGET_TIME = "OS_YEARS"
TARGET_EVENT = "OS_STATUS"

# Clinical numeric columns
CLINICAL_NUMERIC_COLS: List[str] = [
    "BM_BLAST",  # Bone marrow blasts %
    "WBC",       # White Blood Cell count (Giga/L)
    "ANC",       # Absolute Neutrophil count (Giga/L)
    "MONOCYTES", # Monocyte count (Giga/L)
    "HB",        # Hemoglobin (g/dL)
    "PLT",       # Platelets count (Giga/L)
]

# Clinical categorical columns
CLINICAL_CAT_COLS: List[str] = ["CENTER"]

# Clinical text column (cytogenetics - ISCN notation)
CLINICAL_TEXT_COL = "CYTOGENETICS"

# Molecular columns
MOLECULAR_COLS = [
    "ID", "CHR", "START", "END", "REF", "ALT",
    "GENE", "PROTEIN_CHANGE", "EFFECT", "VAF", "DEPTH"
]

# =========================================================
# Model parameters
# =========================================================
RANDOM_STATE = 42
TAU_YEARS = 7.0  # Truncation time for IPCW C-index

# Cross-validation
N_SPLITS_CV = 3
TEST_SIZE = 0.2

# =========================================================
# Feature engineering
# =========================================================
TOP_GENES = 60      # Number of top genes to use as features (selected on TRAIN only)
TOP_EFFECTS = 25    # Number of top mutation effects to use (selected on TRAIN only)

# Cytogenetics text encoding
# NOTE: ISCN is best captured with char n-grams
TFIDF_MAX_FEATURES = 10000
SVD_COMPONENTS = 150

# =========================================================
# Random Survival Forest default parameters
# =========================================================
RSF_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 10,
    "min_samples_split": 10,
    "max_features": "sqrt",
}

# Hyperparameter grid for RSF tuning (REDUCED for speed)
RSF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "min_samples_leaf": [10, 20],
    "min_samples_split": [10],
    "max_features": ["sqrt", 0.5],
}
