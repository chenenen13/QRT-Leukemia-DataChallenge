# ============================================================
# src/preprocessing.py
# ============================================================
"""
Preprocessing pipelines for clinical, categorical, and text data.
"""
from typing import List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .config import (
    CLINICAL_CAT_COLS, CLINICAL_TEXT_COL,
    RANDOM_STATE, TFIDF_MAX_FEATURES, SVD_COMPONENTS
)


def create_preprocessing_pipeline(
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    text_col: Optional[str] = None,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
    svd_components: int = SVD_COMPONENTS,
    random_state: int = RANDOM_STATE
) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing features.

    Pipeline structure:
    - Numeric: median imputation + standardization
    - Categorical: most frequent imputation + one-hot encoding
    - Text (cytogenetics): TF-IDF (char n-grams) + SVD

    Notes:
    - For ISCN cytogenetics, char n-grams are generally superior to word n-grams.
    """
    transformers = []

    # Numeric pipeline
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    # Categorical pipeline
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    # Text pipeline (cytogenetics)
    if text_col:
        text_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_1d", FunctionTransformer(lambda x: x.ravel(), validate=False)),
            ("to_str", FunctionTransformer(lambda x: x.astype(str), validate=False)),
            ("tfidf", TfidfVectorizer(
                max_features=tfidf_max_features,
                analyzer="char",
                ngram_range=(3, 6),
                lowercase=True,
            )),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
        ])
        transformers.append(("txt", text_pipeline, [text_col]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_default_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """
    Create default preprocessor based on known column structure.

    Automatically identifies:
    - Numeric: everything except categorical + text
    - Categorical: CLINICAL_CAT_COLS if present
    - Text: CYTOGENETICS if present
    """
    num_cols = [c for c in feature_cols if c not in CLINICAL_CAT_COLS and c != CLINICAL_TEXT_COL]
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None

    return create_preprocessing_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        text_col=text_col,
    )
