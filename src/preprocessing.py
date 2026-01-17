"""
Preprocessing pipelines for clinical, categorical, and text data.
"""
import numpy as np
from typing import List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .config import (
    CLINICAL_NUMERIC_COLS, CLINICAL_CAT_COLS, CLINICAL_TEXT_COL,
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
    - Text (cytogenetics): TF-IDF + SVD dimensionality reduction
    
    Parameters
    ----------
    num_cols : List[str], optional
        Numeric column names
    cat_cols : List[str], optional
        Categorical column names
    text_col : str, optional
        Text column name (cytogenetics)
    tfidf_max_features : int
        Max features for TF-IDF
    svd_components : int
        Number of SVD components for text
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """
    transformers = []
    
    # Numeric pipeline
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))
    
    # Categorical pipeline  
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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
                ngram_range=(1, 2),
                lowercase=True
            )),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=random_state)),
        ])
        transformers.append(("txt", text_pipeline, [text_col]))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_default_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """
    Create default preprocessor based on known column structure.
    
    Automatically identifies:
    - Numeric columns from CLINICAL_NUMERIC_COLS + molecular aggregates
    - Categorical columns from CLINICAL_CAT_COLS
    - Text column (CYTOGENETICS)
    
    Parameters
    ----------
    feature_cols : List[str]
        All feature column names
        
    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline
    """
    # Identify column types
    num_cols = [c for c in feature_cols 
                if c not in CLINICAL_CAT_COLS and c != CLINICAL_TEXT_COL]
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None
    
    return create_preprocessing_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        text_col=text_col
    )
