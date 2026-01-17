"""
Evaluation metrics and cross-validation utilities for survival analysis.

This module provides:
- IPCW C-index computation
- Survival data format conversion for scikit-survival
- Cross-validation with survival metrics
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TAU_YEARS, TARGET_EVENT, TARGET_TIME, N_SPLITS_CV


def to_sksurv_y(df: pd.DataFrame) -> np.ndarray:
    """
    Convert dataframe to scikit-survival structured array.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OS_STATUS and OS_YEARS columns
        
    Returns
    -------
    np.ndarray
        Structured array with ('event', 'time') dtype
    """
    event = df[TARGET_EVENT].astype(int).values == 1
    time = df[TARGET_TIME].astype(float).values
    return np.array(list(zip(event, time)), dtype=[("event", "bool"), ("time", "f8")])


def ipcw_cindex(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    tau: float = TAU_YEARS
) -> float:
    """
    Compute IPCW (Inverse Probability of Censoring Weighted) C-index.
    
    This is the challenge evaluation metric, truncated at tau years.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training survival data (for censoring estimation)
    y_test : np.ndarray
        Test survival data
    risk_scores : np.ndarray
        Predicted risk scores (higher = higher risk)
    tau : float
        Truncation time in years
        
    Returns
    -------
    float
        IPCW C-index score
    """
    from sksurv.metrics import concordance_index_ipcw
    
    result = concordance_index_ipcw(y_train, y_test, risk_scores, tau=tau)
    return float(result[0])


def cross_validate_survival(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_factory,
    preprocessor,
    n_splits: int = N_SPLITS_CV,
    tau: float = TAU_YEARS,
    random_state: int = RANDOM_STATE
) -> List[float]:
    """
    Perform cross-validation with IPCW C-index for survival models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full training dataframe with features and targets
    feature_cols : List[str]
        Feature column names
    model_factory : callable
        Function that returns a new model instance
    preprocessor : sklearn transformer
        Preprocessing pipeline (will be cloned per fold)
    n_splits : int
        Number of CV folds
    tau : float
        Truncation time for IPCW C-index
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    List[float]
        IPCW C-index scores for each fold
    """
    from sklearn.base import clone
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, test_idx in kf.split(df):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        y_train = to_sksurv_y(df_train)
        y_test = to_sksurv_y(df_test)
        
        # Clone preprocessor for this fold
        prep = clone(preprocessor)
        model = model_factory()
        
        pipeline = Pipeline([("prep", prep), ("model", model)])
        pipeline.fit(df_train[feature_cols], y_train)
        
        risk = pipeline.predict(df_test[feature_cols])
        score = ipcw_cindex(y_train, y_test, risk, tau=tau)
        scores.append(score)
    
    return scores


def grid_search_survival(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_class,
    param_grid: Dict[str, List[Any]],
    preprocessor,
    n_splits: int = 2,  # Reduced from 3 to 2 for speed
    tau: float = TAU_YEARS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
    fast_mode: bool = True  # Pre-transform data once to save time
) -> Tuple[Dict[str, Any], float]:
    """
    Optimized grid search with cross-validation for survival models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe
    feature_cols : List[str]
        Feature column names
    model_class : class
        Model class to instantiate
    param_grid : Dict[str, List[Any]]
        Hyperparameter grid
    preprocessor : sklearn transformer
        Preprocessing pipeline
    n_splits : int
        CV folds (default: 2 for speed)
    tau : float
        IPCW truncation time
    random_state : int
        Random state
    verbose : bool
        Print progress
    fast_mode : bool
        If True, pre-transform data once to avoid repeated preprocessing
        
    Returns
    -------
    Tuple[Dict[str, Any], float]
        (best_params, best_score)
    """
    from itertools import product
    from sklearn.base import clone
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    n_combinations = 1
    for v in param_values:
        n_combinations *= len(v)
    total_fits = n_combinations * n_splits
    
    if verbose:
        print(f"Grid Search: {n_combinations} param combinations × {n_splits} folds = {total_fits} fits")
    
    best_params = None
    best_score = -1.0
    
    if fast_mode:
        # PRE-TRANSFORM DATA ONCE - huge speedup!
        # Fit preprocessor on all data (acceptable for hyperparameter tuning)
        prep = clone(preprocessor)
        X_transformed = prep.fit_transform(df[feature_cols])
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        X_df = pd.DataFrame(X_transformed)
        
        y_all = to_sksurv_y(df)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(kf.split(X_df))
        
        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))
            fold_scores = []
            
            for train_idx, test_idx in fold_indices:
                X_train = X_df.iloc[train_idx].values
                X_test = X_df.iloc[test_idx].values
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                
                model = model_class(random_state=random_state, n_jobs=-1, **params)
                model.fit(X_train, y_train)
                
                risk = model.predict(X_test)
                score = ipcw_cindex(y_train, y_test, risk, tau=tau)
                fold_scores.append(score)
            
            mean_score = float(np.mean(fold_scores))
            
            if verbose:
                print(f"[{i+1}/{n_combinations}] {params} -> CV: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
    else:
        # Original slow mode: full pipeline per fold
        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))
            
            def model_factory():
                return model_class(random_state=random_state, n_jobs=-1, **params)
            
            scores = cross_validate_survival(
                df, feature_cols, model_factory, preprocessor,
                n_splits=n_splits, tau=tau, random_state=random_state
            )
            mean_score = float(np.mean(scores))
            
            if verbose:
                print(f"[{i+1}/{n_combinations}] {params} -> CV: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
    
    return best_params, best_score


def permutation_importance_survival(
    model,
    df_valid: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: List[str],
    cols_to_test: List[str],
    base_score: float,
    n_repeats: int = 5,
    tau: float = TAU_YEARS,
    random_state: int = RANDOM_STATE
) -> pd.Series:
    """
    Compute permutation importance for survival models.
    
    Measures the drop in IPCW C-index when each feature is permuted.
    
    Parameters
    ----------
    model : fitted model/pipeline
        Model with predict() method
    df_valid : pd.DataFrame
        Validation dataframe
    y_train : np.ndarray
        Training survival array (for IPCW estimation)
    feature_cols : List[str]
        All feature columns
    cols_to_test : List[str]
        Columns to compute importance for
    base_score : float
        Baseline IPCW C-index
    n_repeats : int
        Number of permutations per feature
    tau : float
        IPCW truncation time
    random_state : int
        Random state
        
    Returns
    -------
    pd.Series
        Feature importances (score drop), sorted descending
    """
    rng = np.random.RandomState(random_state)
    importances = {}
    
    X_valid = df_valid[feature_cols].copy()
    y_valid = to_sksurv_y(df_valid)
    
    for col in cols_to_test:
        if col not in X_valid.columns:
            continue
            
        drops = []
        for _ in range(n_repeats):
            X_perm = X_valid.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            risk = model.predict(X_perm)
            score = ipcw_cindex(y_train, y_valid, risk, tau=tau)
            drops.append(base_score - score)
        
        importances[col] = float(np.mean(drops))
    
    return pd.Series(importances).sort_values(ascending=False)
