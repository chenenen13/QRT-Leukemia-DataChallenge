"""
Survival models and baselines.

This module provides:
- Baseline regression model (ignores censoring)
- Cox Proportional Hazards wrapper
- Random Survival Forest wrapper
- Clustering-based risk estimation
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from .config import (
    RANDOM_STATE, RSF_DEFAULT_PARAMS, TARGET_TIME, TARGET_EVENT
)


class BaselineRiskModel:
    """
    Baseline model that predicts survival time with Ridge regression.
    
    Risk score is computed as negative predicted time (shorter time = higher risk).
    This baseline ignores censoring but provides a simple reference.
    """
    
    def __init__(
        self,
        preprocessor,
        alpha: float = 1.0,
        random_state: int = RANDOM_STATE
    ):
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.random_state = random_state
        self.pipeline_ = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the baseline model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : np.ndarray
            Survival structured array (event, time)
        """
        # Extract just the time component
        times = y["time"] if hasattr(y, "dtype") and "time" in y.dtype.names else y
        
        self.pipeline_ = Pipeline([
            ("prep", self.preprocessor),
            ("model", Ridge(alpha=self.alpha))
        ])
        self.pipeline_.fit(X, times)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
            
        Returns
        -------
        np.ndarray
            Risk scores (negative predicted time)
        """
        pred_time = self.pipeline_.predict(X)
        return -pred_time  # Shorter predicted time = higher risk


class ClusteringRiskModel:
    """
    Unsupervised risk model based on KMeans clustering.
    
    Clusters patients based on features, then assigns risk scores
    based on median survival time in each cluster.
    """
    
    def __init__(
        self,
        preprocessor,
        n_clusters: int = 3,
        random_state: int = RANDOM_STATE
    ):
        self.preprocessor = preprocessor
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.cluster_risk_map_ = None
        self._preprocessor_fitted = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit clustering model and compute cluster risk mapping.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : np.ndarray
            Survival structured array
        """
        from sklearn.base import clone
        
        # Fit preprocessor and transform
        self._preprocessor_fitted = clone(self.preprocessor)
        X_transformed = self._preprocessor_fitted.fit_transform(X)
        
        # Fit KMeans
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.kmeans_.fit_predict(X_transformed)
        
        # Compute risk mapping: negative median survival per cluster
        times = y["time"]
        df_tmp = pd.DataFrame({"cluster": cluster_labels, "time": times})
        cluster_medians = df_tmp.groupby("cluster")["time"].median()
        
        self.cluster_risk_map_ = {c: -float(m) for c, m in cluster_medians.items()}
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores based on cluster membership.
        """
        X_transformed = self._preprocessor_fitted.transform(X)
        clusters = self.kmeans_.predict(X_transformed)
        return np.array([self.cluster_risk_map_[c] for c in clusters])
    
    def get_cluster_summary(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Get survival statistics per cluster.
        
        Returns
        -------
        pd.DataFrame
            Summary with n, death_rate, os_median, os_mean per cluster
        """
        X_transformed = self._preprocessor_fitted.transform(X)
        clusters = self.kmeans_.predict(X_transformed)
        
        df = pd.DataFrame({
            "cluster": clusters,
            "time": y["time"],
            "event": y["event"]
        })
        
        return df.groupby("cluster").agg(
            n=("cluster", "size"),
            death_rate=("event", "mean"),
            os_median=("time", "median"),
            os_mean=("time", "mean")
        ).sort_values("os_median")


def tune_kmeans_clusters(
    X_transformed: np.ndarray,
    candidate_k: List[int] = [3, 4, 5, 6, 7, 8, 10],
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Tune number of clusters using silhouette score.
    
    Parameters
    ----------
    X_transformed : np.ndarray
        Preprocessed feature matrix
    candidate_k : List[int]
        Candidate values for k
    random_state : int
        Random state
        
    Returns
    -------
    tuple
        (best_k, silhouette_scores_dict)
    """
    from sklearn.metrics import silhouette_score
    
    scores = {}
    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_transformed)
        scores[k] = silhouette_score(X_transformed, labels)
    
    best_k = max(scores, key=scores.get)
    return best_k, scores


def create_rsf_model(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = RANDOM_STATE
):
    """
    Create a Random Survival Forest model.
    
    Parameters
    ----------
    params : Dict[str, Any], optional
        Model parameters. Uses defaults if not provided.
    random_state : int
        Random state
        
    Returns
    -------
    RandomSurvivalForest
        Configured model instance
    """
    from sksurv.ensemble import RandomSurvivalForest
    
    model_params = RSF_DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    
    return RandomSurvivalForest(
        random_state=random_state,
        n_jobs=-1,
        **model_params
    )


def create_cox_model(alpha: float = 0.0):
    """
    Create a Cox Proportional Hazards model.
    
    Parameters
    ----------
    alpha : float
        Regularization strength
        
    Returns
    -------
    CoxPHSurvivalAnalysis
        Configured model instance
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    
    return CoxPHSurvivalAnalysis(alpha=alpha)
