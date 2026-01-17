"""
Numba-optimized computations for performance-critical operations.

This module provides JIT-compiled functions for:
- Concordance index computation
- Distance calculations
- Feature aggregations
"""
import numpy as np
from numba import jit, prange
from typing import Tuple


@jit(nopython=True, parallel=True, cache=True)
def fast_concordance_pairs(
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray
) -> Tuple[int, int, int]:
    """
    Fast computation of concordant/discordant pairs for C-index.
    
    This is a basic concordance implementation, not IPCW-weighted.
    For the official challenge metric, use sksurv.metrics.concordance_index_ipcw.
    
    Parameters
    ----------
    event : np.ndarray
        Boolean array indicating if event occurred
    time : np.ndarray
        Survival times
    risk : np.ndarray
        Predicted risk scores
        
    Returns
    -------
    Tuple[int, int, int]
        (concordant, discordant, tied_risk)
    """
    n = len(time)
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in prange(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1
    
    return concordant, discordant, tied


@jit(nopython=True, cache=True)
def fast_cindex(
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray
) -> float:
    """
    Fast computation of Harrell's C-index.
    
    Parameters
    ----------
    event : np.ndarray
        Boolean array indicating if event occurred
    time : np.ndarray
        Survival times
    risk : np.ndarray
        Predicted risk scores
        
    Returns
    -------
    float
        C-index score
    """
    conc, disc, tied = fast_concordance_pairs(event, time, risk)
    total = conc + disc + tied
    
    if total == 0:
        return 0.5
    
    return (conc + 0.5 * tied) / total


@jit(nopython=True, parallel=True, cache=True)
def fast_aggregate_by_id(
    ids: np.ndarray,
    values: np.ndarray,
    unique_ids: np.ndarray
) -> np.ndarray:
    """
    Fast aggregation of values by ID (sum).
    
    Parameters
    ----------
    ids : np.ndarray
        ID array (integer-encoded)
    values : np.ndarray
        Values to aggregate
    unique_ids : np.ndarray
        Unique IDs in order
        
    Returns
    -------
    np.ndarray
        Aggregated sums per unique ID
    """
    n_unique = len(unique_ids)
    result = np.zeros(n_unique, dtype=np.float64)
    
    # Create mapping from id to index
    max_id = np.max(unique_ids) + 1
    id_to_idx = np.full(max_id, -1, dtype=np.int64)
    
    for i, uid in enumerate(unique_ids):
        id_to_idx[uid] = i
    
    # Aggregate
    for i in prange(len(ids)):
        idx = id_to_idx[ids[i]]
        if idx >= 0:
            result[idx] += values[i]
    
    return result


@jit(nopython=True, cache=True)
def fast_count_by_id(
    ids: np.ndarray,
    unique_ids: np.ndarray
) -> np.ndarray:
    """
    Fast count of occurrences by ID.
    
    Parameters
    ----------
    ids : np.ndarray
        ID array (integer-encoded)
    unique_ids : np.ndarray
        Unique IDs in order
        
    Returns
    -------
    np.ndarray
        Counts per unique ID
    """
    n_unique = len(unique_ids)
    result = np.zeros(n_unique, dtype=np.int64)
    
    max_id = np.max(unique_ids) + 1
    id_to_idx = np.full(max_id, -1, dtype=np.int64)
    
    for i, uid in enumerate(unique_ids):
        id_to_idx[uid] = i
    
    for i in range(len(ids)):
        idx = id_to_idx[ids[i]]
        if idx >= 0:
            result[idx] += 1
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Fast pairwise Euclidean distance computation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
        
    Returns
    -------
    np.ndarray
        Distance matrix (n_samples, n_samples)
    """
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in prange(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                d += diff * diff
            d = np.sqrt(d)
            dist[i, j] = d
            dist[j, i] = d
    
    return dist


@jit(nopython=True, cache=True)
def fast_vaf_stats(
    patient_ids: np.ndarray,
    vafs: np.ndarray,
    unique_patients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast computation of VAF statistics per patient.
    
    Parameters
    ----------
    patient_ids : np.ndarray
        Integer-encoded patient IDs
    vafs : np.ndarray
        VAF values
    unique_patients : np.ndarray
        Unique patient IDs
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (vaf_mean, vaf_std, vaf_max) per patient
    """
    n_patients = len(unique_patients)
    
    # First pass: compute counts and sums
    sums = np.zeros(n_patients)
    counts = np.zeros(n_patients)
    maxvals = np.full(n_patients, -np.inf)
    
    max_id = np.max(unique_patients) + 1
    id_to_idx = np.full(max_id, -1, dtype=np.int64)
    for i, uid in enumerate(unique_patients):
        id_to_idx[uid] = i
    
    for i in range(len(patient_ids)):
        if np.isnan(vafs[i]):
            continue
        idx = id_to_idx[patient_ids[i]]
        if idx >= 0:
            sums[idx] += vafs[i]
            counts[idx] += 1
            if vafs[i] > maxvals[idx]:
                maxvals[idx] = vafs[i]
    
    # Compute means
    means = np.zeros(n_patients)
    for i in range(n_patients):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
        else:
            maxvals[i] = 0.0
    
    # Second pass: compute variance
    var_sums = np.zeros(n_patients)
    for i in range(len(patient_ids)):
        if np.isnan(vafs[i]):
            continue
        idx = id_to_idx[patient_ids[i]]
        if idx >= 0:
            diff = vafs[i] - means[idx]
            var_sums[idx] += diff * diff
    
    stds = np.zeros(n_patients)
    for i in range(n_patients):
        if counts[i] > 1:
            stds[i] = np.sqrt(var_sums[i] / (counts[i] - 1))
    
    return means, stds, maxvals


def prepare_for_numba(df, id_col: str = "ID"):
    """
    Prepare dataframe for numba operations by encoding IDs as integers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_col : str
        ID column name
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        (encoded_ids, unique_ids, id_mapping)
    """
    import pandas as pd
    
    # Create integer encoding
    unique_ids = df[id_col].unique()
    id_to_int = {v: i for i, v in enumerate(unique_ids)}
    
    encoded = df[id_col].map(id_to_int).values.astype(np.int64)
    unique_encoded = np.arange(len(unique_ids), dtype=np.int64)
    
    return encoded, unique_encoded, id_to_int
