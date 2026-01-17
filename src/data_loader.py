"""
Data loading and validation utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from .config import (
    CLINICAL_TRAIN_PATH, CLINICAL_TEST_PATH,
    MOLECULAR_TRAIN_PATH, MOLECULAR_TEST_PATH,
    TARGET_TRAIN_PATH, ID_COL, TARGET_TIME, TARGET_EVENT
)


def load_clinical_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clinical train and test data.
    
    Parameters
    ----------
    train_path : Path, optional
        Path to clinical train CSV. Defaults to config path.
    test_path : Path, optional
        Path to clinical test CSV. Defaults to config path.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (clinical_train, clinical_test)
    """
    train_path = train_path or CLINICAL_TRAIN_PATH
    test_path = test_path or CLINICAL_TEST_PATH
    
    clinical_train = pd.read_csv(train_path)
    clinical_test = pd.read_csv(test_path)
    
    return clinical_train, clinical_test


def load_molecular_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load molecular (mutation) train and test data.
    
    Parameters
    ----------
    train_path : Path, optional
        Path to molecular train CSV. Defaults to config path.
    test_path : Path, optional
        Path to molecular test CSV. Defaults to config path.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (molecular_train, molecular_test)
    """
    train_path = train_path or MOLECULAR_TRAIN_PATH
    test_path = test_path or MOLECULAR_TEST_PATH
    
    molecular_train = pd.read_csv(train_path)
    molecular_test = pd.read_csv(test_path)
    
    return molecular_train, molecular_test


def load_target(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load target (survival) data.
    
    Parameters
    ----------
    path : Path, optional
        Path to target CSV. Defaults to config path.
        
    Returns
    -------
    pd.DataFrame
        Target dataframe with ID, OS_YEARS, OS_STATUS
    """
    path = path or TARGET_TRAIN_PATH
    return pd.read_csv(path)


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                              pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets at once.
    
    Returns
    -------
    Tuple containing:
        - clinical_train
        - clinical_test  
        - molecular_train
        - molecular_test
        - target_train
    """
    clinical_train, clinical_test = load_clinical_data()
    molecular_train, molecular_test = load_molecular_data()
    target_train = load_target()
    
    return clinical_train, clinical_test, molecular_train, molecular_test, target_train


def validate_data(
    clinical_train: pd.DataFrame,
    target_train: pd.DataFrame
) -> dict:
    """
    Perform sanity checks on the data.
    
    Parameters
    ----------
    clinical_train : pd.DataFrame
        Clinical training data
    target_train : pd.DataFrame
        Target training data
        
    Returns
    -------
    dict
        Validation results with keys:
        - duplicated_clinical_ids: int
        - duplicated_target_ids: int
        - missing_targets: int
        - censoring_rate: float
        - death_rate: float
    """
    results = {}
    
    # Check for duplicates
    results["duplicated_clinical_ids"] = clinical_train[ID_COL].duplicated().sum()
    results["duplicated_target_ids"] = target_train[ID_COL].duplicated().sum()
    
    # Check for missing targets
    missing = set(clinical_train[ID_COL]) - set(target_train[ID_COL])
    results["missing_targets"] = len(missing)
    
    # Compute censoring and death rates
    results["death_rate"] = target_train[TARGET_EVENT].mean()
    results["censoring_rate"] = 1 - results["death_rate"]
    
    return results


def merge_train_data(
    clinical: pd.DataFrame,
    molecular_features: pd.DataFrame,
    target: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge clinical, molecular features, and target into a single dataframe.
    
    Parameters
    ----------
    clinical : pd.DataFrame
        Clinical data
    molecular_features : pd.DataFrame
        Aggregated molecular features (one row per patient)
    target : pd.DataFrame
        Target data with OS_YEARS and OS_STATUS
        
    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """
    merged = clinical.merge(molecular_features, on=ID_COL, how="left").fillna(0)
    merged = merged.merge(target, on=ID_COL, how="inner")
    return merged


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean target columns: drop NaN, cast types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OS_YEARS and OS_STATUS columns
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.dropna(subset=[TARGET_EVENT, TARGET_TIME]).copy()
    df[TARGET_EVENT] = df[TARGET_EVENT].astype(int)
    df[TARGET_TIME] = df[TARGET_TIME].astype(float)
    return df
