"""
Feature engineering for molecular and clinical data.

This module contains functions to aggregate mutation-level data 
to patient-level features.
"""
import numpy as np
import pandas as pd
from typing import List, Optional

from .config import TOP_GENES, TOP_EFFECTS


def build_molecular_features(
    mol_df: pd.DataFrame,
    top_genes: int = TOP_GENES,
    top_effects: int = TOP_EFFECTS,
    gene_list: Optional[List[str]] = None,
    effect_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate mutation-level data to patient-level features.
    
    Creates the following features per patient:
    - n_mut: total number of mutations
    - n_unique_genes: number of unique mutated genes
    - n_unique_effects: number of unique mutation effects
    - vaf_mean, vaf_std, vaf_max: VAF statistics
    - depth_mean, depth_max: sequencing depth statistics
    - GENE__<gene>: binary indicator for each top gene
    - EFFECT__<effect>: binary indicator for each top effect
    
    Parameters
    ----------
    mol_df : pd.DataFrame
        Molecular data with one row per mutation
    top_genes : int
        Number of top genes to include as features
    top_effects : int
        Number of top effects to include as features
    gene_list : List[str], optional
        Explicit list of genes to use (overrides top_genes)
    effect_list : List[str], optional
        Explicit list of effects to use (overrides top_effects)
        
    Returns
    -------
    pd.DataFrame
        Patient-level molecular features
    """
    df = mol_df.copy()
    
    # Ensure numeric columns
    df["VAF"] = pd.to_numeric(df["VAF"], errors="coerce")
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    
    # Basic aggregates
    agg = df.groupby("ID").agg(
        n_mut=("GENE", "size"),
        n_unique_genes=("GENE", "nunique"),
        n_unique_effects=("EFFECT", "nunique"),
        vaf_mean=("VAF", "mean"),
        vaf_std=("VAF", "std"),
        vaf_max=("VAF", "max"),
        depth_mean=("DEPTH", "mean"),
        depth_max=("DEPTH", "max"),
    ).reset_index()
    
    # Top genes pivot
    if gene_list is None:
        gene_list = df["GENE"].value_counts().head(top_genes).index.tolist()
    
    gene_counts = (
        df[df["GENE"].isin(gene_list)]
        .assign(val=1)
        .pivot_table(index="ID", columns="GENE", values="val", aggfunc="sum", fill_value=0)
    )
    gene_counts.columns = [f"GENE__{c}" for c in gene_counts.columns]
    gene_counts = gene_counts.reset_index()
    
    # Top effects pivot
    if effect_list is None:
        effect_list = df["EFFECT"].value_counts().head(top_effects).index.tolist()
    
    eff_counts = (
        df[df["EFFECT"].isin(effect_list)]
        .assign(val=1)
        .pivot_table(index="ID", columns="EFFECT", values="val", aggfunc="sum", fill_value=0)
    )
    eff_counts.columns = [f"EFFECT__{c}" for c in eff_counts.columns]
    eff_counts = eff_counts.reset_index()
    
    # Merge all
    out = agg.merge(gene_counts, on="ID", how="left").merge(eff_counts, on="ID", how="left")
    return out.fillna(0)


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of feature columns (excluding ID and target columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with features and targets
    exclude_cols : List[str], optional
        Additional columns to exclude
        
    Returns
    -------
    List[str]
        Feature column names
    """
    from .config import ID_COL, TARGET_TIME, TARGET_EVENT
    
    exclude = {ID_COL, TARGET_TIME, TARGET_EVENT}
    if exclude_cols:
        exclude.update(exclude_cols)
    
    return [c for c in df.columns if c not in exclude]


def extract_cytogenetics_features(cyto_series: pd.Series) -> pd.DataFrame:
    """
    Extract structured features from cytogenetics text (ISCN notation).
    
    Extracts:
    - is_normal_karyotype: 46,XX or 46,XY without abnormalities
    - has_deletion: presence of del()
    - has_monosomy_7: -7 or monosomy 7
    - has_trisomy_8: +8 or trisomy 8
    - has_translocation: t() notation
    - n_abnormalities: count of abnormalities
    
    Parameters
    ----------
    cyto_series : pd.Series
        Series of cytogenetics strings
        
    Returns
    -------
    pd.DataFrame
        Extracted cytogenetics features
    """
    import re
    
    features = []
    
    for cyto in cyto_series.fillna(""):
        cyto_str = str(cyto).lower()
        
        feat = {
            "is_normal_karyotype": int(bool(re.match(r"^46,(xx|xy)$", cyto_str.strip()))),
            "has_deletion": int("del(" in cyto_str),
            "has_monosomy_7": int("-7" in cyto_str or "monosomy 7" in cyto_str),
            "has_trisomy_8": int("+8" in cyto_str or "trisomy 8" in cyto_str),
            "has_translocation": int("t(" in cyto_str),
            "n_abnormalities": len(re.findall(r"[\+\-]?\d+|del|inv|t\(|dup", cyto_str)),
        }
        features.append(feat)
    
    return pd.DataFrame(features)
