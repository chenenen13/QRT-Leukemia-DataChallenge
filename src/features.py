# ============================================================
# src/features.py
# ============================================================
"""
Feature engineering for molecular and clinical data.

This module contains functions to aggregate mutation-level data
to patient-level features.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .config import TOP_GENES, TOP_EFFECTS


def get_top_lists(
    mol_df: pd.DataFrame,
    top_genes: int = TOP_GENES,
    top_effects: int = TOP_EFFECTS,
) -> Tuple[List[str], List[str]]:
    """
    Compute top genes/effects on TRAIN only (to avoid leakage).
    """
    gene_list = mol_df["GENE"].value_counts().head(top_genes).index.tolist()
    effect_list = mol_df["EFFECT"].value_counts().head(top_effects).index.tolist()
    return gene_list, effect_list


def build_molecular_features(
    mol_df: pd.DataFrame,
    top_genes: int = TOP_GENES,
    top_effects: int = TOP_EFFECTS,
    gene_list: Optional[List[str]] = None,
    effect_list: Optional[List[str]] = None,
    binary_indicators: bool = True,
) -> pd.DataFrame:
    """
    Aggregate mutation-level data to patient-level features.

    Creates the following features per patient:
    - n_mut: total number of mutations
    - n_unique_genes: number of unique mutated genes
    - n_unique_effects: number of unique mutation effects
    - vaf_mean, vaf_std, vaf_max: VAF statistics
    - vaf_p90, vaf_sum, n_vaf_gt_30, n_vaf_lt_10: additional clonal burden proxies
    - depth_mean, depth_max: sequencing depth statistics
    - GENE__<gene>: indicator (binary by default) for each selected gene
    - EFFECT__<effect>: indicator (binary by default) for each selected effect

    IMPORTANT:
    - To prevent leakage, compute (gene_list, effect_list) on TRAIN and reuse for TEST.

    Parameters
    ----------
    mol_df : pd.DataFrame
        Molecular data with one row per mutation
    top_genes : int
        Number of top genes to include as features (if gene_list is None)
    top_effects : int
        Number of top effects to include as features (if effect_list is None)
    gene_list : List[str], optional
        Explicit list of genes to use (recommended: computed on TRAIN)
    effect_list : List[str], optional
        Explicit list of effects to use (recommended: computed on TRAIN)
    binary_indicators : bool
        If True, GENE__/EFFECT__ columns are binary indicators (presence/absence).
        If False, they are counts.

    Returns
    -------
    pd.DataFrame
        Patient-level molecular features
    """
    df = mol_df.copy()

    # Ensure numeric columns
    df["VAF"] = pd.to_numeric(df["VAF"], errors="coerce")
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    # Basic aggregates + richer VAF stats
    def _p90(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return np.nanpercentile(x, 90) if len(x) else np.nan

    agg = df.groupby("ID").agg(
        n_mut=("GENE", "size"),
        n_unique_genes=("GENE", "nunique"),
        n_unique_effects=("EFFECT", "nunique"),
        vaf_mean=("VAF", "mean"),
        vaf_std=("VAF", "std"),
        vaf_max=("VAF", "max"),
        vaf_p90=("VAF", _p90),
        vaf_sum=("VAF", "sum"),
        n_vaf_gt_30=("VAF", lambda x: np.sum(pd.to_numeric(x, errors="coerce") > 0.30)),
        n_vaf_lt_10=("VAF", lambda x: np.sum(pd.to_numeric(x, errors="coerce") < 0.10)),
        depth_mean=("DEPTH", "mean"),
        depth_max=("DEPTH", "max"),
    ).reset_index()

    # Top genes list
    if gene_list is None:
        gene_list = df["GENE"].value_counts().head(top_genes).index.tolist()

    # Gene pivot (binary or counts)
    gene_pivot = (
        df[df["GENE"].isin(gene_list)]
        .assign(val=1)
        .pivot_table(
            index="ID",
            columns="GENE",
            values="val",
            aggfunc=("max" if binary_indicators else "sum"),
            fill_value=0,
        )
    )
    gene_pivot.columns = [f"GENE__{c}" for c in gene_pivot.columns]
    gene_pivot = gene_pivot.reset_index()

    # Top effects list
    if effect_list is None:
        effect_list = df["EFFECT"].value_counts().head(top_effects).index.tolist()

    # Effect pivot (binary or counts)
    eff_pivot = (
        df[df["EFFECT"].isin(effect_list)]
        .assign(val=1)
        .pivot_table(
            index="ID",
            columns="EFFECT",
            values="val",
            aggfunc=("max" if binary_indicators else "sum"),
            fill_value=0,
        )
    )
    eff_pivot.columns = [f"EFFECT__{c}" for c in eff_pivot.columns]
    eff_pivot = eff_pivot.reset_index()

    # Merge all
    out = agg.merge(gene_pivot, on="ID", how="left").merge(eff_pivot, on="ID", how="left")
    return out.fillna(0)


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of feature columns (excluding ID and target columns).
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
    - is_normal_karyotype: exactly "46,XX" or "46,XY" (no abnormalities)
    - has_deletion: presence of del()
    - has_monosomy_7: -7 or monosomy 7
    - has_trisomy_8: +8 or trisomy 8
    - has_translocation: t() notation
    - n_abnormalities: rough count of abnormality tokens

    Notes:
    - ISCN strings are messy; keep features robust & simple.
    """
    import re

    feats = []
    for cyto in cyto_series.fillna(""):
        s = str(cyto).strip().lower()

        feats.append(
            {
                "cyto_is_normal_karyotype": int(bool(re.match(r"^46,(xx|xy)$", s))),
                "cyto_has_deletion": int("del(" in s),
                "cyto_has_monosomy_7": int("-7" in s or "monosomy 7" in s),
                "cyto_has_trisomy_8": int("+8" in s or "trisomy 8" in s),
                "cyto_has_translocation": int("t(" in s),
                # heuristic token count (not perfect but works in boosting)
                "cyto_n_abnormalities": int(
                    len(re.findall(r"del\(|inv\(|dup\(|t\(|\+\d+|-\d+", s))
                ),
            }
        )

    return pd.DataFrame(feats)
