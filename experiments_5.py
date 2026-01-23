# ============================================================
# experiments_5.py
# Goal: push robust CV up by adding high-signal CYTO regex features
#       + stratified CV on (event x time-quantile)
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.config import (
    RANDOM_STATE, ID_COL, TARGET_TIME, TARGET_EVENT,
    TOP_GENES, TOP_EFFECTS, CLINICAL_CAT_COLS, CLINICAL_TEXT_COL
)
from src.data_loader import load_all_data, merge_train_data, clean_target
from src.features import (
    build_molecular_features,
    get_feature_columns,
    extract_cytogenetics_features,
    get_top_lists,
)
from src.evaluation import to_sksurv_y, ipcw_cindex

np.random.seed(RANDOM_STATE)

# ---- Frozen best text params (from your search)
TFIDF_MAX = 5000
SVD_K = 120

# ---- Models (safe-ish)
RSF_PARAMS_SAFE = dict(
    n_estimators=200,
    min_samples_leaf=10,
    min_samples_split=10,
    max_features="sqrt",
    max_samples=0.7,
    random_state=RANDOM_STATE,
    n_jobs=2
)

GBSA_TUNED_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    verbose=0
)

COXNET_PARAMS = dict(
    l1_ratio=0.9,
    alpha_min_ratio=0.01,
    n_alphas=100
)

def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def rank01(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    if len(r) <= 1:
        return np.zeros_like(r, dtype=float)
    return (r - 1.0) / (len(r) - 1.0)

# ============================================================
# New: stronger CYTO targeted features (AML/MDS-ish patterns)
# ============================================================
def extract_cyto_aml_rules(cyto_series: pd.Series) -> pd.DataFrame:
    """
    Targeted cytogenetics flags beyond generic regex.
    These are intentionally simple / robust string rules.
    """
    feats = []
    for s in cyto_series.fillna(""):
        t = str(s).lower().replace(" ", "")
        f = {}

        # core patterns
        f["cyto_has_complex"] = int(("complex" in t) or ("cx" in t) or ("≥3" in t) or (">=3" in t))
        f["cyto_has_monosomal_karyotype"] = int(("monosomal" in t) or ("mk" in t))
        f["cyto_has_del5q"] = int(("del(5q" in t) or ("-5" in t) or ("5q-" in t))
        f["cyto_has_mono7_or_del7q"] = int(("-7" in t) or ("del(7q" in t) or ("7q-" in t))
        f["cyto_has_trisomy8"] = int(("+8" in t) or ("trisomy8" in t))
        f["cyto_has_inv16"] = int(("inv(16" in t) or ("t(16;16" in t))
        f["cyto_has_t8_21"] = int(("t(8;21" in t))
        f["cyto_has_t15_17"] = int(("t(15;17" in t))
        f["cyto_has_11q23_kmt2a"] = int(("11q23" in t) or ("kmt2a" in t) or ("mll" in t))
        f["cyto_has_3q_abn"] = int(("3q" in t) or ("inv(3" in t) or ("t(3;3" in t))
        f["cyto_has_17p"] = int(("17p" in t) or ("del(17p" in t))
        f["cyto_has_20q_del"] = int(("del(20q" in t) or ("20q-" in t))
        f["cyto_has_12p_abn"] = int(("12p" in t) or ("del(12p" in t))

        # crude count of abnormalities
        # (kept simple to avoid fragile ISCN parsing)
        f["cyto_abn_count_proxy"] = int(len(re.findall(r"(del\(|inv\(|t\(|dup\(|\+\d+|-\d+)", t)))

        feats.append(f)
    return pd.DataFrame(feats)

# ============================================================
# Preprocessor
# - num: includes molecular aggregates + cyto regex structured + cyto AML rules
# - cat: center
# - txt: cytogenetics TFIDF char ngrams + SVD
# ============================================================
def make_preprocessor(feature_cols):
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None
    num_cols = [c for c in feature_cols if c not in cat_cols and c != text_col]

    transformers = []

    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols
        ))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols
        ))

    if text_col:
        transformers.append((
            "txt",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("to_1d", FunctionTransformer(lambda x: x.ravel(), validate=False)),
                ("to_str", FunctionTransformer(lambda x: x.astype(str), validate=False)),
                ("tfidf", TfidfVectorizer(
                    max_features=TFIDF_MAX,
                    analyzer="char",
                    ngram_range=(3, 6),
                    lowercase=True,
                )),
                ("svd", TruncatedSVD(n_components=SVD_K, random_state=RANDOM_STATE)),
            ]),
            [text_col]
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")

# ============================================================
# Data assembly
# ============================================================
def build_train_table():
    header("Load data")
    clinical_train, _, molecular_train, _, y_train = load_all_data()

    header("Add CYTO structured features (generic + AML rules)")
    cyto_basic = extract_cytogenetics_features(clinical_train[CLINICAL_TEXT_COL])
    cyto_basic.columns = [f"cyto_{c}" if not c.startswith("cyto_") else c for c in cyto_basic.columns]
    cyto_rules = extract_cyto_aml_rules(clinical_train[CLINICAL_TEXT_COL])

    clinical_aug = pd.concat(
        [clinical_train.reset_index(drop=True), cyto_basic.reset_index(drop=True), cyto_rules.reset_index(drop=True)],
        axis=1
    )
    print(f"Added cyto features: {list(cyto_basic.columns) + list(cyto_rules.columns)}")

    header("Build molecular features (lists from TRAIN only)")
    gene_list, effect_list = get_top_lists(molecular_train, top_genes=TOP_GENES, top_effects=TOP_EFFECTS)
    mol_feat = build_molecular_features(
        molecular_train,
        gene_list=gene_list,
        effect_list=effect_list,
        binary_indicators=True
    )
    print(f"Top genes={len(gene_list)} | top effects={len(effect_list)}")
    print(f"Mol features train: {mol_feat.shape}")

    header("Merge clinical + molecular + target")
    X_train_full = merge_train_data(clinical_aug, mol_feat, y_train)
    train_full = clean_target(X_train_full)

    feature_cols = get_feature_columns(train_full)
    print(f"Train full: {train_full.shape} | #features: {len(feature_cols)}")
    return train_full, feature_cols

# ============================================================
# Better stratification: (event x time-bin)
# ============================================================
def make_strata(df: pd.DataFrame, n_bins: int = 3) -> np.ndarray:
    event = df[TARGET_EVENT].astype(int)
    # qcut can fail if many ties; fall back to cut on ranks
    t = df[TARGET_TIME].astype(float)
    try:
        bins = pd.qcut(t, q=n_bins, labels=False, duplicates="drop")
        if bins.isna().any():
            raise ValueError("qcut produced NaNs")
    except Exception:
        r = t.rank(method="average")
        bins = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")
    bins = bins.astype(int)
    strata = (event.astype(str) + "_" + bins.astype(str)).values
    return strata

# ============================================================
# Per-fold fit / eval
# ============================================================
def eval_fold(train_df, valid_df, feature_cols):
    pre = make_preprocessor(feature_cols)
    Xtr = pre.fit_transform(train_df[feature_cols])
    Xva = pre.transform(valid_df[feature_cols])

    ytr = to_sksurv_y(train_df)
    yva = to_sksurv_y(valid_df)

    rsf = RandomSurvivalForest(**RSF_PARAMS_SAFE).fit(Xtr, ytr)
    gb  = GradientBoostingSurvivalAnalysis(**GBSA_TUNED_PARAMS).fit(Xtr, ytr)
    cx  = CoxnetSurvivalAnalysis(**COXNET_PARAMS).fit(Xtr, ytr)

    rsf_va = rsf.predict(Xva)
    gb_va  = gb.predict(Xva)
    cx_va  = cx.predict(Xva)

    s_rsf = ipcw_cindex(ytr, yva, rsf_va)
    s_gb  = ipcw_cindex(ytr, yva, gb_va)
    s_cx  = ipcw_cindex(ytr, yva, cx_va)

    ens3_rank = (rank01(rsf_va) + rank01(gb_va) + rank01(cx_va)) / 3.0
    s_ens3 = ipcw_cindex(ytr, yva, ens3_rank)

    best_w, best_s = None, -1
    for w in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        ens = w * rank01(gb_va) + (1 - w) * (rank01(rsf_va) + rank01(cx_va)) / 2.0
        sc = ipcw_cindex(ytr, yva, ens)
        if sc > best_s:
            best_s, best_w = sc, w

    return {
        "RSF": s_rsf,
        "GBSA_tuned": s_gb,
        "Coxnet": s_cx,
        "ENS3_rank": s_ens3,
        f"ENS_wGBSA_{best_w}": best_s,
    }

def main():
    train_full, feature_cols = build_train_table()

    header("3-fold CV stratified on (event x time-bin)")
    strata = make_strata(train_full, n_bins=3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_full, strata), start=1):
        tr = train_full.iloc[tr_idx].copy()
        va = train_full.iloc[va_idx].copy()

        print(f"\nFold {fold}/3 | Train={len(tr)} Valid={len(va)} "
              f"| event tr={tr[TARGET_EVENT].mean():.3f} va={va[TARGET_EVENT].mean():.3f} "
              f"| time mean tr={tr[TARGET_TIME].mean():.3f} va={va[TARGET_TIME].mean():.3f}")

        res = eval_fold(tr, va, feature_cols)
        res["fold"] = fold
        rows.append(res)

        print("  " + " | ".join([f"{k}={v:.4f}" for k,v in res.items() if k!="fold"]))

    df = pd.DataFrame(rows)

    header("CV results (per fold)")
    print(df.to_string(index=False))

    header("CV mean ± std")
    metrics = [c for c in df.columns if c != "fold"]
    out = []
    for m in metrics:
        out.append((m, df[m].mean(), df[m].std()))
    out_df = pd.DataFrame(out, columns=["metric", "mean", "std"]).sort_values("mean", ascending=False)
    print(out_df.to_string(index=False))

    out_df.to_csv("experiments_5_cv_summary.csv", index=False)
    print("\nSaved: experiments_5_cv_summary.csv")

if __name__ == "__main__":
    main()
