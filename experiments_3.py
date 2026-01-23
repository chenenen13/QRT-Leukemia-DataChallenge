# ============================================================
# experiments_4.py
# Best text params + Coxnet + rank-ensembles + optional submission
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.config import (
    RANDOM_STATE, ID_COL, TARGET_EVENT, TARGET_TIME,
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

BEST_TFIDF_MAX = 5000
BEST_SVD_K = 120

def rank01(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    if len(r) <= 1:
        return np.zeros_like(r, dtype=float)
    return (r - 1.0) / (len(r) - 1.0)

def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def make_preprocessor(feature_cols, tfidf_max_features=BEST_TFIDF_MAX, svd_components=BEST_SVD_K):
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None
    num_cols = [c for c in feature_cols if c not in cat_cols and c != text_col]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
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
                    max_features=tfidf_max_features,
                    analyzer="char",
                    ngram_range=(3, 6),
                    lowercase=True,
                )),
                ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
            ]),
            [text_col]
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def align_feature_cols(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    te = test_df.copy()
    for c in feature_cols:
        if c not in te.columns:
            te[c] = 0
    return te[[ID_COL] + feature_cols]

def main(make_submission: bool = True):
    print_header("Load data")
    clinical_train, clinical_test, molecular_train, molecular_test, y_train = load_all_data()

    # CYTO regex features
    cyto_train_feat = extract_cytogenetics_features(clinical_train[CLINICAL_TEXT_COL])
    cyto_test_feat  = extract_cytogenetics_features(clinical_test[CLINICAL_TEXT_COL])
    clinical_train_aug = pd.concat([clinical_train.reset_index(drop=True), cyto_train_feat], axis=1)
    clinical_test_aug  = pd.concat([clinical_test.reset_index(drop=True),  cyto_test_feat], axis=1)

    # Molecular features (no leakage)
    gene_list, effect_list = get_top_lists(molecular_train, top_genes=TOP_GENES, top_effects=TOP_EFFECTS)
    mol_feat_train = build_molecular_features(molecular_train, gene_list=gene_list, effect_list=effect_list, binary_indicators=True)
    mol_feat_test  = build_molecular_features(molecular_test,  gene_list=gene_list, effect_list=effect_list, binary_indicators=True)

    # Merge
    X_train_full = merge_train_data(clinical_train_aug, mol_feat_train, y_train)
    train_full = clean_target(X_train_full)
    feature_cols = get_feature_columns(train_full)

    X_test_full = clinical_test_aug.merge(mol_feat_test, on=ID_COL, how="left").fillna(0)
    X_test_full = align_feature_cols(train_full, X_test_full, feature_cols)

    # Fixed split
    train_idx, valid_idx = train_test_split(
        train_full.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_full[TARGET_EVENT],
    )
    train_df = train_full.loc[train_idx].copy()
    valid_df = train_full.loc[valid_idx].copy()

    ytr_s = to_sksurv_y(train_df)
    yva_s = to_sksurv_y(valid_df)

    print_header(f"Preprocess (TFIDF={BEST_TFIDF_MAX}, SVD={BEST_SVD_K})")
    pre = make_preprocessor(feature_cols, BEST_TFIDF_MAX, BEST_SVD_K)
    Xtr = pre.fit_transform(train_df[feature_cols])
    Xva = pre.transform(valid_df[feature_cols])

    print(f"Xtr: {Xtr.shape} | Xva: {Xva.shape}")

    print_header("Train/evaluate")
    results = {}

    # RSF
    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=10,
        min_samples_split=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rsf.fit(Xtr, ytr_s)
    rsf_risk = rsf.predict(Xva)
    results["RSF"] = ipcw_cindex(ytr_s, yva_s, rsf_risk)
    print(f"  RSF -> {results['RSF']:.4f}")

    # GBSA tuned
    gbsa_t = GradientBoostingSurvivalAnalysis(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    gbsa_t.fit(Xtr, ytr_s)
    gbsa_t_risk = gbsa_t.predict(Xva)
    results["GBSA_tuned"] = ipcw_cindex(ytr_s, yva_s, gbsa_t_risk)
    print(f"  GBSA_tuned -> {results['GBSA_tuned']:.4f}")

    # Coxnet (ElasticNet Cox)
    # l1_ratio: 0.1..0.9 ; alphas auto
    best_cx = (-1, None)
    best_cx_model = None
    best_cx_risk = None

    for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cx = CoxnetSurvivalAnalysis(l1_ratio=l1, alpha_min_ratio=0.01, n_alphas=100)
        cx.fit(Xtr, ytr_s)

        # Coxnet gives coef path; predict uses last alpha by default.
        risk = cx.predict(Xva)
        c = ipcw_cindex(ytr_s, yva_s, risk)
        print(f"  Coxnet l1_ratio={l1:.1f} -> {c:.4f}")
        if c > best_cx[0]:
            best_cx = (c, l1)
            best_cx_model = cx
            best_cx_risk = risk

    results["Coxnet_best"] = best_cx[0]
    print(f"\n  Best Coxnet: {best_cx[0]:.4f} at l1_ratio={best_cx[1]}")

    # Ensembles (rank)
    ens3 = (rank01(rsf_risk) + rank01(gbsa_t_risk) + rank01(best_cx_risk)) / 3.0
    results["ENS_RSF_GBSA_Coxnet_rank"] = ipcw_cindex(ytr_s, yva_s, ens3)
    print(f"  ENS (RSF+GBSA_tuned+Coxnet rank avg) -> {results['ENS_RSF_GBSA_Coxnet_rank']:.4f}")

    print_header("Summary")
    print(pd.Series(results).sort_values(ascending=False).to_string())

    # Optional: fit on full train & produce submission
    if make_submission:
        print_header("Fit on FULL train & generate submission candidates")
        # preprocess on full train
        pre_full = make_preprocessor(feature_cols, BEST_TFIDF_MAX, BEST_SVD_K)
        X_full = pre_full.fit_transform(train_full[feature_cols])
        X_test = pre_full.transform(X_test_full[feature_cols])

        y_full = to_sksurv_y(train_full)

        # Fit models on full
        rsf_full = RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=10, min_samples_split=10,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_full, y_full)
        gbsa_full = GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.1, max_depth=3,
            min_samples_split=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, verbose=0
        ).fit(X_full, y_full)
        cx_full = CoxnetSurvivalAnalysis(
            l1_ratio=float(best_cx[1]), alpha_min_ratio=0.01, n_alphas=100
        ).fit(X_full, y_full)

        rsf_test = rsf_full.predict(X_test)
        gbsa_test = gbsa_full.predict(X_test)
        cx_test = cx_full.predict(X_test)

        ens_test = (rank01(rsf_test) + rank01(gbsa_test) + rank01(cx_test)) / 3.0

        sub = pd.DataFrame({ID_COL: X_test_full[ID_COL].values, "risk_score": ens_test})
        out_path = "submission_experiments_4.csv"
        sub.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main(make_submission=True)
