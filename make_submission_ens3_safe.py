import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.config import (
    RANDOM_STATE, ID_COL, TARGET_EVENT, CLINICAL_CAT_COLS, CLINICAL_TEXT_COL,
    TOP_GENES, TOP_EFFECTS
)
from src.data_loader import load_all_data, merge_train_data, clean_target
from src.features import (
    build_molecular_features,
    get_feature_columns,
    extract_cytogenetics_features,
    get_top_lists,
)
from src.evaluation import to_sksurv_y

np.random.seed(RANDOM_STATE)

BEST_TFIDF_MAX = 5000
BEST_SVD_K = 120
BEST_L1_RATIO = 0.9

def rank01(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    return (r - 1.0) / (len(r) - 1.0) if len(r) > 1 else np.zeros_like(r, dtype=float)

def make_preprocessor(feature_cols):
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None
    num_cols = [c for c in feature_cols if c not in cat_cols and c != text_col]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols))

    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))

    if text_col:
        transformers.append(("txt", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_1d", FunctionTransformer(lambda x: x.ravel(), validate=False)),
            ("to_str", FunctionTransformer(lambda x: x.astype(str), validate=False)),
            ("tfidf", TfidfVectorizer(
                max_features=BEST_TFIDF_MAX,
                analyzer="char",
                ngram_range=(3, 6),
                lowercase=True,
            )),
            ("svd", TruncatedSVD(n_components=BEST_SVD_K, random_state=RANDOM_STATE)),
        ]), [text_col]))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def main():
    clinical_train, clinical_test, molecular_train, molecular_test, y_train = load_all_data()

    # CYTO regex features
    cyto_train_feat = extract_cytogenetics_features(clinical_train[CLINICAL_TEXT_COL])
    cyto_test_feat  = extract_cytogenetics_features(clinical_test[CLINICAL_TEXT_COL])
    clinical_train_aug = pd.concat([clinical_train.reset_index(drop=True), cyto_train_feat], axis=1)
    clinical_test_aug  = pd.concat([clinical_test.reset_index(drop=True),  cyto_test_feat], axis=1)

    # Molecular features (no leakage)
    gene_list, effect_list = get_top_lists(molecular_train, top_genes=TOP_GENES, top_effects=TOP_EFFECTS)
    mol_feat_train = build_molecular_features(
        molecular_train, gene_list=gene_list, effect_list=effect_list, binary_indicators=True
    )
    mol_feat_test = build_molecular_features(
        molecular_test, gene_list=gene_list, effect_list=effect_list, binary_indicators=True
    )

    # Merge train/test
    X_train_full = merge_train_data(clinical_train_aug, mol_feat_train, y_train)
    train_full = clean_target(X_train_full)
    feature_cols = get_feature_columns(train_full)

    X_test_full = clinical_test_aug.merge(mol_feat_test, on=ID_COL, how="left").fillna(0)
    for c in feature_cols:
        if c not in X_test_full.columns:
            X_test_full[c] = 0
    X_test_full = X_test_full[[ID_COL] + feature_cols]

    # Preprocess
    pre = make_preprocessor(feature_cols)
    X_full = pre.fit_transform(train_full[feature_cols])
    X_test = pre.transform(X_test_full[feature_cols])
    y_full = to_sksurv_y(train_full)

    # Fit models (SAFE RSF)
    rsf = RandomSurvivalForest(
        n_estimators=150,
        min_samples_leaf=10,
        min_samples_split=10,
        max_features="sqrt",
        max_samples=0.7,
        random_state=RANDOM_STATE,
        n_jobs=2,
    ).fit(X_full, y_full)

    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.1, max_depth=3,
        min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, verbose=0
    ).fit(X_full, y_full)

    coxnet = CoxnetSurvivalAnalysis(
        l1_ratio=float(BEST_L1_RATIO),
        alpha_min_ratio=0.01,
        n_alphas=100
    ).fit(X_full, y_full)

    rsf_test = rsf.predict(X_test)
    gb_test = gbsa.predict(X_test)
    cx_test = coxnet.predict(X_test)

    ens3 = (rank01(rsf_test) + rank01(gb_test) + rank01(cx_test)) / 3.0

    sub = pd.DataFrame({ID_COL: X_test_full[ID_COL].values, "risk_score": ens3})
    out_path = "submission_ens3_safe.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | shape={sub.shape}")

if __name__ == "__main__":
    main()
