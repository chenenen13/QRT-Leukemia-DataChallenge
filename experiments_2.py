# ============================================================
# experiments_3.py
# Sweep TF-IDF/SVD + CENTER target encoding (sur split fixe)
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


# -----------------------
# Utils
# -----------------------
def rank01(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    if len(r) <= 1:
        return np.zeros_like(r, dtype=float)
    return (r - 1.0) / (len(r) - 1.0)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def make_preprocessor(feature_cols, tfidf_max_features, svd_components):
    """
    Builds the same logic as get_default_preprocessor, but parameterized.
    Numeric: all except cat + text
    Cat: CENTER
    Text: CYTOGENETICS -> char TFIDF + SVD
    """
    cat_cols = [c for c in CLINICAL_CAT_COLS if c in feature_cols]
    text_col = CLINICAL_TEXT_COL if CLINICAL_TEXT_COL in feature_cols else None
    num_cols = [c for c in feature_cols if c not in cat_cols and c != text_col]

    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if text_col:
        text_pipeline = Pipeline([
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
        ])
        transformers.append(("txt", text_pipeline, [text_col]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def center_target_encode(train_df, valid_df, col="CENTER"):
    """
    Target-encode CENTER using ONLY train_df (no leakage).
    We use mean(OS_STATUS) as a simple proxy.
    """
    global_mean = train_df[TARGET_EVENT].mean()
    m = train_df.groupby(col)[TARGET_EVENT].mean()

    tr = train_df.copy()
    va = valid_df.copy()

    tr["CENTER_TE"] = tr[col].map(m).fillna(global_mean).astype(float)
    va["CENTER_TE"] = va[col].map(m).fillna(global_mean).astype(float)
    return tr, va


def fit_eval_all(Xtr, ytr, Xva, yva):
    """
    Fit RSF + GBSA + GBSA tuned and compute rank-ensemble.
    Returns dict of scores.
    """
    out = {}

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=10,
        min_samples_split=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rsf.fit(Xtr, ytr)
    rsf_risk = rsf.predict(Xva)
    out["RSF"] = ipcw_cindex(ytr, yva, rsf_risk)

    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    gbsa.fit(Xtr, ytr)
    gbsa_risk = gbsa.predict(Xva)
    out["GBSA"] = ipcw_cindex(ytr, yva, gbsa_risk)

    gbsa_t = GradientBoostingSurvivalAnalysis(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    gbsa_t.fit(Xtr, ytr)
    gbsa_t_risk = gbsa_t.predict(Xva)
    out["GBSA_tuned"] = ipcw_cindex(ytr, yva, gbsa_t_risk)

    # rank ensemble of 3
    ens3 = (rank01(rsf_risk) + rank01(gbsa_risk) + rank01(gbsa_t_risk)) / 3.0
    out["ENS_3models_rank"] = ipcw_cindex(ytr, yva, ens3)

    return out


def main():
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

    print(f"Train full: {train_full.shape} | #features: {len(feature_cols)}")

    # Fixed split
    train_idx, valid_idx = train_test_split(
        train_full.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_full[TARGET_EVENT],
    )
    base_train_df = train_full.loc[train_idx].copy()
    base_valid_df = train_full.loc[valid_idx].copy()

    ytr_s = to_sksurv_y(base_train_df)
    yva_s = to_sksurv_y(base_valid_df)

    # Sweep grid
    sweep = [
        (5000,  80),
        (5000, 120),
        (10000, 100),
        (10000, 150),
        (20000, 150),
        (20000, 200),
    ]

    rows = []

    for tfidf_max, svd_k in sweep:
        print_header(f"Config: TFIDF_MAX_FEATURES={tfidf_max} | SVD_COMPONENTS={svd_k}")

        # --------- baseline (no CENTER_TE)
        pre = make_preprocessor(feature_cols, tfidf_max, svd_k)
        Xtr = pre.fit_transform(base_train_df[feature_cols])
        Xva = pre.transform(base_valid_df[feature_cols])

        scores = fit_eval_all(Xtr, ytr_s, Xva, yva_s)
        scores["TFIDF_MAX"] = tfidf_max
        scores["SVD_K"] = svd_k
        scores["CENTER_TE"] = 0
        scores["Xdim"] = Xtr.shape[1]
        print(pd.Series(scores)[["RSF","GBSA","GBSA_tuned","ENS_3models_rank"]].to_string())
        rows.append(scores)

        # --------- with CENTER target encoding (adds numeric feature)
        tr_te, va_te = center_target_encode(base_train_df, base_valid_df, col="CENTER")
        feature_cols_te = feature_cols + ["CENTER_TE"]

        pre_te = make_preprocessor(feature_cols_te, tfidf_max, svd_k)
        Xtr_te = pre_te.fit_transform(tr_te[feature_cols_te])
        Xva_te = pre_te.transform(va_te[feature_cols_te])

        scores_te = fit_eval_all(Xtr_te, ytr_s, Xva_te, yva_s)
        scores_te["TFIDF_MAX"] = tfidf_max
        scores_te["SVD_K"] = svd_k
        scores_te["CENTER_TE"] = 1
        scores_te["Xdim"] = Xtr_te.shape[1]
        print("With CENTER_TE:")
        print(pd.Series(scores_te)[["RSF","GBSA","GBSA_tuned","ENS_3models_rank"]].to_string())
        rows.append(scores_te)

    # Summary table
    res = pd.DataFrame(rows)
    res = res.sort_values("ENS_3models_rank", ascending=False)

    print_header("Top results (by ENS_3models_rank)")
    show_cols = ["TFIDF_MAX","SVD_K","CENTER_TE","Xdim","RSF","GBSA","GBSA_tuned","ENS_3models_rank"]
    print(res[show_cols].head(15).to_string(index=False))

    out_path = "experiments_3_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
