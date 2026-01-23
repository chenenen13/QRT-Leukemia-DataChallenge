# ============================================================
# experiments_4.py
# 3-fold CV + best text params + ENS rank + simple stacking CoxPH
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

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

# ==== Frozen best params from your search
TFIDF_MAX = 5000
SVD_K = 120

# ==== Base models params (safe for codespace)
RSF_PARAMS_SAFE = dict(
    n_estimators=150,
    min_samples_leaf=10,
    min_samples_split=10,
    max_features="sqrt",
    max_samples=0.7,          # reduces RAM
    random_state=RANDOM_STATE,
    n_jobs=2                  # avoid OOM/kill
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

def build_full_training_table():
    header("Load data")
    clinical_train, _, molecular_train, _, y_train = load_all_data()

    # CYTO regex features
    header("Add CYTO structured features (regex)")
    cyto_feat = extract_cytogenetics_features(clinical_train[CLINICAL_TEXT_COL])
    clinical_train_aug = pd.concat([clinical_train.reset_index(drop=True), cyto_feat], axis=1)

    # Molecular features (NO leakage relative to test: lists from molecular_train only)
    header("Build molecular features (lists from TRAIN only)")
    gene_list, effect_list = get_top_lists(molecular_train, top_genes=TOP_GENES, top_effects=TOP_EFFECTS)
    mol_feat_train = build_molecular_features(
        molecular_train,
        gene_list=gene_list,
        effect_list=effect_list,
        binary_indicators=True
    )
    print(f"Top genes: {len(gene_list)} | top effects: {len(effect_list)}")
    print(f"Mol features train: {mol_feat_train.shape}")

    # Merge clinical + molecular + target
    header("Merge clinical + molecular + target")
    X_train_full = merge_train_data(clinical_train_aug, mol_feat_train, y_train)
    train_full = clean_target(X_train_full)
    feature_cols = get_feature_columns(train_full)

    print(f"Train full: {train_full.shape} | #features: {len(feature_cols)}")
    return train_full, feature_cols

def fit_predict_fold(train_df, valid_df, feature_cols):
    # preprocess per fold (fit on train fold only)
    pre = make_preprocessor(feature_cols)
    Xtr = pre.fit_transform(train_df[feature_cols])
    Xva = pre.transform(valid_df[feature_cols])

    ytr = to_sksurv_y(train_df)
    yva = to_sksurv_y(valid_df)

    # ---- Base models
    rsf = RandomSurvivalForest(**RSF_PARAMS_SAFE).fit(Xtr, ytr)
    gb  = GradientBoostingSurvivalAnalysis(**GBSA_TUNED_PARAMS).fit(Xtr, ytr)
    cx  = CoxnetSurvivalAnalysis(**COXNET_PARAMS).fit(Xtr, ytr)

    rsf_tr = rsf.predict(Xtr)
    gb_tr  = gb.predict(Xtr)
    cx_tr  = cx.predict(Xtr)

    rsf_va = rsf.predict(Xva)
    gb_va  = gb.predict(Xva)
    cx_va  = cx.predict(Xva)

    # ---- Scores (base)
    s_rsf = ipcw_cindex(ytr, yva, rsf_va)
    s_gb  = ipcw_cindex(ytr, yva, gb_va)
    s_cx  = ipcw_cindex(ytr, yva, cx_va)

    # ---- Rank ensembles (no tuning -> no leakage)
    ens3_rank = (rank01(rsf_va) + rank01(gb_va) + rank01(cx_va)) / 3.0
    s_ens3 = ipcw_cindex(ytr, yva, ens3_rank)

    # ---- Weighted rank ensemble (grid over weight on GBSA)
    # ens = w*gb + (1-w)*avg(rsf,cx)
    best_w, best_s = None, -1
    for w in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        ens = w * rank01(gb_va) + (1 - w) * (rank01(rsf_va) + rank01(cx_va)) / 2.0
        sc = ipcw_cindex(ytr, yva, ens)
        if sc > best_s:
            best_s, best_w = sc, w

    # ---- Simple stacking meta CoxPH (note: meta fit uses in-sample train preds)
    # This is a bit optimistic but useful to see if stacking is worth it.
    Ztr = np.vstack([rsf_tr, gb_tr, cx_tr]).T
    Zva = np.vstack([rsf_va, gb_va, cx_va]).T

    Ztr = StandardScaler().fit_transform(Ztr)
    Zva = StandardScaler().fit_transform(Zva)

    meta = CoxPHSurvivalAnalysis().fit(Ztr, ytr)
    meta_va = meta.predict(Zva)
    s_stack = ipcw_cindex(ytr, yva, meta_va)

    return {
        "RSF": s_rsf,
        "GBSA_tuned": s_gb,
        "Coxnet": s_cx,
        "ENS3_rank": s_ens3,
        f"ENS_wGBSA_{best_w}": best_s,
        "STACK_CoxPH": s_stack,
    }

def main():
    train_full, feature_cols = build_full_training_table()

    header("3-fold CV")
    y_event = train_full[TARGET_EVENT].astype(int).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_full, y_event), start=1):
        train_df = train_full.iloc[tr_idx].copy()
        valid_df = train_full.iloc[va_idx].copy()

        print(f"\nFold {fold}/3 | Train={len(train_df)} Valid={len(valid_df)} "
              f"| event rate tr={train_df[TARGET_EVENT].mean():.3f} va={valid_df[TARGET_EVENT].mean():.3f}")

        res = fit_predict_fold(train_df, valid_df, feature_cols)
        res["fold"] = fold
        fold_rows.append(res)

        # print fold summary
        print("  " + " | ".join([f"{k}={v:.4f}" for k,v in res.items() if k!="fold"]))

    df = pd.DataFrame(fold_rows)
    header("CV results (per fold)")
    print(df.to_string(index=False))

    header("CV mean ± std")
    metrics = [c for c in df.columns if c != "fold"]
    out = []
    for m in metrics:
        out.append((m, df[m].mean(), df[m].std()))
    out_df = pd.DataFrame(out, columns=["metric", "mean", "std"]).sort_values("mean", ascending=False)
    print(out_df.to_string(index=False))

    out_df.to_csv("experiments_4_cv_summary.csv", index=False)
    print("\nSaved: experiments_4_cv_summary.csv")

if __name__ == "__main__":
    main()
