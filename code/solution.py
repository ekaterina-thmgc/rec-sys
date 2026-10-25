import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

SEED = 42
PRIOR = 0.25
ALPHA = 10.0


def make_dataset(n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    n_users, n_items = 100, 500

    user_id = rng.integers(1, n_users + 1, n)
    item_id = rng.integers(1, n_items + 1, n)
    category_map = {
        i: rng.choice(["A", "B", "C"]) for i in range(1, n_items + 1)
    }
    item_category = [category_map[i] for i in item_id]
    timestamp = pd.date_range("2024-01-01", periods=n, freq="h")
    event_type = rng.choice(
        ["view", "click", "purchase"], n, p=[0.7, 0.2, 0.1]
    )

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "item_id": item_id,
            "item_category": item_category,
            "timestamp": timestamp,
            "event_type": event_type,
        }
    )
    df["target"] = df["event_type"].isin(["click", "purchase"]).astype(int)
    df["is_view"] = (df["event_type"] == "view").astype(int)
    df["is_click"] = (df["event_type"] == "click").astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    
    user_stats = train.groupby("user_id").agg(
        user_ctr=("target", "mean"),
        user_n=("target", "count"),
        user_views=("is_view", "sum"),
        user_clicks=("is_click", "sum"),
    ).reset_index()
    
    item_stats = train.groupby("item_id").agg(
        item_ctr=("target", "mean"),
        item_n=("target", "count"),
    ).reset_index()
    
    cat_stats = train.groupby("item_category").agg(
        cat_ctr=("target", "mean"),
    ).reset_index()
    
    user_cat_stats = train.groupby(["user_id", "item_category"]).agg(
        user_cat_ctr=("target", "mean"),
        user_cat_n=("target", "count"),
    ).reset_index()
    
    user_item_stats = train.groupby(["user_id", "item_id"]).agg(
        user_item_n=("target", "count"),
    ).reset_index()
    
    train = train.merge(user_stats, on="user_id", how="left")
    train = train.merge(item_stats, on="item_id", how="left")
    train = train.merge(cat_stats, on="item_category", how="left")
    train = train.merge(user_cat_stats, on=["user_id", "item_category"], how="left")
    train = train.merge(user_item_stats, on=["user_id", "item_id"], how="left")
    
    test = test.merge(user_stats, on="user_id", how="left")
    test = test.merge(item_stats, on="item_id", how="left")
    test = test.merge(cat_stats, on="item_category", how="left")
    test = test.merge(user_cat_stats, on=["user_id", "item_category"], how="left")
    test = test.merge(user_item_stats, on=["user_id", "item_id"], how="left")
    
    for df in [train, test]:
        df["user_ctr"] = df["user_ctr"].fillna(PRIOR)
        df["item_ctr"] = df["item_ctr"].fillna(PRIOR)
        df["cat_ctr"] = df["cat_ctr"].fillna(PRIOR)
        df["user_cat_ctr"] = df["user_cat_ctr"].fillna(PRIOR)
        df["user_item_n"] = df["user_item_n"].fillna(0)
        
        df["item_rel_pop"] = df["item_ctr"] / df["cat_ctr"].clip(lower=1e-6)
        df["user_click_view_ratio"] = (df["user_clicks"] + 1.0) / (df["user_views"] + 2.0)
        df["user_category_affinity"] = df["user_cat_ctr"] - df["cat_ctr"]
        df["repeat_interaction"] = (df["user_item_n"] > 0).astype(int)
    
    return train, test


def train_model(X, y, features, cat_features):
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5,
        cat_features=cat_features,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X[features], y)
    return model


def main():
    df = make_dataset()
    split = df["timestamp"].quantile(0.8)
    train_df = df[df["timestamp"] < split].reset_index(drop=True)
    test_df = df[df["timestamp"] >= split].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Недостаточно данных для временного разбиения.")

    train_full, test_full = build_features(train_df, test_df)

    baseline_features = [
        "user_ctr",
        "user_n",
        "item_ctr",
        "item_n",
        "item_category",
    ]
    extra_features = [
        "cat_ctr",
        "user_cat_ctr",
        "user_cat_n",
        "user_item_n",
        "item_rel_pop",
        "user_click_view_ratio",
        "user_category_affinity",
        "repeat_interaction",
    ]
    cat_features = ["item_category"]

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split_date": str(split),
    }

    model_base = train_model(
        train_full, train_df["target"], baseline_features, cat_features
    )
    p_base = model_base.predict_proba(test_full[baseline_features])[:, 1]

    results["baseline"] = {
        "roc_auc": float(roc_auc_score(test_df["target"], p_base)),
        "logloss": float(log_loss(test_df["target"], p_base)),
        "features": baseline_features,
    }

    all_features = baseline_features + extra_features
    model_llm = train_model(
        train_full, train_df["target"], all_features, cat_features
    )
    p_llm = model_llm.predict_proba(test_full[all_features])[:, 1]

    importances = model_llm.get_feature_importance()
    results["llm_augmented"] = {
        "roc_auc": float(roc_auc_score(test_df["target"], p_llm)),
        "logloss": float(log_loss(test_df["target"], p_llm)),
        "features": all_features,
        "feature_importance": {
            f: float(v)
            for f, v in sorted(
                zip(all_features, importances), key=lambda x: x[1], reverse=True
            )
        },
    }

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Split: {split}")
    print(
        f"BASELINE        ROC-AUC: {results['baseline']['roc_auc']:.4f} | "
        f"LogLoss: {results['baseline']['logloss']:.4f}"
    )
    print(
        f"LLM-AUGMENTED   ROC-AUC: {results['llm_augmented']['roc_auc']:.4f} | "
        f"LogLoss: {results['llm_augmented']['logloss']:.4f}"
    )
    print("\nFeature importance:")
    for name, value in results["llm_augmented"]["feature_importance"].items():
        print(f"  {name}: {value:.2f}")


if __name__ == "__main__":
    main()
