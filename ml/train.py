"""
Train the Fraud-Spike Detector.

Architecture (deliberately mirrors ATDS):
  IsolationForest  -> unsupervised anomaly score (catches NOVEL fraud patterns
                       not in training labels -- the "zero-day" case)
  RandomForest     -> supervised classifier (catches KNOWN fraud patterns
                       with high precision)
  Blended score    -> weighted average, calibrated against a business cost
                       matrix rather than a naive 0.5 threshold.

Split: 70% train / 15% validation (threshold tuning) / 15% held-out test
(final reported numbers -- untouched until the very end).
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # project root, works on Windows/Mac/Linux
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, average_precision_score,
)

FEATURES = [
    "amount", "hour_of_day", "account_age_days", "avg_historical_ticket",
    "amount_zscore", "velocity_1h", "velocity_24h", "distinct_merchants_24h",
    "cross_merchant_velocity_1h", "geo_mismatch", "new_device", "new_payee",
    "payee_added_to_txn_minutes", "sim_or_device_change_recent",
    "session_duration_sec", "is_international", "chargeback_history_count",
    "merchant_risk_score", "billing_shipping_mismatch",
]

# ---- Business cost matrix (₹) ----------------------------------------------
# Missing a fraud (false negative): merchant eats the full transaction loss
# via chargeback + chargeback fee + reputational/processing penalty.
# Wrongly blocking a good customer (false positive): lost margin on that
# order + support-ticket cost + a slice of expected future lifetime value
# lost to churn. FN is modeled as scaling with transaction amount; FP is
# modeled as a flatter cost since most blocked orders are modest tickets
# and the customer can usually retry.
COST_FN_MULTIPLIER = 1.15   # fraud amount * this (chargeback fee + admin overhead)
COST_FP_FLAT = 120          # ~ support cost + est. margin/goodwill hit per wrongly-blocked order

FRAUD_PATTERNS_IN_DATA = [
    "upi_collect_scam", "remote_access_session",
    "account_takeover_simswap", "mule_cashout_ring",
]


def cost_of_threshold(y_true, amounts, scores, threshold):
    pred = (scores >= threshold).astype(int)
    fn_mask = (y_true == 1) & (pred == 0)
    fp_mask = (y_true == 0) & (pred == 1)
    fn_cost = (amounts[fn_mask] * COST_FN_MULTIPLIER).sum()
    fp_cost = fp_mask.sum() * COST_FP_FLAT
    return fn_cost + fp_cost, fn_mask.sum(), fp_mask.sum()


def main():
    df = pd.read_csv(BASE / "data" / "transactions.csv")
    X = df[FEATURES]
    y = df["is_fraud"]
    amt = df["amount"]
    pattern = df["fraud_pattern"]

    X_train, X_temp, y_train, y_temp, amt_train, amt_temp, pat_train, pat_temp = train_test_split(
        X, y, amt, pattern, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, amt_val, amt_test, pat_val, pat_test = train_test_split(
        X_temp, y_temp, amt_temp, pat_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # --- Unsupervised anomaly baseline (trained only on presumed-legit data,
    #     same pattern as ATDS's Isolation Forest baseline on normal traffic)
    iso = IsolationForest(
        n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1
    )
    iso.fit(X_train[y_train == 0])
    # score_samples: higher = more normal. Flip & min-max scale to 0..1 risk.
    def iso_risk(Xs):
        raw = -iso.score_samples(Xs)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    # --- Supervised classifier
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    def blended_score(Xs):
        rf_p = rf.predict_proba(Xs)[:, 1]
        iso_p = iso_risk(Xs)
        return 0.7 * rf_p + 0.3 * iso_p  # RF weighted higher: it has labels, iso covers novel patterns

    # --- Tune threshold on VALIDATION set only, by minimizing business cost
    val_scores = blended_score(X_val)
    thresholds = np.linspace(0.01, 0.95, 190)
    best = min(
        (cost_of_threshold(y_val.values, amt_val.values, val_scores, t) + (t,)
         for t in thresholds),
        key=lambda r: r[0]
    )
    best_cost, best_fn, best_fp, best_threshold = best

    # naive 0.5 threshold for comparison
    naive_cost, naive_fn, naive_fp = cost_of_threshold(y_val.values, amt_val.values, val_scores, 0.5)

    # --- FINAL evaluation on untouched TEST set
    test_scores = blended_score(X_test)
    test_pred = (test_scores >= best_threshold).astype(int)

    metrics = {
        "chosen_threshold": round(float(best_threshold), 4),
        "test_precision": round(precision_score(y_test, test_pred), 4),
        "test_recall": round(recall_score(y_test, test_pred), 4),
        "test_f1": round(f1_score(y_test, test_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, test_scores), 4),
        "test_avg_precision_pr_auc": round(average_precision_score(y_test, test_scores), 4),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),  # [[TN,FP],[FN,TP]]
        "test_n": int(len(y_test)),
        "test_n_fraud": int(y_test.sum()),
        "validation_cost_at_tuned_threshold_inr": round(float(best_cost), 2),
        "validation_cost_at_naive_0.5_threshold_inr": round(float(naive_cost), 2),
        "validation_fn_fp_at_tuned": {"missed_fraud": int(best_fn), "false_positives": int(best_fp)},
        "validation_fn_fp_at_naive": {"missed_fraud": int(naive_fn), "false_positives": int(naive_fp)},
        "cost_assumptions": {
            "cost_per_missed_fraud": "transaction_amount * 1.15 (chargeback fee + admin)",
            "cost_per_false_positive_inr": COST_FP_FLAT,
        },
    }

    test_fn_cost, test_fn_count, test_fp_count = cost_of_threshold(
        y_test.values, amt_test.values, test_scores, best_threshold
    )
    metrics["test_estimated_cost_inr"] = round(float(test_fn_cost), 2)

    # --- Recall broken down by fraud modus operandi (the actual differentiator:
    #     does the model catch EVERY fraud pattern, or just the loudest one?)
    recall_by_pattern = {}
    for p in FRAUD_PATTERNS_IN_DATA:
        mask = (pat_test.values == p)
        if mask.sum() == 0:
            continue
        recall_by_pattern[p] = {
            "n_cases": int(mask.sum()),
            "recall": round(float(test_pred[mask].sum() / mask.sum()), 4),
        }
    metrics["recall_by_fraud_pattern"] = recall_by_pattern

    # --- Cost curve across a threshold sweep, for the sensitivity chart
    cost_curve = []
    for t in np.linspace(0.05, 0.9, 35):
        c, fn, fp = cost_of_threshold(y_val.values, amt_val.values, val_scores, t)
        cost_curve.append({"threshold": round(float(t), 3), "cost_inr": round(float(c), 2),
                            "missed_fraud": int(fn), "false_positives": int(fp)})
    metrics["validation_cost_curve"] = cost_curve

    print(json.dumps(metrics, indent=2))

    with open(BASE / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(rf, BASE / "models" / "rf_classifier.pkl")
    joblib.dump(iso, BASE / "models" / "iso_forest.pkl")
    with open(BASE / "models" / "threshold.json", "w") as f:
        json.dump({"threshold": best_threshold, "features": FEATURES}, f)

    return metrics


if __name__ == "__main__":
    main()
