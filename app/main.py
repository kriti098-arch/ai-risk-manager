"""
AI Risk Manager -- Fraud-Spike Detector API.
Defense-only: this service SCORES and EXPLAINS transactions. It has no
capability to construct, simulate, or optimize fraud/attack patterns --
only to flag and justify why a transaction looks risky.
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pathlib import Path

from app.schemas import Transaction, ScoreResponse

BASE = Path(__file__).resolve().parent.parent

app = FastAPI(title="AI Risk Manager - Fraud Spike Detector", version="0.1.0")

rf = joblib.load(BASE / "models/rf_classifier.pkl")
iso = joblib.load(BASE / "models/iso_forest.pkl")
cfg = json.loads((BASE / "models/threshold.json").read_text())
FEATURES = cfg["features"]
THRESHOLD = cfg["threshold"]
REVIEW_BAND = 0.15  # score within [threshold-band, threshold) -> manual review, not auto-block

explainer = shap.TreeExplainer(rf)


def iso_risk(row_df: pd.DataFrame) -> float:
    raw = -iso.score_samples(row_df)[0]
    # static min/max captured at train time would be more correct in prod;
    # here we clip against a reasonable observed range for the demo
    return float(np.clip((raw + 0.6) / 1.2, 0, 1))


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


def guess_pattern(reasons: list[dict]) -> str:
    """Lightweight heuristic label based on which SHAP reasons dominate a
    specific score -- NOT a separate classifier, just a human-readable gloss
    on top of the same model's explanation, so a review-queue analyst gets a
    starting hypothesis instead of a bare feature list. Checked most-specific
    signature first so a shared feature (e.g. session_duration) doesn't
    steal the match from a more distinctive combination."""
    top_features = {r["feature"] for r in reasons if r["impact"] > 0}

    if {"cross_merchant_velocity_1h", "distinct_merchants_24h"} <= top_features and "account_age_days" in top_features:
        return "possible mule/cash-out ring"
    if "sim_or_device_change_recent" in top_features and "new_payee" in top_features:
        return "possible account-takeover / SIM-swap"
    if {"new_payee", "payee_added_to_txn_minutes"} <= top_features:
        return "possible UPI collect-request scam"
    if {"cross_merchant_velocity_1h", "session_duration_sec"} & top_features and "velocity_1h" in top_features:
        return "possible remote-access/screen-share session"
    return "no single dominant pattern -- review manually"


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    row = pd.DataFrame([txn.dict()])[FEATURES]

    rf_p = float(rf.predict_proba(row)[0, 1])
    iso_p = iso_risk(row)
    risk_score = round(0.7 * rf_p + 0.3 * iso_p, 4)

    if risk_score >= THRESHOLD:
        decision = "block"
    elif risk_score >= THRESHOLD - REVIEW_BAND:
        decision = "review"
    else:
        decision = "allow"

    sv = explainer.shap_values(row)
    sv1 = sv[1][0] if isinstance(sv, list) else sv[0, :, 1]
    all_reasons = sorted(
        [{"feature": f, "value": row.iloc[0][f], "impact": round(float(v), 4)}
         for f, v in zip(FEATURES, sv1)],
        key=lambda r: -abs(r["impact"])
    )
    reasons = all_reasons[:5]

    return ScoreResponse(
        risk_score=risk_score,
        decision=decision,
        threshold=THRESHOLD,
        top_reasons=reasons,
        likely_pattern_hint=guess_pattern(all_reasons[:8]),  # wider pool so a defining-but-smaller-impact feature (e.g. sim_or_device_change_recent) isn't missed just because one other feature spiked hard
    )
