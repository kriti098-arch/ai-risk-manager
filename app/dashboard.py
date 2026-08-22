"""
AI Risk Manager -- Streamlit dashboard.

This is the actual demo surface: a transaction scorer with a live risk
gauge and plain-language SHAP explanation, a model-performance/analytics
view, and a review-queue demo showing what an analyst would actually see.

Run with:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

st.set_page_config(page_title="AI Risk Manager", page_icon="\U0001F6E1\ufe0f", layout="wide")

# ---------------------------------------------------------------- load ----
@st.cache_resource
def load_models():
    rf = joblib.load(BASE / "models" / "rf_classifier.pkl")
    iso = joblib.load(BASE / "models" / "iso_forest.pkl")
    cfg = json.loads((BASE / "models" / "threshold.json").read_text())
    explainer = shap.TreeExplainer(rf)
    return rf, iso, cfg, explainer

@st.cache_data
def load_metrics():
    return json.loads((BASE / "reports" / "metrics.json").read_text())

@st.cache_data
def load_dataset():
    return pd.read_csv(BASE / "data" / "transactions.csv")

rf, iso, cfg, explainer = load_models()
FEATURES = cfg["features"]
THRESHOLD = cfg["threshold"]
REVIEW_BAND = 0.15
metrics = load_metrics()

PRESETS = {
    "Legit transaction": dict(
        amount=850.0, hour_of_day=14, account_age_days=400, avg_historical_ticket=900.0,
        amount_zscore=-0.1, velocity_1h=0, velocity_24h=1, distinct_merchants_24h=1,
        cross_merchant_velocity_1h=0, geo_mismatch=0, new_device=0, new_payee=0,
        payee_added_to_txn_minutes=5000.0, sim_or_device_change_recent=0,
        session_duration_sec=90.0, is_international=0, chargeback_history_count=0,
        merchant_risk_score=0.1, billing_shipping_mismatch=0,
    ),
    "UPI collect-request scam": dict(
        amount=1400.0, hour_of_day=22, account_age_days=500, avg_historical_ticket=1000.0,
        amount_zscore=0.9, velocity_1h=0, velocity_24h=1, distinct_merchants_24h=1,
        cross_merchant_velocity_1h=0, geo_mismatch=0, new_device=0, new_payee=1,
        payee_added_to_txn_minutes=2.0, sim_or_device_change_recent=0,
        session_duration_sec=12.0, is_international=0, chargeback_history_count=0,
        merchant_risk_score=0.3, billing_shipping_mismatch=0,
    ),
    "Account takeover / SIM-swap": dict(
        amount=6500.0, hour_of_day=2, account_age_days=600, avg_historical_ticket=900.0,
        amount_zscore=4.0, velocity_1h=1, velocity_24h=2, distinct_merchants_24h=1,
        cross_merchant_velocity_1h=0, geo_mismatch=0, new_device=1, new_payee=1,
        payee_added_to_txn_minutes=8.0, sim_or_device_change_recent=1,
        session_duration_sec=40.0, is_international=0, chargeback_history_count=0,
        merchant_risk_score=0.2, billing_shipping_mismatch=0,
    ),
    "Mule cash-out ring": dict(
        amount=90.0, hour_of_day=13, account_age_days=5, avg_historical_ticket=300.0,
        amount_zscore=-0.5, velocity_1h=4, velocity_24h=20, distinct_merchants_24h=9,
        cross_merchant_velocity_1h=6, geo_mismatch=0, new_device=0, new_payee=1,
        payee_added_to_txn_minutes=3000.0, sim_or_device_change_recent=0,
        session_duration_sec=60.0, is_international=0, chargeback_history_count=0,
        merchant_risk_score=0.4, billing_shipping_mismatch=0,
    ),
}

FEATURE_LABELS = {
    "amount": "Amount (₹)", "hour_of_day": "Hour of day (0-23)",
    "account_age_days": "Account age (days)", "avg_historical_ticket": "Avg historical ticket (₹)",
    "amount_zscore": "Amount z-score", "velocity_1h": "Txns in last 1h",
    "velocity_24h": "Txns in last 24h", "distinct_merchants_24h": "Distinct merchants in 24h",
    "cross_merchant_velocity_1h": "Cross-merchant velocity (1h)", "geo_mismatch": "Geo mismatch",
    "new_device": "New device", "new_payee": "New payee",
    "payee_added_to_txn_minutes": "Payee added X min before txn", "sim_or_device_change_recent": "Recent SIM/device change",
    "session_duration_sec": "Session duration (sec)", "is_international": "International txn",
    "chargeback_history_count": "Chargeback history count", "merchant_risk_score": "Merchant risk score (0-1)",
    "billing_shipping_mismatch": "Billing/shipping mismatch",
}


def iso_risk(row_df):
    raw = -iso.score_samples(row_df)[0]
    return float(np.clip((raw + 0.6) / 1.2, 0, 1))


def guess_pattern(reasons):
    top = {r["feature"] for r in reasons if r["impact"] > 0}
    if {"cross_merchant_velocity_1h", "distinct_merchants_24h"} <= top and "account_age_days" in top:
        return "Possible mule / cash-out ring"
    if "sim_or_device_change_recent" in top and "new_payee" in top:
        return "Possible account-takeover / SIM-swap"
    if {"new_payee", "payee_added_to_txn_minutes"} <= top:
        return "Possible UPI collect-request scam"
    if {"cross_merchant_velocity_1h", "session_duration_sec"} & top and "velocity_1h" in top:
        return "Possible remote-access/screen-share session"
    return "No single dominant pattern -- review manually"


def score_transaction(values: dict):
    row = pd.DataFrame([values])[FEATURES]
    rf_p = float(rf.predict_proba(row)[0, 1])
    iso_p = iso_risk(row)
    risk = round(0.7 * rf_p + 0.3 * iso_p, 4)
    if risk >= THRESHOLD:
        decision = "BLOCK"
    elif risk >= THRESHOLD - REVIEW_BAND:
        decision = "REVIEW"
    else:
        decision = "ALLOW"
    sv = explainer.shap_values(row)
    sv1 = sv[1][0] if isinstance(sv, list) else sv[0, :, 1]
    all_reasons = sorted(
        [{"feature": f, "value": row.iloc[0][f], "impact": round(float(v), 4)} for f, v in zip(FEATURES, sv1)],
        key=lambda r: -abs(r["impact"])
    )
    return risk, decision, all_reasons[:5], guess_pattern(all_reasons[:8])


# ------------------------------------------------------------------- UI ---
st.title("\U0001F6E1\ufe0f AI Risk Manager")
st.caption("Fraud-spike detector for Indian digital-payment fraud -- UPI scams, SIM-swap takeover, remote-access sessions, mule rings. Defense-only: scores and explains, never simulates or optimizes attacks.")

tab1, tab2, tab3 = st.tabs(["\U0001F50E Score a Transaction", "\U0001F4CA Model Performance", "\U0001F4CB Review Queue Demo"])

# ---- TAB 1: Live scorer ----------------------------------------------
with tab1:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Transaction details")
        preset_name = st.selectbox("Load an example", list(PRESETS.keys()))
        preset = PRESETS[preset_name]

        with st.form("txn_form"):
            c1, c2 = st.columns(2)
            vals = {}
            with c1:
                vals["amount"] = st.number_input(FEATURE_LABELS["amount"], value=preset["amount"], min_value=1.0)
                vals["hour_of_day"] = st.slider(FEATURE_LABELS["hour_of_day"], 0, 23, preset["hour_of_day"])
                vals["account_age_days"] = st.number_input(FEATURE_LABELS["account_age_days"], value=float(preset["account_age_days"]), min_value=0.0)
                vals["avg_historical_ticket"] = st.number_input(FEATURE_LABELS["avg_historical_ticket"], value=preset["avg_historical_ticket"], min_value=1.0)
                vals["amount_zscore"] = st.number_input(FEATURE_LABELS["amount_zscore"], value=preset["amount_zscore"])
                vals["velocity_1h"] = st.number_input(FEATURE_LABELS["velocity_1h"], value=float(preset["velocity_1h"]), min_value=0.0)
                vals["velocity_24h"] = st.number_input(FEATURE_LABELS["velocity_24h"], value=float(preset["velocity_24h"]), min_value=0.0)
                vals["distinct_merchants_24h"] = st.number_input(FEATURE_LABELS["distinct_merchants_24h"], value=float(preset["distinct_merchants_24h"]), min_value=0.0)
                vals["cross_merchant_velocity_1h"] = st.number_input(FEATURE_LABELS["cross_merchant_velocity_1h"], value=float(preset["cross_merchant_velocity_1h"]), min_value=0.0)
                vals["payee_added_to_txn_minutes"] = st.number_input(FEATURE_LABELS["payee_added_to_txn_minutes"], value=preset["payee_added_to_txn_minutes"], min_value=0.0)
            with c2:
                vals["session_duration_sec"] = st.number_input(FEATURE_LABELS["session_duration_sec"], value=preset["session_duration_sec"], min_value=1.0)
                vals["chargeback_history_count"] = st.number_input(FEATURE_LABELS["chargeback_history_count"], value=float(preset["chargeback_history_count"]), min_value=0.0)
                vals["merchant_risk_score"] = st.slider(FEATURE_LABELS["merchant_risk_score"], 0.0, 1.0, preset["merchant_risk_score"])
                vals["geo_mismatch"] = int(st.checkbox(FEATURE_LABELS["geo_mismatch"], value=bool(preset["geo_mismatch"])))
                vals["new_device"] = int(st.checkbox(FEATURE_LABELS["new_device"], value=bool(preset["new_device"])))
                vals["new_payee"] = int(st.checkbox(FEATURE_LABELS["new_payee"], value=bool(preset["new_payee"])))
                vals["sim_or_device_change_recent"] = int(st.checkbox(FEATURE_LABELS["sim_or_device_change_recent"], value=bool(preset["sim_or_device_change_recent"])))
                vals["is_international"] = int(st.checkbox(FEATURE_LABELS["is_international"], value=bool(preset["is_international"])))
                vals["billing_shipping_mismatch"] = int(st.checkbox(FEATURE_LABELS["billing_shipping_mismatch"], value=bool(preset["billing_shipping_mismatch"])))

            submitted = st.form_submit_button("Score this transaction", type="primary", use_container_width=True)

    with right:
        st.subheader("Risk assessment")
        if submitted:
            risk, decision, reasons, pattern = score_transaction(vals)

            color = {"BLOCK": "#e74c3c", "REVIEW": "#f39c12", "ALLOW": "#27ae60"}[decision]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, (THRESHOLD - REVIEW_BAND) * 100], "color": "#eafaf1"},
                        {"range": [(THRESHOLD - REVIEW_BAND) * 100, THRESHOLD * 100], "color": "#fef5e7"},
                        {"range": [THRESHOLD * 100, 100], "color": "#fdedec"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": THRESHOLD * 100},
                },
                title={"text": "Risk score"},
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### Decision: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
            st.markdown(f"**Likely pattern:** {pattern}")

            st.markdown("#### Why this score -- top contributing factors")
            reason_df = pd.DataFrame(reasons)
            reason_df["label"] = reason_df["feature"].map(FEATURE_LABELS)
            reason_df["direction"] = np.where(reason_df["impact"] > 0, "Pushes risk UP", "Pushes risk DOWN")
            fig2 = px.bar(
                reason_df.sort_values("impact"), x="impact", y="label", color="direction",
                color_discrete_map={"Pushes risk UP": "#e74c3c", "Pushes risk DOWN": "#27ae60"},
                orientation="h", labels={"impact": "SHAP impact on risk score", "label": ""},
            )
            fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

            if decision == "BLOCK":
                st.error("This transaction would be **blocked** and routed for manual/step-up verification, not silently declined without recourse.")
            elif decision == "REVIEW":
                st.warning("This transaction would be sent to a **human review queue** -- not auto-blocked, not auto-allowed.")
            else:
                st.success("This transaction would be **allowed** to proceed normally.")
        else:
            st.info("Pick a preset or fill in details on the left, then click **Score this transaction**.")

# ---- TAB 2: Model performance ------------------------------------------
with tab2:
    st.subheader("Held-out test set performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['test_precision']*100:.1f}%")
    c2.metric("Recall", f"{metrics['test_recall']*100:.1f}%")
    c3.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
    c4.metric("F1", f"{metrics['test_f1']:.3f}")

    st.markdown("#### Recall by fraud pattern -- the honest breakdown")
    rbp = metrics["recall_by_fraud_pattern"]
    rbp_df = pd.DataFrame([{"pattern": k.replace("_", " ").title(), "recall": v["recall"], "cases": v["n_cases"]} for k, v in rbp.items()])
    fig3 = px.bar(rbp_df.sort_values("recall"), x="recall", y="pattern", orientation="h",
                  text=rbp_df.sort_values("recall")["recall"].map(lambda x: f"{x*100:.1f}%"),
                  color="recall", color_continuous_scale="RdYlGn", range_color=[0.5, 1.0])
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                        xaxis_title="Recall", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("UPI collect-request scams are the hardest to catch -- no velocity spike, no device change, just a victim-approved payment to a fresh payee. Mule rings are caught almost every time because cross-merchant velocity is hard to hide.")

    st.markdown("#### Cost-aware threshold tuning")
    cc1, cc2 = st.columns(2)
    cc1.metric("Cost at naive 0.5 threshold", f"₹{metrics['validation_cost_at_naive_0.5_threshold_inr']:,.0f}")
    cc2.metric("Cost at tuned threshold", f"₹{metrics['validation_cost_at_tuned_threshold_inr']:,.0f}",
               delta=f"-₹{metrics['validation_cost_at_naive_0.5_threshold_inr'] - metrics['validation_cost_at_tuned_threshold_inr']:,.0f}",
               delta_color="inverse")

    cost_curve_path = BASE / "reports" / "cost_curve.png"
    if cost_curve_path.exists():
        st.image(str(cost_curve_path), caption="Cost vs. decision threshold on the validation set")

    shap_path = BASE / "reports" / "shap_summary.png"
    if shap_path.exists():
        st.markdown("#### Global feature importance (SHAP)")
        st.image(str(shap_path))

# ---- TAB 3: Review queue demo ------------------------------------------
with tab3:
    st.subheader("Review queue -- what an analyst would actually see")
    st.caption("A random sample of held-out transactions, scored and sorted by risk. This is the human-in-the-loop layer: BLOCK/REVIEW cases land here instead of vanishing into an automated decision.")

    n_sample = st.slider("Sample size", 10, 100, 30)
    if st.button("Pull a fresh batch"):
        df = load_dataset()
        sample = df.sample(n_sample, random_state=None).reset_index(drop=True)
        results = []
        for _, row in sample.iterrows():
            vals = {f: row[f] for f in FEATURES}
            risk, decision, reasons, pattern = score_transaction(vals)
            results.append({
                "amount": row["amount"], "hour": row["hour_of_day"], "risk_score": risk,
                "decision": decision, "likely_pattern": pattern,
                "actually_fraud": "YES" if row["is_fraud"] == 1 else "no",
                "top_reason": reasons[0]["feature"] if reasons else "",
            })
        res_df = pd.DataFrame(results).sort_values("risk_score", ascending=False)
        st.session_state["queue"] = res_df

    if "queue" in st.session_state:
        res_df = st.session_state["queue"]
        def highlight(row):
            color = {"BLOCK": "background-color: #fdedec", "REVIEW": "background-color: #fef5e7", "ALLOW": "background-color: #eafaf1"}[row["decision"]]
            return [color] * len(row)
        st.dataframe(res_df.style.apply(highlight, axis=1), use_container_width=True, height=500)

        n_flagged = (res_df["decision"] != "ALLOW").sum()
        n_caught = ((res_df["decision"] != "ALLOW") & (res_df["actually_fraud"] == "YES")).sum()
        n_missed = ((res_df["decision"] == "ALLOW") & (res_df["actually_fraud"] == "YES")).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sent to review/block queue", int(n_flagged))
        c2.metric("Actual fraud caught", int(n_caught))
        c3.metric("Actual fraud missed", int(n_missed))
    else:
        st.info("Click **Pull a fresh batch** to see the review queue in action.")
