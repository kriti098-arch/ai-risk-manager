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

from ml.drift import compute_drift_report, psi_verdict

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

@st.cache_data
def load_holdout_test():
    """The TRUE held-out test set -- rows the model never saw during
    training or threshold tuning. This is what the Review Queue and any
    'genuinely unseen data' demo should use, NOT the full transactions.csv
    (which includes rows the model was trained on)."""
    path = BASE / "data" / "holdout_test.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_validation_set():
    """Validation set with pre-computed risk_score -- used for the live
    interactive cost-curve explorer so it doesn't need to recompute
    predictions from scratch on every slider move."""
    path = BASE / "data" / "validation_set.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

rf, iso, cfg, explainer = load_models()
FEATURES = cfg["features"]
THRESHOLD = cfg["threshold"]
ISO_RAW_LOW = cfg["iso_raw_low"]
ISO_RAW_HIGH = cfg["iso_raw_high"]
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
    # SAME fixed calibration range as train.py/main.py -- see comment there.
    return float(np.clip((raw - ISO_RAW_LOW) / (ISO_RAW_HIGH - ISO_RAW_LOW + 1e-9), 0, 1))


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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001F50E Score a Transaction", "\U0001F4CA Model Performance", "\U0001F4CB Review Queue Demo",
    "\U0001F4C9 Data Drift Monitor", "\U0001F4C1 Batch Score (CSV)", "\U0001F4B0 Cost Explorer",
])

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
    holdout = load_holdout_test()
    if holdout is not None:
        st.caption("A random sample from the **genuinely held-out test set** -- rows the model never saw during training or threshold tuning. Scored and sorted by risk. This is the human-in-the-loop layer: BLOCK/REVIEW cases land here instead of vanishing into an automated decision.")
    else:
        st.caption("⚠️ data/holdout_test.csv not found -- falling back to the full dataset (re-run `python ml/train.py` to generate a proper held-out sample).")

    n_sample = st.slider("Sample size", 10, 100, 30)
    if st.button("Pull a fresh batch"):
        df = holdout if holdout is not None else load_dataset()
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

# ---- TAB 4: Data drift monitor ------------------------------------------
DRIFT_FEATURES = [
    "amount", "amount_zscore", "velocity_1h", "velocity_24h",
    "distinct_merchants_24h", "cross_merchant_velocity_1h",
    "account_age_days", "payee_added_to_txn_minutes", "session_duration_sec",
    "merchant_risk_score",
]

with tab4:
    st.subheader("Data drift monitor")
    st.caption(
        "Fraud patterns aren't static — scammers adapt, new UPI-scam variants emerge, "
        "mule rings change their cadence. This model was trained once on a fixed dataset; "
        "if live traffic drifts away from that distribution, recall quietly degrades with "
        "no crash and no error. This panel uses Population Stability Index (PSI) to compare "
        "a live batch against the training distribution and flag when the model may need retraining."
    )

    st.markdown("""
    | PSI range | Meaning |
    |---|---|
    | < 0.10 | Stable — no significant shift |
    | 0.10 – 0.25 | Moderate shift — monitor closely |
    | ≥ 0.25 | Significant shift — investigate / retrain |
    """)

    df_full = load_dataset()
    reference = df_full.sample(min(40000, len(df_full)), random_state=1)

    c1, c2 = st.columns(2)
    pull_normal = c1.button("Simulate normal traffic (no drift)", use_container_width=True)
    pull_drift = c2.button("Simulate a fraud wave evolving (drift)", use_container_width=True)

    if pull_normal or pull_drift:
        batch = df_full.sample(3000, random_state=None).copy()
        if pull_drift:
            # simulate a new mule-ring wave that's gotten faster/quieter than
            # what the model was trained on -- the realistic "drift" story
            rng = np.random.default_rng()
            batch["payee_added_to_txn_minutes"] = batch["payee_added_to_txn_minutes"].astype(float)
            shift_idx = batch.sample(frac=0.4, random_state=None).index
            batch.loc[shift_idx, "cross_merchant_velocity_1h"] += rng.poisson(5, len(shift_idx))
            batch.loc[shift_idx, "distinct_merchants_24h"] += rng.poisson(4, len(shift_idx))
            batch.loc[shift_idx, "payee_added_to_txn_minutes"] *= 0.15
            batch["amount"] = batch["amount"] * rng.uniform(1.3, 1.6)
            st.session_state["drift_label"] = "Simulated drifted batch (fraud wave evolving)"
        else:
            st.session_state["drift_label"] = "Simulated normal batch (no drift)"

        report = compute_drift_report(reference, batch, DRIFT_FEATURES)
        st.session_state["drift_report"] = report

    if "drift_report" in st.session_state:
        report = st.session_state["drift_report"]
        st.markdown(f"#### {st.session_state['drift_label']}")

        overall_psi = report["psi"].mean()
        overall_verdict = psi_verdict(overall_psi)
        verdict_color = {"stable": "green", "moderate shift -- monitor": "orange",
                          "significant shift -- investigate/retrain": "red"}[overall_verdict]
        st.markdown(f"**Overall verdict:** :{verdict_color}[{overall_verdict}]  (mean PSI = {overall_psi:.3f})")

        fig4 = px.bar(
            report, x="psi", y="feature", orientation="h", color="psi",
            color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"], range_color=[0, 0.5],
            labels={"psi": "PSI (higher = more drift)", "feature": ""},
        )
        fig4.add_vline(x=0.10, line_dash="dot", line_color="orange")
        fig4.add_vline(x=0.25, line_dash="dot", line_color="red")
        fig4.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

        drifted_features = report[report["psi"] >= 0.10]["feature"].tolist()
        if drifted_features:
            st.warning(f"Features showing drift: **{', '.join(drifted_features)}**. "
                       f"If this persists across multiple live batches (not just one noisy sample), "
                       f"it's a signal to retrain on recent data rather than trusting the current model indefinitely.")
        else:
            st.success("No features show meaningful drift — the model's training distribution still matches live traffic.")
    else:
        st.info("Click one of the buttons above to pull a batch and check for drift.")

# ---- TAB 5: Batch score via CSV upload -----------------------------------
with tab5:
    st.subheader("Batch score your own transactions")
    st.caption(
        "Upload a CSV of transactions and get all of them scored at once. "
        "This is the honest test of whether the system generalizes -- not just "
        "the 4 built-in presets, but whatever data you bring."
    )

    template_df = pd.DataFrame([PRESETS["Legit transaction"]])
    st.download_button(
        "Download a template CSV (with 1 example row)",
        data=template_df.to_csv(index=False),
        file_name="transaction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    MAX_BATCH_ROWS = 2000

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            batch_df = None

        if batch_df is not None:
            missing = [f for f in FEATURES if f not in batch_df.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}. "
                         f"Download the template above to see the exact format needed.")
            else:
                if len(batch_df) > MAX_BATCH_ROWS:
                    st.warning(f"File has {len(batch_df)} rows — scoring only the first {MAX_BATCH_ROWS} to keep this responsive.")
                    batch_df = batch_df.head(MAX_BATCH_ROWS)

                with st.spinner(f"Scoring {len(batch_df)} transactions..."):
                    X_batch = batch_df[FEATURES].copy()
                    rf_p = rf.predict_proba(X_batch)[:, 1]
                    iso_raw = -iso.score_samples(X_batch)
                    iso_p = np.clip((iso_raw - ISO_RAW_LOW) / (ISO_RAW_HIGH - ISO_RAW_LOW + 1e-9), 0, 1)
                    risk_scores = 0.7 * rf_p + 0.3 * iso_p

                    decisions = np.where(
                        risk_scores >= THRESHOLD, "BLOCK",
                        np.where(risk_scores >= THRESHOLD - REVIEW_BAND, "REVIEW", "ALLOW")
                    )

                    # SHAP for the whole batch in one call -- much faster than
                    # per-row explainer calls, and TreeExplainer supports this directly
                    sv = explainer.shap_values(X_batch)
                    sv_all = sv[1] if isinstance(sv, list) else sv[:, :, 1]

                    top_reasons_list = []
                    pattern_list = []
                    for i in range(len(X_batch)):
                        row_reasons = sorted(
                            [{"feature": f, "impact": round(float(v), 4)} for f, v in zip(FEATURES, sv_all[i])],
                            key=lambda r: -abs(r["impact"])
                        )
                        top_reasons_list.append(row_reasons[0]["feature"] if row_reasons else "")
                        pattern_list.append(guess_pattern(row_reasons[:8]))

                    out_df = batch_df.copy()
                    out_df["risk_score"] = np.round(risk_scores, 4)
                    out_df["decision"] = decisions
                    out_df["top_reason"] = top_reasons_list
                    out_df["likely_pattern"] = pattern_list
                    out_df = out_df.sort_values("risk_score", ascending=False)

                st.success(f"Scored {len(out_df)} transactions.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Flagged BLOCK", int((out_df["decision"] == "BLOCK").sum()))
                c2.metric("Flagged REVIEW", int((out_df["decision"] == "REVIEW").sum()))
                c3.metric("ALLOW", int((out_df["decision"] == "ALLOW").sum()))

                def highlight_batch(row):
                    color = {"BLOCK": "background-color: #fdedec", "REVIEW": "background-color: #fef5e7", "ALLOW": "background-color: #eafaf1"}[row["decision"]]
                    return [color] * len(row)
                st.dataframe(out_df.style.apply(highlight_batch, axis=1), use_container_width=True, height=450)

                st.download_button(
                    "Download scored results as CSV",
                    data=out_df.to_csv(index=False),
                    file_name="scored_transactions.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a CSV to score it, or try the template above as a starting point.")

# ---- TAB 6: Interactive cost explorer -------------------------------------
with tab6:
    st.subheader("Cost-assumption explorer")
    st.caption(
        "The threshold used everywhere else in this app was tuned against ONE set of cost "
        "assumptions (₹120 per false positive, 1.15× transaction amount per missed fraud). "
        "Real merchants have different economics. Drag the sliders below to see how the "
        "optimal threshold shifts for different assumptions — computed live on the "
        "validation set, not pre-baked."
    )

    val_set = load_validation_set()
    if val_set is None:
        st.warning("⚠️ data/validation_set.csv not found — re-run `python ml/train.py` to generate it.")
    else:
        c1, c2 = st.columns(2)
        fn_multiplier = c1.slider(
            "False-negative cost multiplier (× transaction amount)", 1.0, 3.0, 1.15, 0.05,
            help="Cost of missing a fraud, as a multiple of the transaction amount. Higher = fraud losses hurt more relative to false alarms."
        )
        fp_flat = c2.slider(
            "False-positive flat cost (₹)", 20, 500, 120, 10,
            help="Cost of wrongly blocking a genuine transaction — support cost, lost margin, customer annoyance."
        )

        y_true = val_set["is_fraud"].values
        amounts = val_set["amount"].values
        scores = val_set["risk_score"].values

        thresholds = np.linspace(0.01, 0.95, 190)
        costs, fns, fps = [], [], []
        for t in thresholds:
            pred = (scores >= t).astype(int)
            fn_mask = (y_true == 1) & (pred == 0)
            fp_mask = (y_true == 0) & (pred == 1)
            cost = (amounts[fn_mask] * fn_multiplier).sum() + fp_mask.sum() * fp_flat
            costs.append(cost)
            fns.append(fn_mask.sum())
            fps.append(fp_mask.sum())

        costs = np.array(costs)
        best_idx = int(np.argmin(costs))
        best_t = thresholds[best_idx]
        best_cost = costs[best_idx]

        naive_idx = int(np.argmin(np.abs(thresholds - 0.5)))
        naive_cost = costs[naive_idx]

        c1, c2, c3 = st.columns(3)
        c1.metric("Your optimal threshold", f"{best_t:.3f}")
        c2.metric("Cost at your optimal threshold", f"₹{best_cost:,.0f}")
        c3.metric("Cost at naive 0.5 threshold", f"₹{naive_cost:,.0f}",
                  delta=f"-₹{naive_cost - best_cost:,.0f}" if naive_cost > best_cost else f"+₹{best_cost - naive_cost:,.0f}",
                  delta_color="inverse")

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines", line=dict(color="#c0392b", width=2), name="Cost"))
        fig5.add_vline(x=best_t, line_dash="dash", line_color="#2c3e50", annotation_text="your optimal")
        fig5.add_vline(x=0.5, line_dash="dot", line_color="#7f8c8d", annotation_text="naive 0.5")
        fig5.update_layout(
            xaxis_title="Decision threshold", yaxis_title="Estimated cost (₹) on validation set",
            height=380, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.caption(
            f"At your chosen cost assumptions, the optimal threshold is **{best_t:.3f}** "
            f"({int(fns[best_idx])} missed fraud, {int(fps[best_idx])} false positives on this validation set) — "
            f"try pushing the false-negative multiplier higher to see the optimal threshold drop "
            f"(the system gets more aggressive about blocking when missed fraud is assumed to hurt more)."
        )