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
from ml.spike_monitor import spike_z_score, spike_verdict
from app.ai_explainer import generate_ai_explanation
from app.fraud_reports import load_reports, add_report, merchant_risk_summary

st.set_page_config(page_title="AI Risk Manager", page_icon="\U0001F6E1\ufe0f", layout="wide")

# ------------------------------------------------------------ palette -----
# "Control room, not carnival" -- color carries meaning only for the three
# decision states; everything else stays neutral ink/paper/steel.
INK = "#14181B"       # base background
PAPER = "#F5F3EE"     # card/content background, primary text-on-dark
STEEL = "#7C8B90"     # secondary text, neutral lines
ALARM = "#C1442C"     # BLOCK / high risk
ALARM_TINT = "#F2E0DB"
CAUTION = "#B8862E"   # REVIEW / moderate risk
CAUTION_TINT = "#F3E9D6"
CLEAR = "#3E6350"     # ALLOW / low risk
CLEAR_TINT = "#E8EDE7"
STATUS_COLOR = {"BLOCK": ALARM, "REVIEW": CAUTION, "ALLOW": CLEAR}
STATUS_TINT = {"BLOCK": ALARM_TINT, "REVIEW": CAUTION_TINT, "ALLOW": CLEAR_TINT}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
[data-testid="stAppViewContainer"] * {{
    font-family: 'Inter', sans-serif;
}}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stTabs"] button p {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}
[data-testid="stAppViewContainer"] h1 {{ font-weight: 700 !important; }}
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {{ font-weight: 600 !important; }}
/* status badges -- used via st.markdown wherever a decision is shown inline */
.status-badge {{
    display: inline-block;
    padding: 0.2em 0.75em;
    border-radius: 4px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95em;
}}
[data-testid="stMetric"] {{
    background-color: {PAPER}0d;
    border: 1px solid {STEEL}33;
    border-radius: 6px;
    padding: 0.75rem 1rem;
}}
/* nav bar (tabs) -- distinct band + accent underline instead of Streamlit's default blue */
[data-testid="stTabs"] {{
    border-bottom: 1px solid {STEEL}40;
    margin-bottom: 1rem;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 4px;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    border-bottom-color: {CAUTION} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] p {{
    color: {CAUTION} !important;
}}
/* quieten Streamlit's default chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# custom logo mark -- a hand-drawn shield + radar-pulse SVG in the app's own
# accent color, replacing the raw OS emoji (which renders inconsistently
# and looks generic rather than designed).
# NOTE: built as ONE unbroken line with zero leading whitespace or blank
# lines -- st.markdown's HTML-block detection can silently fall back to
# rendering indented/blank-line-separated content as a literal code block
# instead of raw HTML, which is exactly what happened with a prettier,
# multi-line version of this same string.
LOGO_SVG = (
    '<svg width="42" height="42" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">'
    f'<path d="M21 2.5 L37.5 8.5 V19.5 C37.5 29 30.5 36.5 21 39.5 C11.5 36.5 4.5 29 4.5 19.5 V8.5 Z" stroke="{CAUTION}" stroke-width="1.6" fill="{INK}"/>'
    f'<circle cx="21" cy="19.5" r="7" stroke="{CAUTION}" stroke-width="1.2" fill="none" opacity="0.55"/>'
    f'<circle cx="21" cy="19.5" r="3.2" fill="{CAUTION}"/>'
    '</svg>'
)
HEADER_HTML = (
    '<div style="display:flex; align-items:center; gap:0.85rem; margin-bottom:0.1rem;">'
    + LOGO_SVG
    + '<h1 style="margin:0; padding:0;">AI Risk Manager</h1>'
    + '</div>'
)
st.markdown(HEADER_HTML, unsafe_allow_html=True)


def status_badge(decision: str) -> str:
    color = STATUS_COLOR.get(decision, STEEL)
    return f'<span class="status-badge" style="background-color:{color}22; color:{color}; border:1px solid {color}55;">{decision}</span>'

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
BASELINE_SUSPICIOUS_RATE = cfg.get("baseline_suspicious_rate", 0.125)
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
st.caption("Fraud-spike detector for Indian digital-payment fraud -- UPI scams, SIM-swap takeover, remote-access sessions, mule rings. Defense-only: scores and explains, never simulates or optimizes attacks.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "\U0001F50E Score a Transaction", "\U0001F4CA Model Performance", "\U0001F4CB Review Queue Demo",
    "\U0001F4C9 Data Drift Monitor", "\U0001F4C1 Batch Score (CSV)", "\U0001F4B0 Cost Explorer",
    "\U0001F6A9 Report Confirmed Fraud", "\U0001F4C8 Spike-Rate Monitor",
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
            # Persist into session_state -- a NEW transaction was just scored, so
            # any previous AI explanation belongs to the OLD transaction and must
            # be cleared, not shown stale against these new results.
            st.session_state["score_result"] = {
                "vals": vals, "risk": risk, "decision": decision,
                "reasons": reasons, "pattern": pattern,
            }
            st.session_state.pop("ai_explanation", None)

        if "score_result" in st.session_state:
            sr = st.session_state["score_result"]
            vals, risk, decision, reasons, pattern = sr["vals"], sr["risk"], sr["decision"], sr["reasons"], sr["pattern"]

            color = STATUS_COLOR[decision]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, (THRESHOLD - REVIEW_BAND) * 100], "color": CLEAR_TINT},
                        {"range": [(THRESHOLD - REVIEW_BAND) * 100, THRESHOLD * 100], "color": CAUTION_TINT},
                        {"range": [THRESHOLD * 100, 100], "color": ALARM_TINT},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": THRESHOLD * 100},
                },
                title={"text": "Risk score"},
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### Decision &nbsp; {status_badge(decision)}", unsafe_allow_html=True)
            st.markdown(f"**Likely pattern:** {pattern}")

            st.markdown("#### Why this score -- top contributing factors")
            reason_df = pd.DataFrame(reasons)
            reason_df["label"] = reason_df["feature"].map(FEATURE_LABELS)
            reason_df["direction"] = np.where(reason_df["impact"] > 0, "Pushes risk UP", "Pushes risk DOWN")
            fig2 = px.bar(
                reason_df.sort_values("impact"), x="impact", y="label", color="direction",
                color_discrete_map={"Pushes risk UP": ALARM, "Pushes risk DOWN": CLEAR},
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

            st.markdown("#### 🤖 AI-generated explanation")
            st.caption("The chart above is the real, exact SHAP explanation. This button asks a language model (Gemini) to translate that SAME data into a plain-English paragraph for a non-technical reader -- it doesn't re-decide anything, only explains the decision already made above.")
            if st.button("Generate AI explanation", key="ai_explain_btn"):
                try:
                    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
                except Exception:
                    # st.secrets raises (rather than returning the default) when
                    # no secrets.toml exists at all -- which is the normal state
                    # before a key has been configured, not an error condition.
                    gemini_key = ""
                with st.spinner("Asking Gemini to explain this..."):
                    explanation = generate_ai_explanation(
                        vals, risk, decision, reasons, pattern, api_key=gemini_key
                    )
                st.session_state["ai_explanation"] = explanation

            if "ai_explanation" in st.session_state:
                explanation = st.session_state["ai_explanation"]
                if explanation.startswith("⚠️"):
                    st.warning(explanation)
                else:
                    st.info(explanation)
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

        # Build a dark-compatible HTML table instead of st.dataframe which
        # brings its own light theme that clashes with the dark background.
        def row_color(decision):
            return {"BLOCK": ALARM, "REVIEW": CAUTION, "ALLOW": CLEAR}.get(decision, STEEL)

        rows_html = ""
        for _, row in res_df.iterrows():
            dc = row_color(row["decision"])
            rows_html += (
                f'<tr style="border-bottom:1px solid #ffffff15;">'
                f'<td style="padding:6px 10px; color:{dc}; font-weight:600;">{row["decision"]}</td>'
                f'<td style="padding:6px 10px;">₹{row["amount"]:,.0f}</td>'
                f'<td style="padding:6px 10px;">{row["hour"]:02.0f}:00</td>'
                f'<td style="padding:6px 10px; color:{dc};">{row["risk_score"]:.3f}</td>'
                f'<td style="padding:6px 10px; font-size:0.85em; color:#aaa;">{row["likely_pattern"]}</td>'
                f'<td style="padding:6px 10px; color:{"#e74c3c" if row["actually_fraud"]=="YES" else "#888"}; font-weight:{"600" if row["actually_fraud"]=="YES" else "400"};">{row["actually_fraud"]}</td>'
                f'</tr>'
            )

        table_html = (
            '<div style="overflow-x:auto; max-height:460px; overflow-y:auto; border:1px solid #ffffff15; border-radius:6px;">'
            '<table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif; font-size:0.9em;">'
            '<thead><tr style="background:#ffffff10; position:sticky; top:0;">'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Decision</th>'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Amount</th>'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Hour</th>'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Risk Score</th>'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Likely Pattern</th>'
            '<th style="padding:8px 10px; text-align:left; color:#aaa; font-weight:600;">Actually Fraud?</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            '</table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("")

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
            color_continuous_scale=[CLEAR, CAUTION, ALARM], range_color=[0, 0.5],
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
                    color = f"background-color: {STATUS_TINT[row['decision']]}"
                    return [color] * len(row)
                # Use st.dataframe with use_container_width for the full
                # batch table -- this one is fine as-is since it shows many
                # columns the user will want to scroll/sort; the dark theme
                # from config.toml applies here.
                st.dataframe(
                    out_df[["decision", "risk_score", "top_reason", "likely_pattern"] +
                           [c for c in out_df.columns if c not in
                            ["decision", "risk_score", "top_reason", "likely_pattern", "is_fraud", "fraud_pattern"]]]
                    .style.apply(highlight_batch, axis=1),
                    use_container_width=True, height=450
                )

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
        fig5.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines", line=dict(color=ALARM, width=2), name="Cost"))
        fig5.add_vline(x=best_t, line_dash="dash", line_color=INK, annotation_text="your optimal")
        fig5.add_vline(x=0.5, line_dash="dot", line_color=STEEL, annotation_text="naive 0.5")
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

# ---- TAB 7: Report confirmed fraud (the feedback loop) --------------------
with tab7:
    st.subheader("Report confirmed fraud")
    st.caption(
        "A risk model that only decides at the moment of a transaction has no way to learn "
        "\"we got this wrong, it turned out to be fraud after the fact\" — which is how a lot of real "
        "fraud actually gets confirmed (a customer disputes a charge days later, a bank flags a "
        "chargeback). This closes that loop: log a confirmed-fraud report against a transaction and "
        "merchant, and repeated reports automatically flag that merchant for enhanced monitoring. "
        "In a real system, accumulated reports like these are exactly what would feed periodic model retraining."
    )

    prefill = st.session_state.get("score_result")
    use_prefill = False
    if prefill:
        use_prefill = st.checkbox(
            f"Prefill from the transaction you just scored (risk={prefill['risk']:.1%}, decision={prefill['decision']})",
            value=False,
        )

    with st.form("report_fraud_form"):
        c1, c2 = st.columns(2)
        merchant_id = c1.text_input("Merchant ID / name", value="")
        transaction_amount = c2.number_input(
            "Transaction amount (₹)", min_value=0.0,
            value=float(prefill["vals"]["amount"]) if use_prefill and prefill else 0.0,
        )
        transaction_date = c1.text_input("Transaction date (optional)", value="")
        risk_score_at_time = prefill["risk"] if use_prefill and prefill else None
        decision_at_time = prefill["decision"] if use_prefill and prefill else None
        if use_prefill and prefill:
            c2.markdown(f"Risk score at the time: **{prefill['risk']:.1%}** ({prefill['decision']})")
        reason = st.text_area("Reason this is being reported as fraud", value="")
        submitted_report = st.form_submit_button("Submit report", type="primary")

    if submitted_report:
        if not merchant_id.strip() or not reason.strip():
            st.error("Merchant ID and reason are required.")
        else:
            add_report(
                BASE, merchant_id, transaction_amount, transaction_date,
                risk_score_at_time, decision_at_time, reason,
            )
            st.success("Report logged.")
            st.rerun()  # load_reports() reads fresh from disk each call (not cached), rerun just refreshes the display immediately

    st.markdown("#### All reports logged")
    reports_df = load_reports(BASE)
    if reports_df.empty:
        st.info("No fraud reports logged yet — submit one above.")
    else:
        # HTML table for dark-theme compatibility
        rpt_rows = ""
        for _, row in reports_df.sort_values("reported_at", ascending=False).iterrows():
            dc_color = {"BLOCK": ALARM, "REVIEW": CAUTION, "ALLOW": CLEAR}.get(str(row.get("decision_at_time", "")), STEEL)
            rpt_rows += (
                f'<tr style="border-bottom:1px solid #ffffff15;">'
                f'<td style="padding:5px 8px; font-family:monospace; font-size:0.8em; color:#888;">{row["report_id"]}</td>'
                f'<td style="padding:5px 8px; font-size:0.82em; color:#aaa;">{str(row["reported_at"])[:16]}</td>'
                f'<td style="padding:5px 8px; font-weight:600;">{row["merchant_id"]}</td>'
                f'<td style="padding:5px 8px;">₹{float(row["transaction_amount"]):,.0f}</td>'
                f'<td style="padding:5px 8px; color:{dc_color}; font-weight:600;">{row.get("decision_at_time","—")}</td>'
                f'<td style="padding:5px 8px; font-size:0.85em; color:#ccc;">{str(row["reason"])[:60]}{"…" if len(str(row["reason"]))>60 else ""}</td>'
                f'</tr>'
            )
        st.markdown(
            '<div style="overflow-x:auto; border:1px solid #ffffff15; border-radius:6px;">'
            '<table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif; font-size:0.88em;">'
            '<thead><tr style="background:#ffffff10;">'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">ID</th>'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">Reported at</th>'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">Merchant</th>'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">Amount</th>'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">Decision at time</th>'
            '<th style="padding:7px 8px; text-align:left; color:#aaa;">Reason</th>'
            f'</tr></thead><tbody>{rpt_rows}</tbody></table></div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Merchant risk summary")
        summary_df = merchant_risk_summary(reports_df)
        smry_rows = ""
        for _, row in summary_df.iterrows():
            flag_color = ALARM if "High risk" in str(row["flag"]) else CAUTION
            smry_rows += (
                f'<tr style="border-bottom:1px solid #ffffff15;">'
                f'<td style="padding:6px 10px; font-weight:600;">{row["merchant_id"]}</td>'
                f'<td style="padding:6px 10px;">{int(row["report_count"])}</td>'
                f'<td style="padding:6px 10px;">₹{float(row["total_reported_amount"]):,.0f}</td>'
                f'<td style="padding:6px 10px; color:{flag_color}; font-weight:600;">{row["flag"]}</td>'
                f'</tr>'
            )
        st.markdown(
            '<div style="overflow-x:auto; border:1px solid #ffffff15; border-radius:6px;">'
            '<table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif; font-size:0.9em;">'
            '<thead><tr style="background:#ffffff10;">'
            '<th style="padding:7px 10px; text-align:left; color:#aaa;">Merchant</th>'
            '<th style="padding:7px 10px; text-align:left; color:#aaa;">Reports</th>'
            '<th style="padding:7px 10px; text-align:left; color:#aaa;">Total amount</th>'
            '<th style="padding:7px 10px; text-align:left; color:#aaa;">Flag</th>'
            f'</tr></thead><tbody>{smry_rows}</tbody></table></div>',
            unsafe_allow_html=True
        )
        st.caption("Merchants with 2+ confirmed reports are automatically flagged for enhanced monitoring — this is the concrete mechanism for \"tagging a merchant\" after fraud is discovered.")

    st.markdown("---")
    st.caption(
        "⚠️ **Persistence note:** reports are saved to a CSV file (`data/fraud_reports.csv`) on disk. "
        "This works correctly for a local run or within a single live Streamlit Cloud session, but is "
        "**not durable across a cloud redeploy/reboot** — a production version of this would write to "
        "a real database. This demonstrates the workflow and concept, not a production-grade audit log."
    )

# ---- TAB 8: Spike-rate monitor ---------------------------------------------
with tab8:
    st.subheader("Spike-rate monitor")
    st.caption(
        "The Data Drift Monitor asks: do individual FEATURE distributions look different from training? "
        "This asks a different question: regardless of which features moved, is the overall RATE of "
        "transactions landing in REVIEW/BLOCK unusually high right now, compared to normal baseline traffic? "
        "A coordinated attack wave can spike this rate sharply even before there's enough volume to show up "
        "clearly as feature drift — this is the earlier, coarser warning signal."
    )
    st.markdown(f"**Baseline suspicious rate** (established on the held-out test set under normal conditions): **{BASELINE_SUSPICIOUS_RATE:.1%}**")

    df_for_spike = load_holdout_test()
    if df_for_spike is None:
        df_for_spike = load_dataset()

    window_size = st.slider("Simulated time-window size (number of transactions)", 50, 500, 200, key="spike_window_size")

    c1, c2 = st.columns(2)
    pull_normal_window = c1.button("Simulate a normal time window", use_container_width=True, key="spike_normal_btn")
    pull_attack_window = c2.button("Simulate an attack wave", use_container_width=True, key="spike_attack_btn")

    if pull_normal_window or pull_attack_window:
        window = df_for_spike.sample(window_size, random_state=None).copy()
        if pull_attack_window:
            # Inject a burst of genuinely fraudulent transactions into the window --
            # simulating a real coordinated attack wave landing in a short period,
            # not just picking a noisier-than-usual random sample.
            fraud_pool = df_for_spike[df_for_spike["is_fraud"] == 1]
            n_inject = min(len(fraud_pool), max(10, window_size // 5))
            injected = fraud_pool.sample(n_inject, random_state=None)
            window = pd.concat([window, injected], ignore_index=True)
            st.session_state["spike_label"] = f"Simulated attack wave ({n_inject} injected fraud cases added to a {window_size}-transaction window)"
        else:
            st.session_state["spike_label"] = f"Simulated normal window ({window_size} transactions, no injection)"

        X_window = window[FEATURES]
        rf_p = rf.predict_proba(X_window)[:, 1]
        iso_raw = -iso.score_samples(X_window)
        iso_p = np.clip((iso_raw - ISO_RAW_LOW) / (ISO_RAW_HIGH - ISO_RAW_LOW + 1e-9), 0, 1)
        risk_scores_window = 0.7 * rf_p + 0.3 * iso_p
        suspicious_mask = risk_scores_window >= (THRESHOLD - REVIEW_BAND)
        observed_rate = float(suspicious_mask.mean())
        n = len(window)

        z = spike_z_score(observed_rate, BASELINE_SUSPICIOUS_RATE, n)
        verdict = spike_verdict(z)

        st.session_state["spike_result"] = {
            "observed_rate": observed_rate, "n": n, "z": z, "verdict": verdict,
        }

    if "spike_result" in st.session_state:
        sr = st.session_state["spike_result"]
        st.markdown(f"#### {st.session_state['spike_label']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Observed suspicious rate", f"{sr['observed_rate']:.1%}",
                   delta=f"{(sr['observed_rate'] - BASELINE_SUSPICIOUS_RATE) / BASELINE_SUSPICIOUS_RATE:+.0%} vs baseline")
        c2.metric("Window size", sr["n"])
        c3.metric("Z-score", f"{sr['z']:.2f}")

        st.markdown(f"**Verdict:** {sr['verdict']}")

        fig6 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sr["observed_rate"] * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, max(50, sr["observed_rate"] * 120)]},
                "bar": {"color": ALARM if sr["z"] >= 3 else (CAUTION if sr["z"] >= 2 else CLEAR)},
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8,
                              "value": BASELINE_SUSPICIOUS_RATE * 100},
            },
            title={"text": "Suspicious rate (black line = baseline)"},
        ))
        fig6.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig6, use_container_width=True)

        if sr["z"] >= 2:
            st.warning("A sustained elevated rate across multiple consecutive windows (not just one) would justify treating this as a genuine attack wave rather than a one-off noisy sample — a single window is a signal to keep watching, not to act on alone.")
    else:
        st.info("Click one of the buttons above to simulate a time window and check for a fraud spike.")