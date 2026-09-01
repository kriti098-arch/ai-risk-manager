"""
Confirmed-fraud reporting log -- the feedback loop the dashboard was
missing. A risk model that only ever scores at the moment of a
transaction has no way to learn "we got this wrong, it turned out to be
fraud after the fact" -- which is how a lot of real fraud actually gets
confirmed (a customer disputes a charge days later, a bank flags a
chargeback, a merchant reports a pattern). This module is a demo-scale
version of that feedback mechanism: anyone can log a confirmed-fraud
report against a transaction/merchant, and reports get aggregated into
a merchant risk summary -- directly answering "how do you tag a
merchant after fraud is discovered."

In a real system, this data (once accumulated) is exactly what would
feed periodic retraining -- it's the concrete answer to "where do new
labeled examples come from" that the Data Drift Monitor tab gestures at
without fully closing the loop on its own.

Persistence note (stated honestly, not hidden): reports are appended to
a CSV file on local disk. This works correctly for a local run or
within a single running Streamlit Cloud session, but is NOT durable
across a cloud redeploy/reboot -- a real production version of this
would write to an actual database. This module demonstrates the
CONCEPT and workflow, not a production-grade audit log.
"""
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORT_COLUMNS = [
    "report_id", "reported_at", "merchant_id", "transaction_amount",
    "transaction_date", "risk_score_at_time", "decision_at_time",
    "reason", "status",
]


def _reports_path(base: Path) -> Path:
    return base / "data" / "fraud_reports.csv"


def load_reports(base: Path) -> pd.DataFrame:
    path = _reports_path(base)
    if not path.exists():
        return pd.DataFrame(columns=REPORT_COLUMNS)
    return pd.read_csv(path)


def add_report(base: Path, merchant_id: str, transaction_amount: float,
               transaction_date: str, risk_score_at_time, decision_at_time: str,
               reason: str) -> pd.DataFrame:
    df = load_reports(base)
    new_row = {
        "report_id": str(uuid.uuid4())[:8],
        "reported_at": datetime.now().isoformat(timespec="seconds"),
        "merchant_id": (merchant_id or "").strip() or "unspecified",
        "transaction_amount": transaction_amount,
        "transaction_date": transaction_date or "",
        "risk_score_at_time": risk_score_at_time if risk_score_at_time not in (None, "") else "",
        "decision_at_time": decision_at_time or "",
        "reason": (reason or "").strip(),
        "status": "Reported",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    path = _reports_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def merchant_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["merchant_id", "report_count", "total_reported_amount", "flag"])
    summary = df.groupby("merchant_id").agg(
        report_count=("report_id", "count"),
        total_reported_amount=("transaction_amount", "sum"),
    ).reset_index()
    summary["flag"] = summary["report_count"].apply(
        lambda n: "🚩 High risk -- recommend enhanced monitoring" if n >= 2 else "Monitor"
    )
    return summary.sort_values("report_count", ascending=False)