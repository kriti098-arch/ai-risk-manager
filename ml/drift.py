"""
Data drift monitor -- Population Stability Index (PSI).

Why this belongs here: fraud patterns aren't static. Scammers adapt,
new UPI-scam variants emerge, mule rings change their cash-out cadence.
A model trained once on a fixed dataset will silently get worse as the
live transaction distribution drifts away from what it was trained on
-- with no error, no crash, just quietly declining recall. PSI is the
standard way production ML systems catch this before it shows up as
losses: bucket a reference distribution (training data) and a current
distribution (live batch) the same way, and measure how much probability
mass moved between buckets.

Interpretation (industry-standard thresholds):
  PSI < 0.10           -> no significant shift, model is still valid
  0.10 <= PSI < 0.25    -> moderate shift, monitor closely
  PSI >= 0.25           -> significant shift, retrain / investigate

This is diagnostic, not a classifier -- it doesn't say WHICH transactions
are fraud, it says "the incoming traffic no longer looks like what this
model learned from," which is a different and complementary signal to
the risk score itself.
"""
import numpy as np
import pandas as pd


def compute_psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """PSI for a single numeric feature. Buckets are defined by reference
    quantiles so each reference bucket has ~equal mass by construction."""
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if len(reference) < buckets or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(reference.quantile(quantiles).values)
    if len(edges) < 3:
        return 0.0  # feature has almost no variance, PSI not meaningful
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def psi_verdict(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate shift -- monitor"
    return "significant shift -- investigate/retrain"


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        psi = compute_psi(reference_df[f], current_df[f])
        rows.append({"feature": f, "psi": psi, "verdict": psi_verdict(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)