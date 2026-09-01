"""
Spike-rate monitor -- is the RATE of suspicious activity right now
higher than the established baseline?

This is a different signal from the Data Drift Monitor. Drift asks:
"do individual FEATURE distributions look different from training?"
Spike asks: "regardless of which features moved, is the overall
proportion of transactions landing in REVIEW/BLOCK unusually high
right now?" A coordinated attack wave (many accounts, many merchants,
short window) can spike this rate sharply even before enough volume
accumulates to show up clearly as feature drift.

Statistical method: a one-proportion z-test. Given a known baseline
rate p0 (the suspicious-rate on the held-out test set under normal
conditions) and an observed rate p_hat in a live window of n
transactions, the z-score:

    z = (p_hat - p0) / sqrt(p0 * (1 - p0) / n)

tells you how many standard deviations above (or below) baseline the
observed rate is, under the null hypothesis that nothing has changed.
This is the same statistical idea as z-score anomaly detectors used in
production fraud-monitoring systems, applied here to a single summary
statistic (suspicious rate) rather than per-transaction features.
"""
import math


def spike_z_score(observed_rate: float, baseline_rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    denom = math.sqrt(baseline_rate * (1 - baseline_rate) / n)
    if denom == 0:
        return 0.0
    return (observed_rate - baseline_rate) / denom


def spike_verdict(z: float) -> str:
    if z >= 3:
        return "🚨 Likely attack wave -- significant, sustained spike"
    if z >= 2:
        return "⚠️ Elevated -- monitor closely, may be an emerging spike"
    return "✅ Normal -- within expected range of baseline"