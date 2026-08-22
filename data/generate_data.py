"""
Synthetic transaction dataset v2 -- India/UPI-specific fraud patterns +
cross-merchant velocity (network effect) signal.

v1 modeled generic card-present fraud (the same pattern every public
Kaggle-clone repo uses). v2 instead simulates four fraud modus operandi
that are actually dominant in Indian digital-payment fraud, per RBI/NPCI
fraud reporting themes:

  1. UPI collect-request scam    -- victim approves a fake "collect" request
                                     without reading it. Low velocity, looks
                                     like a normal one-off payment, but the
                                     payee was just added and the approval
                                     happens fast after the request.
  2. Remote-access/screen-share  -- victim installs a "support" app and
     fraud                         a scammer drives multiple transactions
                                     in one live session, often across
                                     several merchants in a short window.
  3. Account-takeover / SIM-swap -- device and/or SIM changed recently,
                                     then a high-value transaction to a
                                     brand-new payee follows quickly.
  4. Mule-account cash-out ring  -- a cluster of fresh/low-history accounts
                                     moves many small-to-medium transactions
                                     across many DIFFERENT merchants in a
                                     short window -- money layering, not a
                                     single big theft.

Cross-merchant velocity is the key new signal: v1 only ever looked at one
merchant in isolation. Real fraud/mule rings show up in the pattern of a
device/payee/account hitting MANY merchants quickly -- invisible to any
single merchant's own transaction log, but visible to a shared risk layer
(which is the pitch: an "AI Risk Manager" that Razorpay can offer because
it sees across its whole merchant base, not just one storefront).
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # project root, works on Windows/Mac/Linux

N = 60000
FRAUD_RATE = 0.028

FRAUD_PATTERNS = [
    "upi_collect_scam",
    "remote_access_session",
    "account_takeover_simswap",
    "mule_cashout_ring",
]


def generate(n=N, fraud_rate=FRAUD_RATE, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    is_fraud = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(is_fraud)

    rows = []
    for label in is_fraud:
        pattern = rng.choice(FRAUD_PATTERNS) if label == 1 else "legit"

        hour = rng.integers(0, 24)
        account_age_days = max(1, int(rng.gamma(shape=2.2, scale=190)))
        avg_ticket = max(50, rng.normal(1100, 550))
        amount = max(10, rng.normal(avg_ticket, avg_ticket * 0.4))

        # ---- base (legit-shaped) values -- these already have heavier tails
        # than v1 so fraud-like values show up in legit traffic often enough
        # that no single feature is a giveaway on its own.
        velocity_1h = max(0, int(rng.poisson(0.35)))
        velocity_24h = max(0, int(rng.poisson(2.2)))
        distinct_merchants_24h = max(1, int(rng.poisson(1.4)))
        cross_merchant_velocity_1h = max(0, int(rng.poisson(0.25)))  # genuine multi-merchant shoppers exist
        geo_mismatch = rng.random() < 0.03
        new_device = rng.random() < 0.08
        new_payee = rng.random() < 0.18
        # lognormal so legit "add payee then pay immediately" (splitting a bill,
        # paying a new vendor on the spot) overlaps with the scam's fast window
        payee_added_to_txn_minutes = rng.lognormal(mean=6.0, sigma=2.2)
        sim_or_device_change_recent = rng.random() < 0.04  # people do change phones/SIMs legitimately
        session_duration_sec = max(3, rng.lognormal(mean=4.3, sigma=0.8))
        is_international = rng.random() < 0.05
        chargeback_history = rng.poisson(0.04)
        merchant_risk_score = np.clip(rng.normal(0.15, 0.13), 0, 1)
        billing_shipping_mismatch = rng.random() < 0.03

        # ---- pattern-specific shifts, applied PROBABILISTICALLY and as
        # distribution shifts (not hard overrides) -- some fraud cases in
        # each pattern look mild/quiet, mirroring real-world label noise
        # and smarter fraudsters who stay under obvious thresholds.
        if pattern == "upi_collect_scam":
            if rng.random() < 0.8:
                new_payee = True
            payee_added_to_txn_minutes = rng.lognormal(mean=1.8, sigma=1.4)  # overlaps legit's low tail
            session_duration_sec = max(3, rng.lognormal(mean=2.6, sigma=0.9))
            amount = max(50, rng.normal(avg_ticket * 1.15, avg_ticket * 0.7))
            merchant_risk_score = np.clip(rng.normal(0.28, 0.17), 0, 1)

        elif pattern == "remote_access_session":
            velocity_1h = max(velocity_1h, int(rng.poisson(3.0)))
            distinct_merchants_24h = max(distinct_merchants_24h, int(rng.poisson(3.0)) + 1)
            cross_merchant_velocity_1h = max(cross_merchant_velocity_1h, int(rng.poisson(2.2)))
            session_duration_sec = max(session_duration_sec, rng.lognormal(mean=6.3, sigma=0.8))
            if rng.random() < 0.35:
                new_device = True
            if rng.random() < 0.55:
                new_payee = True
            hour = rng.choice([10, 11, 14, 15, 16, 19, 20, 21, 22])
            amount = max(100, rng.uniform(avg_ticket * 0.4, avg_ticket * 3.5))

        elif pattern == "account_takeover_simswap":
            if rng.random() < 0.55:
                sim_or_device_change_recent = True
            if rng.random() < 0.75:
                new_device = True
            if rng.random() < 0.7:
                new_payee = True
            payee_added_to_txn_minutes = rng.lognormal(mean=2.5, sigma=1.6)
            amount = max(300, rng.uniform(avg_ticket * 1.3, avg_ticket * 6))
            hour = rng.choice([0, 1, 2, 3, 4, 5, 22, 23] + list(range(6, 22)))  # not always odd hours
            velocity_24h = max(velocity_24h, int(rng.poisson(1.5)))
            chargeback_history = rng.poisson(0.3)

        elif pattern == "mule_cashout_ring":
            account_age_days = int(rng.gamma(1.3, 25))  # skews young, some overlap with older
            distinct_merchants_24h = max(distinct_merchants_24h, int(rng.poisson(5.5)) + 1)
            cross_merchant_velocity_1h = max(cross_merchant_velocity_1h, int(rng.poisson(3.5)))
            velocity_1h = max(velocity_1h, int(rng.poisson(2.5)))
            velocity_24h = max(velocity_24h, int(rng.poisson(14)))
            amount = rng.choice([rng.uniform(20, 200), rng.uniform(avg_ticket * 0.6, avg_ticket * 1.8)])
            merchant_risk_score = np.clip(rng.normal(0.32, 0.18), 0, 1)
            if rng.random() < 0.5:
                new_payee = True

        # ---- deliberate cross-contamination so no feature is a clean rule ----
        if label == 0 and rng.random() < 0.05:
            new_device = True
            new_payee = True
            distinct_merchants_24h = max(distinct_merchants_24h, int(rng.poisson(3)) + 1)
            cross_merchant_velocity_1h = max(cross_merchant_velocity_1h, int(rng.poisson(1.5)))
        if label == 1 and rng.random() < 0.15:
            # a genuinely quiet fraud case -- stays close to legit-shaped values
            cross_merchant_velocity_1h = max(0, int(rng.poisson(0.2)))
            sim_or_device_change_recent = rng.random() < 0.04
            velocity_1h = max(0, int(rng.poisson(0.3)))
            distinct_merchants_24h = max(1, int(rng.poisson(1.4)))

        amount_zscore = (amount - avg_ticket) / (avg_ticket * 0.4 + 1e-6)

        rows.append(dict(
            amount=round(float(amount), 2),
            hour_of_day=int(hour),
            account_age_days=int(account_age_days),
            avg_historical_ticket=round(float(avg_ticket), 2),
            amount_zscore=round(float(amount_zscore), 3),
            velocity_1h=int(velocity_1h),
            velocity_24h=int(velocity_24h),
            distinct_merchants_24h=int(distinct_merchants_24h),
            cross_merchant_velocity_1h=int(cross_merchant_velocity_1h),
            geo_mismatch=int(geo_mismatch),
            new_device=int(new_device),
            new_payee=int(new_payee),
            payee_added_to_txn_minutes=int(payee_added_to_txn_minutes),
            sim_or_device_change_recent=int(sim_or_device_change_recent),
            session_duration_sec=round(float(session_duration_sec), 1),
            is_international=int(is_international),
            chargeback_history_count=int(chargeback_history),
            merchant_risk_score=round(float(merchant_risk_score), 3),
            billing_shipping_mismatch=int(billing_shipping_mismatch),
            fraud_pattern=pattern,
            is_fraud=int(label),
        ))

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(BASE / "data" / "transactions.csv", index=False)
    print(df.shape)
    print(df["is_fraud"].value_counts(normalize=True))
    print(df[df.is_fraud == 1]["fraud_pattern"].value_counts())
