from pydantic import BaseModel, Field


class Transaction(BaseModel):
    amount: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    account_age_days: int = Field(..., ge=0)
    avg_historical_ticket: float = Field(..., gt=0)
    amount_zscore: float
    velocity_1h: int = Field(..., ge=0)
    velocity_24h: int = Field(..., ge=0)
    distinct_merchants_24h: int = Field(..., ge=0, description="Distinct merchants this account/device touched in 24h")
    cross_merchant_velocity_1h: int = Field(..., ge=0, description="Txns by this device/payee/account across DIFFERENT merchants in the last hour -- the network-effect signal")
    geo_mismatch: int = Field(..., ge=0, le=1)
    new_device: int = Field(..., ge=0, le=1)
    new_payee: int = Field(..., ge=0, le=1)
    payee_added_to_txn_minutes: float = Field(..., ge=0, description="Minutes between payee being added and this transaction -- low value + new_payee is the UPI collect-scam signature")
    sim_or_device_change_recent: int = Field(..., ge=0, le=1, description="SIM or device change flagged recently -- account-takeover signal")
    session_duration_sec: float = Field(..., gt=0, description="Longer sessions can indicate a remote-access/screen-share-driven fraud session")
    is_international: int = Field(..., ge=0, le=1)
    chargeback_history_count: int = Field(..., ge=0)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    billing_shipping_mismatch: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 4899.0,
                "hour_of_day": 2,
                "account_age_days": 3,
                "avg_historical_ticket": 900.0,
                "amount_zscore": 3.2,
                "velocity_1h": 5,
                "velocity_24h": 14,
                "distinct_merchants_24h": 6,
                "cross_merchant_velocity_1h": 4,
                "geo_mismatch": 1,
                "new_device": 1,
                "new_payee": 1,
                "payee_added_to_txn_minutes": 4.0,
                "sim_or_device_change_recent": 1,
                "session_duration_sec": 45.0,
                "is_international": 0,
                "chargeback_history_count": 1,
                "merchant_risk_score": 0.4,
                "billing_shipping_mismatch": 1,
            }
        }


class ScoreResponse(BaseModel):
    risk_score: float
    decision: str  # "allow" | "review" | "block"
    threshold: float
    top_reasons: list[dict]
    likely_pattern_hint: str  # heuristic label based on which reasons dominate, not a hard classifier
