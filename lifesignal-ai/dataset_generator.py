import pandas as pd
import numpy as np
import os

SEED = 42
N_SAMPLES = 10000
OUTPUT_PATH = "data/transactions_behavioral.csv"

LIFE_EVENTS = [
    "job_change",
    "marriage",
    "new_child",
    "relocation",
    "education_loan",
    "salary_hike",
    "medical_event",
    "vehicle_purchase",
    "retirement_approach",
    "travel_pattern",
    "windfall_bonus",
    "digital_shift",
]

FEATURE_COLS = [
    "monthly_spend_delta_pct",
    "salary_credit_variance",
    "emi_to_income_ratio_new",
    "merchant_category_shift_score",
    "peer_transfer_anomaly_score",
    "utility_bill_location_change",
    "sip_initiation_flag",
    "balance_floor_volatility",
]

def make_row(rng, event_idx):
    e = LIFE_EVENTS[event_idx]

    spend_delta         = rng.normal(0, 15)
    salary_variance     = rng.normal(0, 0.1)
    emi_ratio           = rng.uniform(0.0, 0.25)
    merchant_shift      = rng.uniform(0.0, 0.3)
    peer_anomaly        = rng.uniform(0.0, 0.3)
    location_change     = 0
    sip_flag            = 0
    balance_vol         = rng.uniform(0.05, 0.2)

    noise = lambda scale=1.0: rng.normal(0, scale)

    if e == "job_change":
        salary_variance     = rng.uniform(0.25, 0.6)   + noise(0.04)
        merchant_shift      = rng.uniform(0.45, 0.75)  + noise(0.04)
        spend_delta         = rng.uniform(18, 40)       + noise(3)
        peer_anomaly        = rng.uniform(0.35, 0.6)   + noise(0.04)

    elif e == "marriage":
        spend_delta         = rng.uniform(25, 55)       + noise(4)
        merchant_shift      = rng.uniform(0.5, 0.8)    + noise(0.04)
        peer_anomaly        = rng.uniform(0.4, 0.65)   + noise(0.04)
        emi_ratio           = rng.uniform(0.2, 0.45)   + noise(0.03)

    elif e == "new_child":
        spend_delta         = rng.uniform(20, 45)       + noise(4)
        merchant_shift      = rng.uniform(0.4, 0.7)    + noise(0.04)
        emi_ratio           = rng.uniform(0.15, 0.4)   + noise(0.03)
        balance_vol         = rng.uniform(0.25, 0.5)   + noise(0.03)

    elif e == "relocation":
        location_change     = 1
        merchant_shift      = rng.uniform(0.55, 0.8)   + noise(0.04)
        peer_anomaly        = rng.uniform(0.3, 0.55)   + noise(0.04)
        spend_delta         = rng.uniform(15, 35)       + noise(3)

    elif e == "education_loan":
        emi_ratio           = rng.uniform(0.3, 0.55)   + noise(0.03)
        balance_vol         = rng.uniform(0.3, 0.55)   + noise(0.03)
        spend_delta         = rng.uniform(-5, 10)       + noise(2)
        salary_variance     = rng.uniform(-0.1, 0.1)   + noise(0.02)

    elif e == "salary_hike":
        salary_variance     = rng.uniform(0.3, 0.65)   + noise(0.04)
        spend_delta         = rng.uniform(20, 50)       + noise(4)
        sip_flag            = int(rng.uniform(0, 1) > 0.35)
        balance_vol         = rng.uniform(0.1, 0.3)    + noise(0.02)

    elif e == "medical_event":
        spend_delta         = rng.uniform(15, 45)       + noise(4)
        merchant_shift      = rng.uniform(0.45, 0.7)   + noise(0.04)
        balance_vol         = rng.uniform(0.35, 0.6)   + noise(0.03)
        emi_ratio           = rng.uniform(0.2, 0.4)    + noise(0.03)

    elif e == "vehicle_purchase":
        emi_ratio           = rng.uniform(0.25, 0.5)   + noise(0.03)
        spend_delta         = rng.uniform(10, 30)       + noise(3)
        merchant_shift      = rng.uniform(0.3, 0.55)   + noise(0.04)
        balance_vol         = rng.uniform(0.2, 0.45)   + noise(0.03)

    elif e == "retirement_approach":
        sip_flag            = int(rng.uniform(0, 1) > 0.25)
        salary_variance     = rng.uniform(-0.15, 0.05) + noise(0.02)
        balance_vol         = rng.uniform(0.05, 0.2)   + noise(0.02)
        spend_delta         = rng.uniform(-10, 10)      + noise(2)

    elif e == "travel_pattern":
        merchant_shift      = rng.uniform(0.5, 0.75)   + noise(0.04)
        spend_delta         = rng.uniform(20, 50)       + noise(4)
        peer_anomaly        = rng.uniform(0.2, 0.45)   + noise(0.04)
        location_change     = int(rng.uniform(0, 1) > 0.4)

    elif e == "windfall_bonus":
        salary_variance     = rng.uniform(0.5, 0.9)    + noise(0.05)
        spend_delta         = rng.uniform(35, 70)       + noise(5)
        sip_flag            = int(rng.uniform(0, 1) > 0.3)
        balance_vol         = rng.uniform(0.1, 0.3)    + noise(0.02)

    elif e == "digital_shift":
        merchant_shift      = rng.uniform(0.4, 0.65)   + noise(0.04)
        peer_anomaly        = rng.uniform(0.35, 0.6)   + noise(0.04)
        sip_flag            = int(rng.uniform(0, 1) > 0.5)
        spend_delta         = rng.uniform(5, 25)        + noise(3)

    return {
        "monthly_spend_delta_pct":        round(float(np.clip(spend_delta, -50, 150)), 4),
        "salary_credit_variance":         round(float(np.clip(salary_variance, -0.5, 1.5)), 4),
        "emi_to_income_ratio_new":        round(float(np.clip(emi_ratio, 0.0, 0.8)), 4),
        "merchant_category_shift_score":  round(float(np.clip(merchant_shift, 0.0, 1.0)), 4),
        "peer_transfer_anomaly_score":    round(float(np.clip(peer_anomaly, 0.0, 1.0)), 4),
        "utility_bill_location_change":   int(location_change),
        "sip_initiation_flag":            int(sip_flag),
        "balance_floor_volatility":       round(float(np.clip(balance_vol, 0.0, 1.0)), 4),
        "life_event_label":               e,
    }

def generate():
    rng = np.random.default_rng(SEED)
    rows = []

    per_class = N_SAMPLES // len(LIFE_EVENTS)
    for idx in range(len(LIFE_EVENTS)):
        for _ in range(per_class):
            rows.append(make_row(rng, idx))

    remainder = N_SAMPLES - len(rows)
    for i in range(remainder):
        rows.append(make_row(rng, i % len(LIFE_EVENTS)))

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "customer_id", [f"CUST_{str(i+1).zfill(5)}" for i in range(len(df))])
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved: {OUTPUT_PATH}  |  Shape: {df.shape}")
    print(f"Class distribution:\n{df['life_event_label'].value_counts().to_string()}")
    return df

if __name__ == "__main__":
    generate()
