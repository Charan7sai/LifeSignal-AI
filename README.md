# LifeSignal AI 🧠
### Proactive Life-Event Engagement Engine for SBI

> *"Don't wait for your customers to come to you. Meet them at their moment."*

**SBI Hackathon @ GFF 2026 | Theme: Agentic AI & Emerging Tech | PS3: Digital Engagement**

---

## Overview

LifeSignal AI is an agentic behavioral intelligence system that monitors SBI customer transaction streams in real-time, detects life-event signals, and proactively triggers hyper-personalized financial product recommendations — before the customer realizes they need them.

The system assigns each customer a **Life-Stage Transition Index (LSTI)** score across 12 life-event categories using a Random Forest pipeline on anonymized behavioral features. When the LSTI crosses a threshold, a LangChain-based AI agent selects the optimal SBI product and dispatches it via the customer's highest-engagement channel.

---

## Architecture

```
Transaction Stream (Kafka)
        │
        ▼
Feature Engineering (Spark)
        │
        ▼
LSTI Scorer (Random Forest)  ──── 12 Life-Event Categories
        │
   Threshold?
    YES │
        ▼
Agentic Decision Layer (LangChain + Llama 3)
        │
        ▼
Omnichannel Dispatch (YONO / SMS / WhatsApp)
        │
        ▼
Feedback Loop ──────────────────► Monthly Retraining
```

---

## Life-Event Categories Detected

| # | Event | Key Signal |
|---|-------|-----------|
| 1 | Job Change | Salary variance + merchant shift |
| 2 | Marriage | Spend spike + EMI emergence |
| 3 | New Child | Spend delta + balance volatility |
| 4 | Relocation | Location change + merchant shift |
| 5 | Education Loan | EMI ratio + balance volatility |
| 6 | Salary Hike | Salary variance + SIP initiation |
| 7 | Medical Event | Spend spike + merchant shift |
| 8 | Vehicle Purchase | EMI ratio + spend delta |
| 9 | Retirement Approach | SIP flag + salary variance |
| 10 | Travel Pattern | Merchant shift + location change |
| 11 | Windfall/Bonus | Salary variance + spend spike |
| 12 | Digital Shift | Merchant shift + peer anomaly |

---

## Behavioral Features

| Feature | Description |
|---------|-------------|
| `monthly_spend_delta_pct` | Monthly spend change vs 3-month rolling avg (%) |
| `salary_credit_variance` | Salary credit variance ratio (0–1) |
| `emi_to_income_ratio_new` | New EMI-to-income ratio (0–1) |
| `merchant_category_shift_score` | Merchant category distribution shift (0–1) |
| `peer_transfer_anomaly_score` | Peer transfer anomaly score (0–1) |
| `utility_bill_location_change` | Utility bill location changed (0/1) |
| `sip_initiation_flag` | New SIP initiated this month (0/1) |
| `balance_floor_volatility` | Balance floor volatility index (0–1) |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~79.9% |
| CV Accuracy (5-fold) | ~79.4% |
| Training Samples | 8,000 |
| Test Samples | 2,000 |
| Algorithm | Random Forest |


---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Charan7sai/lifesignal-ai.git
cd lifesignal-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python train.py

# 4. Run predictions
python predict.py                          # interactive mode
python predict.py --batch customers.csv    # batch mode
python predict.py --json '{"monthly_spend_delta_pct": 38.5, ...}'  # JSON mode
```

---

## Usage Examples

### Interactive Mode
```bash
$ python predict.py

  LifeSignal AI — Life-Stage Transition Index (LSTI)
  Proactive Customer Engagement Engine for SBI

  Mode: Interactive Single Customer
  Model accuracy: 79.95%  |  Features: 8

  Enter Customer ID: CUST_ARJ_001

  monthly_spend_delta_pct: 38.5
  salary_credit_variance:  0.42
  ...

  LSTI PREDICTION RESULT
  ══════════════════════════════════════════════════════════════

  Customer ID : CUST_ARJ_001

  LSTI Score — Top Life-Event Probabilities:
    1. salary_hike       ████████████░░░░░░░░░░░░░░░░░░  35.5% ◄ TRIGGERED
    2. windfall_bonus    ███████████░░░░░░░░░░░░░░░░░░░  33.8%
    3. job_change        ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7.4%

  STATUS  : LSTI Threshold Crossed — Agentic Layer Activated
  EVENT   : SALARY HIKE
  OFFER   : SBI SIP (Mutual Fund) + Fixed Deposit combo
  CHANNEL : YONO Push Notification (primary) / SMS (fallback)
```

### Batch Mode
```bash
$ python predict.py --batch customers.csv --output predictions.csv

  Processed  : 500 customers
  Triggered  : 187 (37.4%) — LSTI threshold crossed
  Monitoring : 313 — below threshold

  Top triggered events:
    salary_hike              52 customers
    job_change               38 customers
    marriage                 31 customers
```

### JSON Mode (API integration)
```bash
$ python predict.py --json '{"monthly_spend_delta_pct": 38.5, "salary_credit_variance": 0.42, ...}'

{
  "predicted_event": "salary_hike",
  "lsti_confidence": 0.3546,
  "triggered": true,
  "recommended_offer": "SBI SIP (Mutual Fund) + Fixed Deposit combo",
  "all_scores": {
    "salary_hike": 0.3546,
    "windfall_bonus": 0.3377,
    ...
  }
}
```

---

## Repository Structure

```
lifesignal-ai/
├── train.py               # Model training + evaluation
├── predict.py             # Interactive/batch/JSON prediction CLI
├── requirements.txt
├── data/
│   └── transactions_behavioral.csv   # Generated dataset
└── model/
    ├── lsti_rf_model.pkl             # Trained Random Forest
    ├── label_encoder.pkl             # Label encoder
    └── model_meta.json               # Model metadata + accuracy
```

---

## Production Integration — How It Works in the Real App

In production, **the customer never manually inputs anything.** The entire pipeline runs silently in the background the moment a customer logs into YONO.

```
Customer opens YONO app and logs in
        ↓
SBI core banking API silently pulls their transaction history
        ↓
Backend automatically computes all 8 behavioral features
        ↓
LSTI model scores the customer across 12 life-event categories
        ↓
If score crosses threshold → personalized push notification appears
        ↓
Customer just sees: "Arjun, your new salary opens up exciting savings options 🎉"
```

The customer experiences this as a **smart, timely nudge** — they have no idea a behavioral ML pipeline just ran in the background.

### What the current demo simulates
The `predict.py` CLI simulates this production flow using pre-computed behavioral features stored in the dataset (as a stand-in for what would come from SBI's core banking API in real deployment). The model logic, scoring, and offer recommendation are identical to what would run in production.

### Full production stack (post-hackathon roadmap)
- **YONO login event** triggers a feature computation job via Kafka
- **Spark** aggregates the last 90 days of transactions into the 8 behavioral features
- **LSTI scorer** runs in under 100ms via Redis-cached FastAPI endpoint
- **LangChain agent** selects offer + personalizes message
- **Firebase** delivers push notification to YONO app
- **Zero customer input required at any stage**

---

## Compliance & Privacy

- All features are **anonymized behavioral signals** — no PII enters the ML pipeline
- Designed for **on-premise deployment** on SBI private cloud (Kubernetes)
- Compliant with **RBI Data Localization circular (2018)**
- Operates under **DPDP Act 2023** — processing limited to stated purpose
- Every prediction is **logged with timestamp + model version** for audit trail

---

## Tech Stack

- **ML:** Python, scikit-learn (Random Forest)
- **Data:** pandas, NumPy
- **Serving:** joblib model artifacts, FastAPI (production wrapper)
- **Agentic Layer:** LangChain + Llama 3 / Mistral (self-hosted via Ollama/vLLM)
- **Delivery:** Firebase Cloud Messaging (YONO), Twilio SMS/WhatsApp

---

## Author

**Charan Sai Neelisetty**
B.Tech AI/ML, Manipal University Jaipur
IIT Madras BS Data Science & Applications
Cybersecurity AI/ML Research Intern, SwiftSafe

GitHub: [@Charan7sai](https://github.com/Charan7sai)

---

*Built for SBI Hackathon @ GFF 2026 — Theme: Agentic AI & Emerging Tech*
