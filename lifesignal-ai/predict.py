import sys
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

MODEL_PATH   = "model/lsti_rf_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
META_PATH    = "model/model_meta.json"
DATA_PATH    = "data/transactions_behavioral.csv"

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

PRODUCT_MAP = {
    "job_change":          "SBI Salary Account Upgrade + Personal Loan pre-approval",
    "marriage":            "SBI Home Loan + Joint Account + Term Insurance",
    "new_child":           "SBI Child Education Plan + Recurring Deposit",
    "relocation":          "SBI Home Loan + NRI Services (if applicable)",
    "education_loan":      "SBI Student Loan Scheme + Scholar Loan",
    "salary_hike":         "SBI SIP (Mutual Fund) + Fixed Deposit combo",
    "medical_event":       "SBI Health Insurance Top-up + Emergency Overdraft",
    "vehicle_purchase":    "SBI Car Loan / Two-Wheeler Loan",
    "retirement_approach": "SBI Annuity Deposit + Senior Citizen Savings Scheme",
    "travel_pattern":      "SBI Forex Card + Travel Insurance",
    "windfall_bonus":      "SBI Wealth Management + Lump-sum MF Investment",
    "digital_shift":       "SBI YONO Premium + Digital Savings Account",
}

ANSI = {
    "header": "\033[1;36m",
    "bold":   "\033[1m",
    "green":  "\033[1;32m",
    "yellow": "\033[1;33m",
    "red":    "\033[1;31m",
    "cyan":   "\033[36m",
    "gray":   "\033[90m",
    "reset":  "\033[0m",
    "blue":   "\033[1;34m",
    "gold":   "\033[33m",
}

def c(color, text):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(c("red", "[ERROR] Model not found. Run: python train.py"))
        sys.exit(1)
    clf  = joblib.load(MODEL_PATH)
    le   = joblib.load(ENCODER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return clf, le, meta

def load_data():
    if not os.path.exists(DATA_PATH):
        print(c("red", "[ERROR] Dataset not found. Run: python train.py"))
        sys.exit(1)
    return pd.read_csv(DATA_PATH)

def format_bar(prob, width=30):
    filled = int(prob * width)
    bar    = "█" * filled + "░" * (width - filled)
    col    = "green" if prob >= 0.5 else "yellow" if prob >= 0.25 else "gray"
    return c(col, bar)

def print_banner():
    print()
    print(c("header", "═" * 62))
    print(c("header", "   LifeSignal AI — Life-Stage Transition Index (LSTI)"))
    print(c("header", "   Proactive Customer Engagement Engine for SBI"))
    print(c("header", "═" * 62))
    print()

def print_customer_profile(row):
    print(c("bold", "  Customer Profile (Behavioral Features):"))
    print(c("gray",  "  ─" * 30))
    labels = {
        "monthly_spend_delta_pct":       "Monthly Spend Delta",
        "salary_credit_variance":        "Salary Credit Variance",
        "emi_to_income_ratio_new":       "EMI-to-Income Ratio",
        "merchant_category_shift_score": "Merchant Category Shift",
        "peer_transfer_anomaly_score":   "Peer Transfer Anomaly",
        "utility_bill_location_change":  "Location Change",
        "sip_initiation_flag":           "SIP Initiated",
        "balance_floor_volatility":      "Balance Volatility",
    }
    for feat, label in labels.items():
        val = row[feat]
        if feat in ["utility_bill_location_change", "sip_initiation_flag"]:
            display = c("green", "Yes") if val == 1 else c("gray", "No")
        else:
            display = c("cyan", f"{val:.4f}")
        print(f"    {label:<30} {display}")
    print()

def predict_customer(customer_id, clf, le, df, top_n=3, show_profile=True):
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        print(c("red", f"  [ERROR] Customer ID '{customer_id}' not found in dataset."))
        print(c("gray", f"  Try IDs like: CUST_00001 to CUST_{str(len(df)).zfill(5)}"))
        return

    row = row.iloc[0]

    print(c("bold",  f"  Customer ID  : ") + c("cyan", customer_id))
    actual = row.get("life_event_label", None)
    if actual:
        print(c("gray", f"  Actual Event : {actual}  (ground truth from dataset)"))
    print()

    if show_profile:
        print_customer_profile(row)

    features = np.array([[row[f] for f in FEATURE_COLS]])
    probs    = clf.predict_proba(features)[0]
    top_idxs = np.argsort(probs)[::-1][:top_n]

    print(c("bold", "  LSTI Score — Top Life-Event Probabilities:"))
    print()
    for rank, idx in enumerate(top_idxs):
        label = le.classes_[idx]
        prob  = probs[idx]
        bar   = format_bar(prob)
        tag   = c("green", " ◄ TRIGGERED") if rank == 0 and prob >= 0.30 else ""
        print(f"    {rank+1}. {label:<25} {bar}  {prob*100:5.1f}%{tag}")

    print()
    pred_label = le.classes_[np.argmax(probs)]
    pred_prob  = probs[np.argmax(probs)]

    if pred_prob >= 0.30:
        print(c("green",  "  STATUS   : LSTI Threshold Crossed — Agentic Layer Activated"))
        print(c("bold",   f"  EVENT     : {pred_label.upper().replace('_', ' ')}"))
        print(c("yellow", f"  OFFER     : {PRODUCT_MAP.get(pred_label, 'Personalised financial consultation')}"))
        print(c("cyan",   "  CHANNEL   : YONO Push Notification (primary) / SMS (fallback)"))
    else:
        print(c("gray",  "  STATUS   : Below threshold — monitoring continues"))
        print(c("gray",  f"  TOP HIT  : {pred_label} ({pred_prob*100:.1f}%) — insufficient confidence"))

    print()
    print(c("gray", "  " + "─" * 58))
    print()

def interactive_mode(clf, le, meta, df):
    print_banner()
    print(c("bold", "  Mode: Interactive — Lookup by Customer ID"))
    print(c("gray", f"  Model Accuracy : {meta['test_accuracy']*100:.2f}%"))
    print(c("gray", f"  Customers Loaded: {len(df):,}  |  IDs: CUST_00001 to CUST_{str(len(df)).zfill(5)}"))
    print()

    while True:
        cid = input(c("bold", "  Enter Customer ID (or 'q' to quit): ")).strip()
        if cid.lower() in ("q", "quit", "exit"):
            print(c("gray", "\n  Exiting LifeSignal AI. Goodbye.\n"))
            break
        if not cid:
            cid = "CUST_00001"
        print()
        print(c("header", "  " + "═" * 58))
        print(c("header", "  LSTI PREDICTION RESULT"))
        print(c("header", "  " + "═" * 58))
        print()
        predict_customer(cid, clf, le, df)

def batch_mode(clf, le, df, input_csv, output_csv):
    print_banner()
    print(c("bold", f"  Mode: Batch  |  Input: {input_csv}"))
    print()

    if not os.path.exists(input_csv):
        print(c("red", f"[ERROR] File not found: {input_csv}"))
        sys.exit(1)

    batch_df = pd.read_csv(input_csv)

    if "customer_id" in batch_df.columns:
        results = []
        for cid in batch_df["customer_id"]:
            row = df[df["customer_id"] == cid]
            if row.empty:
                continue
            row = row.iloc[0]
            features = np.array([[row[f] for f in FEATURE_COLS]])
            probs    = clf.predict_proba(features)[0]
            pred_idx = np.argmax(probs)
            results.append({
                "customer_id":       cid,
                "predicted_event":   le.classes_[pred_idx],
                "lsti_confidence":   round(float(probs[pred_idx]), 4),
                "triggered":         int(probs[pred_idx] >= 0.30),
                "recommended_offer": PRODUCT_MAP.get(le.classes_[pred_idx], "Personalised consultation"),
            })
        out_df = pd.DataFrame(results)
    else:
        missing = [f for f in FEATURE_COLS if f not in batch_df.columns]
        if missing:
            print(c("red", f"[ERROR] Missing columns: {missing}"))
            sys.exit(1)
        X      = batch_df[FEATURE_COLS].values
        probs  = clf.predict_proba(X)
        preds  = np.argmax(probs, axis=1)
        labels = le.classes_[preds]
        confs  = probs[np.arange(len(preds)), preds]
        batch_df["predicted_event"]   = labels
        batch_df["lsti_confidence"]   = confs.round(4)
        batch_df["triggered"]         = (confs >= 0.30).astype(int)
        batch_df["recommended_offer"] = [PRODUCT_MAP.get(l, "Personalised consultation") for l in labels]
        out_df = batch_df

    out_df.to_csv(output_csv, index=False)
    triggered = out_df["triggered"].sum()
    print(c("bold",  f"  Processed  : {len(out_df)} customers"))
    print(c("green", f"  Triggered  : {triggered} ({triggered/len(out_df)*100:.1f}%) — LSTI threshold crossed"))
    print(c("gray",  f"  Monitoring : {len(out_df)-triggered} — below threshold"))
    print()
    print(c("bold", "  Top triggered events:"))
    top = out_df[out_df["triggered"]==1]["predicted_event"].value_counts().head(5)
    for event, count in top.items():
        print(f"    {event:<30} {count} customers")
    print()
    print(c("green", f"  Output saved: {output_csv}"))
    print()

def main():
    parser = argparse.ArgumentParser(
        description="LifeSignal AI — LSTI Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                             # interactive — enter customer ID
  python predict.py --id CUST_00042            # direct single customer lookup
  python predict.py --batch customer_ids.csv   # batch mode
        """
    )
    parser.add_argument("--id",     type=str, help="Customer ID for direct lookup")
    parser.add_argument("--batch",  type=str, help="CSV with customer_id column for batch prediction")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Output CSV for batch mode")
    args = parser.parse_args()

    clf, le, meta = load_model()
    df            = load_data()

    if args.id:
        print_banner()
        print(c("header", "  LSTI PREDICTION RESULT"))
        print(c("header", "  " + "═" * 58))
        print()
        predict_customer(args.id, clf, le, df)
    elif args.batch:
        batch_mode(clf, le, df, args.batch, args.output)
    else:
        interactive_mode(clf, le, meta, df)

if __name__ == "__main__":
    main()
