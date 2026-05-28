"""
Prediction Script
==================
Load the trained model and classify new / live sessions.

Usage:
  python src/predict.py                  — predict on all unclassified logs
  python src/predict.py --ip 192.168.1.5 — predict on specific IP
"""

import os
import json
import argparse
import joblib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE     = os.path.join(BASE_DIR, "data", "processed_logs.csv")
MODEL_FILE   = os.path.join(BASE_DIR, "models", "rf_model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

FEATURE_COLS = ["login_attempts", "command_count", "session_duration", "has_malicious_cmd"]

RISK_EMOJI = {"Normal": "🟢", "Suspicious": "🟡", "Malicious": "🔴"}


def predict_sessions(df: pd.DataFrame, model, le) -> pd.DataFrame:
    """Run model inference and append predicted_label + confidence columns."""
    X = df[FEATURE_COLS].fillna(0).values
    y_pred     = model.predict(X)
    y_proba    = model.predict_proba(X)
    confidence = y_proba.max(axis=1)

    df = df.copy()
    df["predicted_label"] = le.inverse_transform(y_pred)
    df["confidence"]      = (confidence * 100).round(1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Honeypot ML Predictor")
    parser.add_argument("--ip", type=str, help="Filter predictions for a specific IP")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_FILE):
        print("[!] Model not found. Run: python src/train_model.py first.")
        return

    model = joblib.load(MODEL_FILE)
    le    = joblib.load(ENCODER_FILE)
    print(f"[✓] Model loaded from {MODEL_FILE}")

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(CSV_FILE):
        print("[!] No processed CSV. Run: python src/process_logs.py first.")
        return

    df = pd.read_csv(CSV_FILE)
    if args.ip:
        df = df[df["ip"] == args.ip]
        if df.empty:
            print(f"[!] No sessions found for IP: {args.ip}")
            return

    # ── Predict ────────────────────────────────────────────────────────────────
    df = predict_sessions(df, model, le)

    print(f"\n{'='*65}")
    print("  PREDICTION RESULTS")
    print(f"{'='*65}")
    print(f"  Sessions analysed: {len(df)}")
    print(f"{'─'*65}")

    for _, row in df.iterrows():
        emoji = RISK_EMOJI.get(row["predicted_label"], "⚪")
        print(
            f"  {emoji} {row['predicted_label']:12s} ({row['confidence']:5.1f}%)  "
            f"IP={row['ip']:<15s}  user={row.get('username','?'):<12s}  "
            f"attempts={int(row['login_attempts'])}  cmds={int(row['command_count'])}"
        )

    print(f"{'─'*65}")
    print("  Summary:")
    for label, count in df["predicted_label"].value_counts().items():
        emoji = RISK_EMOJI.get(label, "⚪")
        print(f"    {emoji} {label:12s}: {count}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
