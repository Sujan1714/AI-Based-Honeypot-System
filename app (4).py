"""
Data Processing Module
=======================
Reads honeypot JSON logs → extracts features → writes structured CSV.

Features extracted per session:
  ip, timestamp, username, password, login_attempts,
  command_count, session_duration, has_malicious_cmd,
  label (Normal / Suspicious / Malicious)

Run: python src/process_logs.py
"""

import json
import os
import pandas as pd
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE   = os.path.join(BASE_DIR, "data", "honeypot_logs.json")
CSV_FILE   = os.path.join(BASE_DIR, "data", "processed_logs.csv")

# Commands considered highly malicious
MALICIOUS_PATTERNS = [
    "wget", "curl", "chmod", "chmod +x", "./", "crontab",
    "/etc/shadow", "rm -rf", "nc ", "netcat", "bash -i",
    "python -c", "perl -e", "php -r", "base64", "nmap",
]

# ── Feature Extraction ─────────────────────────────────────────────────────────
def has_malicious_command(commands: list) -> int:
    """Return 1 if any command matches a known malicious pattern."""
    for cmd in commands:
        cmd_lower = cmd.lower()
        if any(pat in cmd_lower for pat in MALICIOUS_PATTERNS):
            return 1
    return 0


def label_session(row) -> str:
    """
    Heuristic labelling rules (used to build training data):
      Malicious   — malicious command OR many attempts + long session
      Suspicious  — multiple attempts OR some commands
      Normal      — single attempt, no commands (scanner / background noise)
    """
    if row["has_malicious_cmd"] == 1:
        return "Malicious"
    if row["login_attempts"] >= 5 or (row["command_count"] > 3 and row["session_duration"] > 30):
        return "Malicious"
    if row["login_attempts"] >= 3 or row["command_count"] >= 1:
        return "Suspicious"
    return "Normal"


def process_logs():
    """Main processing pipeline: JSON → feature DataFrame → CSV."""

    # ── Load logs ──────────────────────────────────────────────────────────────
    if not os.path.exists(LOG_FILE):
        print(f"[!] Log file not found: {LOG_FILE}")
        print("[!] Start the honeypot and simulate some attacks first.")
        return None

    with open(LOG_FILE, "r") as f:
        raw_logs = json.load(f)

    print(f"[*] Loaded {len(raw_logs)} raw log entries.")

    # ── Group entries by session (ip + timestamp minute) ──────────────────────
    records = []
    for entry in raw_logs:
        # Only process session events (not raw auth_attempt events)
        if entry.get("event") == "session":
            records.append({
                "timestamp"       : entry.get("timestamp", ""),
                "ip"              : entry.get("ip", "0.0.0.0"),
                "username"        : entry.get("username", ""),
                "password"        : entry.get("password", ""),
                "login_attempts"  : int(entry.get("login_attempts", 1)),
                "commands"        : entry.get("commands", []),
                "command_count"   : int(entry.get("command_count", 0)),
                "session_duration": float(entry.get("session_duration", 0)),
            })
        elif entry.get("event") == "auth_attempt":
            # Include auth-only events as potential Normal/Suspicious
            records.append({
                "timestamp"       : entry.get("timestamp", ""),
                "ip"              : entry.get("ip", "0.0.0.0"),
                "username"        : entry.get("username", ""),
                "password"        : entry.get("password", ""),
                "login_attempts"  : int(entry.get("login_attempts", 1)),
                "commands"        : [],
                "command_count"   : 0,
                "session_duration": float(entry.get("session_duration", 0)),
            })

    if not records:
        print("[!] No processable log entries found.")
        return None

    df = pd.DataFrame(records)

    # ── Derived features ───────────────────────────────────────────────────────
    df["has_malicious_cmd"] = df["commands"].apply(has_malicious_command)
    df["label"]             = df.apply(label_session, axis=1)

    # Drop raw command list (not needed for ML)
    df_out = df.drop(columns=["commands"])

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df_out.to_csv(CSV_FILE, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total records : {len(df_out)}")
    print(f"  Output CSV    : {CSV_FILE}")
    print(f"\n  Label distribution:")
    for label, count in df_out["label"].value_counts().items():
        print(f"    {label:15s}: {count}")
    print(f"{'='*50}\n")

    print(df_out[["ip", "username", "login_attempts", "command_count",
                  "session_duration", "has_malicious_cmd", "label"]].head(10).to_string(index=False))

    return df_out


if __name__ == "__main__":
    process_logs()
