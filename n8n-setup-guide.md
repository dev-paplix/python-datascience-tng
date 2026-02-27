# n8n Setup Guide: Automating Data & ML Workflows

> **What is n8n?** n8n (pronounced "n-eight-n") is an open-source, self-hostable workflow automation platform. Think of it as a visual programming tool that connects your Python scripts, APIs, databases, and communication tools — no coding needed for the glue.

---

## Table of Contents

1. [Creating a Free n8n Cloud Account](#1-creating-a-free-n8n-cloud-account)
2. [n8n Core Concepts](#2-n8n-core-concepts)
3. [Connecting from Jupyter Notebook (local)](#3-connecting-from-jupyter-notebook-local)
4. [Connecting from Google Colab](#4-connecting-from-google-colab)
5. [Workflow Recipes for Data & ML](#5-workflow-recipes-for-data--ml)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Creating a Free n8n Cloud Account

### Step 1: Sign Up

1. Go to **[https://app.n8n.cloud/register](https://app.n8n.cloud/register)**
2. Enter your name, email address, and a password
3. Click **Create account**
4. Check your email — click the **Verify email** link

### Step 2: Set Up Your Workspace

1. After verifying, you land on the **n8n Dashboard**
2. Your workspace URL will be: `https://YOUR-INSTANCE.app.n8n.cloud`
3. Click **New Workflow** to open the canvas editor

> [!NOTE]
> The **Starter plan** (free) gives you 2,500 workflow executions/month and unlimited workflows. This is more than enough for learning.

### Step 3: Explore the Interface

| Area | Description |
|---|---|
| **Canvas** | The visual workflow builder |
| **Left Panel** | Node library — search for nodes here |
| **Top Bar** | Save, Execute, and Active toggle |
| **Executions** | History of past workflow runs |
| **Credentials** | Store API keys, OAuth tokens |

---

## 2. n8n Core Concepts

```
[Trigger Node] → [Action Node 1] → [Action Node 2] → ...
```

### Key Nodes You'll Use

| Node | Purpose |
|---|---|
| **Schedule Trigger** | Run workflow on a timer (daily, hourly, etc.) |
| **Webhook** | Create an HTTP endpoint to receive data |
| **HTTP Request** | Call any REST API |
| **Execute Command** | Run a shell/Python command on the host machine |
| **Code** | Run JavaScript/Python snippets inline |
| **Send Email** | Send emails via SMTP or Gmail |
| **Slack** | Post messages to Slack channels |
| **Microsoft Teams** | Send messages to Teams channels |
| **Read/Write File** | Read or write files on the host filesystem |
| **IF** | Conditional branching |
| **Set** | Modify data between nodes |

---

## 3. Connecting from Jupyter Notebook (local)

### How it Works

Your Jupyter notebook sends data **to** n8n by hitting a **Webhook URL**. n8n receives the data, processes it, and can respond back.

```
[Jupyter Notebook] → POST /webhook → [n8n Webhook Node] → [...actions...]
```

### Step 1: Create a Webhook Workflow in n8n

1. Click **New Workflow** in your n8n dashboard
2. Click the **+** button and search for **Webhook**
3. Set **HTTP Method** to `POST`
4. Click **Listen for test event** — you'll see your webhook URL, e.g.:
   ```
   https://YOUR-INSTANCE.app.n8n.cloud/webhook-test/your-unique-id
   ```
5. Add a **Respond to Webhook** node at the end (to send data back to your notebook)

### Step 2: Send Data from Jupyter

```python
import requests
import json

# Your n8n Webhook URL (from the Webhook node in n8n)
N8N_WEBHOOK_URL = "https://YOUR-INSTANCE.app.n8n.cloud/webhook-test/your-unique-id"

def send_to_n8n(data: dict, webhook_url: str = N8N_WEBHOOK_URL) -> dict:
    """
    Send a Python dict to an n8n webhook.
    Returns n8n's response.
    """
    response = requests.post(
        webhook_url,
        json=data,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()
    return response.json()

# Example: Send ML prediction results to n8n
result = {
    "model": "salary_predictor_v2",
    "run_date": "2024-01-15",
    "r2_score": 0.87,
    "rmse": 4821.50,
    "predictions_file": "data/exports/predictions_2024-01-15.parquet",
    "status": "success"
}

response = send_to_n8n(result)
print("n8n response:", response)
```

### Step 3: Trigger n8n from a Model Training Cell

```python
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib, os, json
from datetime import datetime

N8N_WEBHOOK_URL = "https://YOUR-INSTANCE.app.n8n.cloud/webhook/your-unique-id"

def train_and_notify(df, feature_cols, target_col, model_path="models/model.pkl"):
    """Train a model and notify n8n when done."""
    X = df[feature_cols]
    y = df[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    # Notify n8n
    payload = {
        "event": "model_trained",
        "timestamp": datetime.now().isoformat(),
        "metrics": {"r2": round(r2, 4)},
        "model_saved_to": model_path,
        "feature_count": len(feature_cols)
    }

    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=15)
        print(f"✅ n8n notified. Status: {r.status_code}")
    except Exception as e:
        print(f"⚠️  n8n notification failed: {e}")

    return model, r2
```

### Step 4: Receive Data Back from n8n in Jupyter

If your n8n workflow returns a prediction or transformed result, capture it:

```python
# Send data and receive processed output from n8n
input_data = {"features": [35, 10, 85]}  # age, experience, performance

response = requests.post(N8N_WEBHOOK_URL, json=input_data, timeout=30)

if response.status_code == 200:
    result = response.json()
    print("Prediction from n8n pipeline:", result.get("prediction"))
else:
    print(f"Error: {response.status_code} - {response.text}")
```

---

## 4. Connecting from Google Colab

> [!IMPORTANT]
> Google Colab runs in the cloud and **cannot receive webhooks** directly (you can't host a server in Colab). However, Colab **can call** n8n webhooks outbound just fine.

### Setting Up in Colab

```python
# Cell 1: Install dependencies (if not already available)
# !pip install requests -q  # Already available in Colab

import requests
import json
from datetime import datetime

# Your n8n Webhook URL
N8N_WEBHOOK_URL = "https://YOUR-INSTANCE.app.n8n.cloud/webhook/your-unique-id"
```

```python
# Cell 2: Helper function
def notify_n8n(event_type: str, data: dict):
    """Send an event to your n8n workflow from Google Colab."""
    payload = {
        "source": "google_colab",
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **data
    }
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=20)
        response.raise_for_status()
        print(f"✅ Sent '{event_type}' to n8n | Status: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ n8n notification failed: {e}")
        return None
```

```python
# Cell 3: Use after training a model in Colab
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- Your training code ---
np.random.seed(42)
X = np.random.randn(500, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
acc = accuracy_score(y_te, preds)
f1  = f1_score(y_te, preds)

print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

# --- Notify n8n ---
notify_n8n("model_trained", {
    "model_type": "RandomForestClassifier",
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "n_estimators": 100,
    "colab_session": "production_run_v3"
})
```

### Sending Files from Colab to n8n

To send a CSV or Parquet file to n8n via its HTTP Request Node:

```python
# Upload a file to n8n via multipart form data
import io

# Example: send a predictions CSV
predictions_df = pd.DataFrame({"id": range(10), "prediction": np.random.randint(0, 2, 10)})
csv_buffer = io.StringIO()
predictions_df.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue()

response = requests.post(
    N8N_WEBHOOK_URL,
    files={"file": ("predictions.csv", csv_content, "text/csv")},
    data={"run_id": "colab_run_001", "rows": len(predictions_df)},
    timeout=30
)
print("Upload status:", response.status_code)
```

---

## 5. Workflow Recipes for Data & ML

### Recipe 1: Daily Dataset Refresh

```
[Schedule: Daily 6AM]
  → [HTTP Request: GET https://api.example.com/data]
  → [Code: Parse JSON, flatten to CSV]
  → [Write Binary File: data/raw/latest.csv]
  → [Execute Command: python scripts/run_pipeline.py --input data/raw/latest.csv]
  → [Slack: ✅ Dataset refreshed and pipeline ran]
```

### Recipe 2: ML Prediction Trigger (Webhook → Predict → Return)

```
[Webhook: POST /predict]
  → [Execute Command: python scripts/predict.py --input '{{ $json.body }}']
  → [Set: prediction = {{ $json.stdout | parseJson }}]
  → [Respond to Webhook: {{ $json.prediction }}]
```

### Recipe 3: Auto-Send Report to Email + Teams

```
[Schedule: Monday 8AM]
  → [Execute Command: python scripts/generate_report.py]
  → [Read Binary File: data/exports/report.html]
  → [Send Email: to=manager@company.com, attach=report.html]
  → [Microsoft Teams: "📊 Weekly report sent!"]
```

### Recipe 4: Alert on Model Performance Degradation

```
[Webhook: POST /metrics (called from Jupyter after each run)]
  → [IF: {{ $json.r2 }} < 0.75]
    → [YES] → [Slack: "⚠️ Model R² dropped to {{ $json.r2 }}! Review needed."]
    → [NO]  → [Slack: "✅ Model healthy. R²={{ $json.r2 }}"]
```

---

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| `ConnectionRefusedError` | Make sure your n8n webhook is active (click "Listen for test event") |
| `404 Not Found` | Use the test URL during development, production URL for live workflows |
| Webhook returns empty | Check the Webhook node settings — ensure "Respond to Webhook" node is connected |
| Colab can't receive callbacks | Colab is outbound-only. Use the n8n Code node to poll Colab instead |
| `Timeout` | Increase `timeout` in your `requests.post` call |
| Long-running scripts time out | Use **Execute Command** asynchronously and send results back via a second webhook |

> [!TIP]
> During development, always use the **test webhook URL** (with `-test` in it). Switch to the **production webhook URL** (without `-test`) only when you activate the workflow with the toggle in the top-right corner of the n8n canvas.

---

## Quick Reference

```python
# Minimal boilerplate: send data from Jupyter/Colab to n8n
import requests

N8N_URL = "https://YOUR-INSTANCE.app.n8n.cloud/webhook/YOUR-ID"

requests.post(N8N_URL, json={
    "event": "pipeline_complete",
    "r2": 0.91,
    "model": "LinearRegression"
})
```
