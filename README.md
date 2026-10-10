# NLP Model Monitoring & Drift Detection on Google Cloud Platform

This project implements an end-to-end Machine Learning Model Monitoring system deployed on **Google Cloud Platform (Cloud Run)**. It serves a text sentiment analysis model and monitors for **Domain Shift (Data Drift)** and **Prediction Drift** across real-time inference payloads.

---

## 🏗️ Architecture & Project Setup

- **Model**: TF-IDF Vectorizer + Logistic Regression trained on news/science article data.
- **Serving API**: FastAPI microservice exposing `/predict` and `/metrics` endpoints.
- **Containerization**: Docker image stored in GCP Artifact Registry and deployed to serverless **GCP Cloud Run**.
- **Monitoring Tools**: GCP Cloud Run Metrics Explorer (p50, p95, p99 latency tracking) and Prometheus histograms.
- **Statistical Drift Engine**: Python engine using SciPy (`ks_2samp`, `chi2_contingency`) and NumPy (`PSI`).

---

## 📁 Repository Structure

```text
nlp-model-monitoring/
├── app/
│   ├── main.py             # FastAPI server with Prometheus metric collection
│   ├── load_test.py        # Client load generator & drift injection script
│   └── drift_analyzer.py   # Statistical drift detection script (KS-Test, PSI, Chi-Square)
├── model/
│   ├── train.py            # Dataset extraction, feature summary, & model training
│   └── model.pkl           # Saved Scikit-Learn model artifact
├── data/
│   ├── reference.csv       # Training baseline summary metrics (Reference Window)
│   └── current_logs.jsonl  # Live production request logs (Current Window)
├── Dockerfile              # Container specification for Cloud Run deployment
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Files excluded from version control
└── README.md               # Project documentation

Drift Detection Methodology
To evaluate drift on unstructured text data without complex embeddings, the pipeline extracts numerical and categorical meta-features from raw input payloads:

Feature / Metric,Statistical Test,Target Drift Identified
text_length,2-Sample Kolmogorov-Smirnov (KS) Test,Input Feature / Data Drift
word_count,2-Sample Kolmogorov-Smirnov (KS) Test,Input Feature / Data Drift
confidence_score,Population Stability Index (PSI),Model Uncertainty / Prediction Drift
prediction,Chi-Square Goodness-of-Fit Test,Output Label Shift

Step-by-Step Execution Guide
1. Local Environment Setup & Training
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model and generate reference dataset baseline
python model/train.py

Run Local API Test
Bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
3. Deploy to GCP Cloud Run
Bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"

# Create Artifact Registry Repository
gcloud artifacts repositories create nlp-repo \
    --repository-format=docker \
    --location=$REGION

# Build & Deploy
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/nlp-repo/sentiment-api:v1 .
gcloud run deploy sentiment-monitoring-service \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/nlp-repo/sentiment-api:v1 \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated
4. Execute Load Test & Inject Domain Drift
Bash
# Update SERVICE_URL in app/load_test.py with your Cloud Run endpoint
python app/load_test.py
5. Evaluate Statistical Drift Results
Bash
python app/drift_analyzer.py
🔒 Security & Billing Compliance
Budget alerts configured in GCP Console.

No hardcoded secrets, API keys, or service account keys committed to git (.gitignore applied).

Resources wiped post-evaluation using gcloud run services delete and gcloud artifacts repositories delete.


---

## Part 2: Setting Up `.gitignore`

Before making Git commits, create a `.gitignore` file in your root folder to ensure temporary files, virtual environments, and secrets are excluded:

Create `.gitignore`:
```text
# Python bytecode & virtual environments
__pycache__/
*.py[cod]
venv/
.venv/
env/

# Data logs generated during production runs
data/current_logs.jsonl

# IDE & OS specific files
.vscode/
.idea/
.DS_Store

# GCP Secrets / Keys (Safety requirement)
*.json
*.pem
key.json