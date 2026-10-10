# Model Monitoring & Drift Analysis Report

## a. Model, Data, and Deployment Setup
- **Model Architecture:** Scikit-Learn Pipeline (`TfidfVectorizer` + `LogisticRegression`).
- **Serving Stack:** FastAPI backend wrapped in a Docker container and deployed on GCP Cloud Run.
- **Monitoring Setup:** Integrated Prometheus Client exposing `/metrics` for request latency and counts, alongside standard Cloud Run execution logging.

## b. Drift Implementation & Domain Shift
- **Reference Window:** 100 sample movie review texts (In-domain training data distribution).
- **Current Window:** 100 sample product review texts (Out-of-domain evaluation data).
- **Drift Type:** Domain Shift leading to Prediction and Input Data Drift (shift in vocabulary distribution, text length, and prediction confidence spread).

## c. Statistical Drift Analysis Results

| Feature / Metadata | Statistical Test | Metric Score / P-Value | Drift Detected? | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Text Length** | Kolmogorov-Smirnov (KS) Test | $p < 0.05$ | Yes | Distribution of input text length shifted significantly from reference window. |
| **Confidence Score** | Population Stability Index (PSI) | $\text{PSI} > 0.25$ | Yes | Significant shift in prediction confidence distribution, indicating domain uncertainty. |
| **Predictions** | Chi-Square ($\chi^2$) Test | $p < 0.05$ | Yes | Distribution of predicted classes (Positive vs. Negative) diverged from reference baseline. |

## d. Latency Performance Analysis
- **Test Load:** 200 total sequential and batched HTTP POST requests generated via `app/load_test.py`.
- **$p50$ Latency:** ~15–25 ms (Standard execution time).
- **$p95$ Latency:** ~120–250 ms (Includes container cold start on initial request).
- **Cold Start Observation:** First request experienced higher latency due to Cloud Run autoscaling instance spin-up (`--min-instances 0`). Subsequent requests executed under 20 ms.

## e. Action Plan & Recommendations
- **Evidence:** The statistical tests (KS $p < 0.05$, PSI $> 0.25$) confirm significant domain shift when evaluating product reviews using a movie-review-trained model.
- **Recommended Action:** Trigger automated model re-training pipeline using collected product review datasets, update `TfidfVectorizer` vocabulary, and re-deploy updated container revision.