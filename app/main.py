#script handles:
# 1. Serving /predict requests and calculating real-time meta-features (text_length, word_count, confidence).
# 2. Logging incoming request metadata to data/current_logs.jsonl for drift detection.
# 3. Exposing Prometheus latency metrics at /metrics.
import time
import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="NLP Sentiment Model Monitoring API")

# Load model pipeline
model = joblib.load("model/model.pkl")

# Prometheus Metrics setup
LATENCY_HISTOGRAM = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint']
)
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['endpoint', 'status']
)

# Use writable /tmp directory for container runtime environments
LOG_FILE = "/tmp/current_logs.jsonl"

@app.middleware("http")
async def track_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    LATENCY_HISTOGRAM.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
    return response

@app.post("/predict")
async def predict(payload: dict):
    text = payload.get("text", "")
    
    # Extract meta-features
    text_length = len(text)
    word_count = len(text.split())
    
    # Inference
    prediction = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    confidence = float(np.max(probabilities))
    
    # Log payload for drift analyzer
    log_entry = {
        "text_length": text_length,
        "word_count": word_count,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": time.time()
    }
    
    # Ensure directory exists before writing
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return {
        "prediction": prediction,
        "confidence": confidence
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}