#client tool
# 1.  It sends real HTTP traffic to deployed Cloud Run URL in two distinct batches (Normal Movie Reviews vs. Shifted Product Reviews) and 
# 2. generate load for latency metrics and log incoming inference payloads.
import requests
import json
import os
import time

# Cloud Run Service Endpoint
SERVICE_URL = "https://gcp-model-monitoring-rk6x4plvmq-uc.a.run.app"
PREDICT_URL = f"{SERVICE_URL}/predict"

# Local output path for log aggregation
os.makedirs("data", exist_ok=True)
log_file = "data/current_logs.jsonl"

# Clear old local logs before fresh run
if os.path.exists(log_file):
    os.remove(log_file)

def send_payloads(texts, label):
    print(f"--- Sending {label.upper()} Window Traffic ---")
    for idx, text in enumerate(texts, 1):
        try:
            res = requests.post(PREDICT_URL, json={"text": text})
            if res.status_code == 200:
                data = res.json()
                log_entry = {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "prediction": data["prediction"],
                    "confidence": data["confidence"],
                    "window": label,
                    "timestamp": time.time()
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            
            if idx % 20 == 0 or idx == len(texts):
                print(f"Sent {idx}/{len(texts)} {label} requests - Status: {res.status_code}")
        except Exception as e:
            print(f"Error sending request: {e}")

if __name__ == "__main__":
    # Reference Window: Movie Reviews (In-Domain Baseline)
    ref_texts = [
        "This movie was fantastic, amazing plot and great acting!",
        "Loved the film! Excellent direction and brilliant performance.",
        "A true masterpiece, highly recommended for everyone.",
        "Wonderful experience, thoroughly enjoyed watching it.",
        "Terrible waste of time, awful acting and boring script.",
        "Horrible product, worst film ever, completely disappointed.",
        "Bad movie, poorly directed and dreadful pacing.",
        "I hated this, completely unwatchable and waste of money."
    ] * 12  # 96 requests

    # Current Window: Product Reviews (Domain Shift)
    curr_texts = [
        "The battery life on this laptop is completely abysmal and drains in an hour.",
        "Fast delivery, item arrived in perfect condition and works as expected.",
        "Synthesizer keys feel cheap, knob broke off after two days of light use.",
        "Super comfortable shoes, fits perfectly and great material quality.",
        "Software keeps crashing on startup, zero support from the vendor.",
        "Extremely durable backpack with plenty of compartments, highly recommend.",
        "Flimsy cable, stopped charging after a single week of normal usage.",
        "Excellent noise cancellation and crisp sound profile, worth every penny."
    ] * 12  # 96 requests

    send_payloads(ref_texts, "reference")
    send_payloads(curr_texts, "current")
    print(f"\nTraffic test completed! Local log generated at: {log_file}")