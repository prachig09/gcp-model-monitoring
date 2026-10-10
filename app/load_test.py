#client tool
# 1.  It sends real HTTP traffic to deployed Cloud Run URL in two distinct batches (Normal Movie Reviews vs. Shifted Product Reviews) and 
# 2. generate load for latency metrics and log incoming inference payloads.
import requests
import time

# REPLACE THIS WITH YOUR DEPLOYED CLOUD RUN URL
SERVICE_URL = "https://gcp-model-monitoring-rk6x4plvmq-uc.a.run.app/predict"
PREDICT_URL = f"{SERVICE_URL}/predict"

# Reference Window Traffic: Movie/Longer Reviews (Matching Training Domain)
movie_traffic = [
    {"text": "The cinematography and acting performances were extraordinary. A deeply moving film experience from start to finish."},
    {"text": "A complete waste of time with a predictable storyline, awful character arcs, and poor direction throughout."},
    {"text": "An absolute masterpiece that balances compelling storytelling with stunning visual design and a memorable soundtrack."},
    {"text": "Boring dialogue and sluggish pacing made this movie almost impossible to finish without falling asleep."}
] * 25  # 100 requests

# Shifted Window Traffic: Product/Short Reviews (Introducing Domain Drift)
product_traffic = [
    {"text": "Broke in 2 days. Cheap plastic build quality."},
    {"text": "Great battery life! Charges quickly and lasts all day."},
    {"text": "DO NOT BUY. Device overheats within ten minutes of use."},
    {"text": "Five stars, fits perfectly on my desk setup and looks sleek."}
] * 25  # 100 requests

print("--- Phase 1: Sending Reference Window Traffic (Movie Reviews Domain) ---")
for i, item in enumerate(movie_traffic, 1):
    try:
        res = requests.post(PREDICT_URL, json=item)
        if i % 20 == 0:
            print(f"Sent {i}/100 reference requests - Status: {res.status_code}")
    except Exception as e:
        print(f"Error sending request: {e}")
    time.sleep(0.05)

print("\n--- Phase 2: Injecting Domain Drift Traffic (Short Product Reviews Domain) ---")
for i, item in enumerate(product_traffic, 1):
    try:
        res = requests.post(PREDICT_URL, json=item)
        if i % 20 == 0:
            print(f"Sent {i}/100 drifted requests - Status: {res.status_code}")
    except Exception as e:
        print(f"Error sending request: {e}")
    time.sleep(0.05)

print("\nLoad generation and drift injection complete!")