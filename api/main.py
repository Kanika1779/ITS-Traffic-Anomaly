from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.pytorch.bilstm import AGTSFNet

# ── 1. App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AGTSF-Net Traffic Anomaly Detection API",
    description="Real-time Intelligent Traffic Monitoring using Dual-Stream BiLSTM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── 2. Load Model ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AGTSFNet(temporal_input_size=8, spatial_input_size=8)
model.load_state_dict(
    torch.load(
        "experiments/results/best_model.pt",
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# Feature names
FEATURES = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h',
            'clouds_all', 'hour', 'dayofweek', 'month']

# In-memory buffer to simulate streaming
stream_buffer = []


# ── 3. Helper: Run Prediction ─────────────────────────────────────────────────
def run_prediction(window: list):
    X = np.array(window, dtype=np.float32)
    X = torch.tensor(X).unsqueeze(0).to(device)  # (1, 60, 8)
    with torch.no_grad():
        logits, attn = model(X, X)
        prob = torch.sigmoid(logits).item()
    is_anomaly = prob > 0.5
    severity   = "critical" if prob > 0.85 else "medium" if prob > 0.65 else "low"
    return {
        "anomaly_probability": round(prob, 4),
        "is_anomaly": is_anomaly,
        "severity": severity if is_anomaly else "none",
        "attention_weights": attn[0, -1, :].cpu().tolist()[:10]
    }


# ── 4. Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AGTSF-Net Traffic Anomaly Detection API",
        "status": "running",
        "device": str(device)
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "AGTSF-Net", "device": str(device)}


@app.post("/predict")
def predict(data: dict):
    """
    Accepts a window of 60 timesteps with 8 features each.
    Returns anomaly prediction + severity + attention weights.
    """
    try:
        window = data.get("window")
        if not window or len(window) != 60:
            return {"error": "window must have exactly 60 timesteps"}
        if len(window[0]) != 8:
            return {"error": "each timestep must have exactly 8 features"}

        result = run_prediction(window)
        return {"status": "success", "prediction": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/stream")
def stream_predict(data: dict):
    """
    Accepts a single timestep, buffers it,
    and returns prediction when buffer reaches 60.
    """
    global stream_buffer
    try:
        timestep = data.get("timestep")
        if not timestep or len(timestep) != 8:
            return {"error": "timestep must have exactly 8 features"}

        stream_buffer.append(timestep)

        if len(stream_buffer) < 60:
            return {
                "status": "buffering",
                "buffer_size": len(stream_buffer),
                "message": f"Need {60 - len(stream_buffer)} more timesteps"
            }

        # Keep only last 60
        window = stream_buffer[-60:]
        result = run_prediction(window)
        return {"status": "success", "prediction": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/buffer/reset")
def reset_buffer():
    global stream_buffer
    stream_buffer = []
    return {"status": "buffer reset"}


# ── 5. WebSocket for Live Dashboard ──────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected!")
    buffer = []
    try:
        while True:
            data = await websocket.receive_text()
            timestep = json.loads(data)

            buffer.append(timestep)

            if len(buffer) >= 60:
                window  = buffer[-60:]
                result  = run_prediction(window)
                await websocket.send_text(json.dumps({
                    "status": "prediction",
                    "prediction": result
                }))
            else:
                await websocket.send_text(json.dumps({
                    "status": "buffering",
                    "buffer_size": len(buffer)
                }))
    except WebSocketDisconnect:
        print("WebSocket client disconnected")