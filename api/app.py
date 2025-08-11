import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime
import logging
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Patch asyncio for Jupyter
nest_asyncio.apply()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure file logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# SQLite setup
DB_PATH = "logs/predictions.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    features TEXT,
    prediction TEXT
)
""")
conn.commit()

# Prometheus metrics
REQUEST_COUNT = Counter("api_request_count", "Total API requests", ["endpoint", "status"])

# Load model
MODEL_PATH = "models/random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

rf_model = joblib.load(MODEL_PATH)

# Get feature names from training
try:
    FEATURE_COLUMNS = rf_model.feature_names_in_.tolist()
except AttributeError:
    FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# FastAPI app
app = FastAPI(
    title="Iris RandomForest API",
    description="Predict Iris species using RandomForest model",
    version="1.0"
)

# Input model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris API is running"}

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        # Prepare dataframe
        df = pd.DataFrame([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]], columns=FEATURE_COLUMNS)

        prediction = rf_model.predict(df)[0]

        # Log to file
        logging.info({
            "features": df.to_dict(orient="records")[0],
            "prediction": prediction
        })

        # Log to SQLite
        cursor.execute(
            "INSERT INTO predictions (timestamp, features, prediction) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), str(df.to_dict(orient="records")[0]), str(prediction))
        )
        conn.commit()

        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        return {"prediction": str(prediction)}

    except Exception as e:
        logging.exception("Prediction error")
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/logs")
def get_logs(limit: int = 10):
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    return {"logs": rows}

# Run inside Jupyter
uvicorn.run(app, host="127.0.0.1", port=7860)
