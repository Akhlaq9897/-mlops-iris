import nest_asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Patch asyncio for Jupyter
nest_asyncio.apply()

# Load model
MODEL_PATH = "models/random_forest_model.pkl" #loading random forest as it performed better
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

# Match your JSON input exactly
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
    df = pd.DataFrame([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]], columns=FEATURE_COLUMNS)

    prediction = rf_model.predict(df)[0]
    return {"prediction": str(prediction)}

# Run inside Jupyter
uvicorn.run(app, host="0.0.0.0", port=7860)
