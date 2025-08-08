from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Lazy-load the trained model so the container can start without the artifact
_model = None


def get_model():
    global _model
    if _model is None:
        model_path = os.getenv("MODEL_PATH", "models/lightgbm_model.pkl")
        _model = joblib.load(model_path)
    return _model

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "sepal_length": 6.7,
                    "sepal_width": 3.1,
                    "petal_length": 4.7,
                    "petal_width": 1.5
                }
            ]
        }

@app.post("/predict")
def predict(input: IrisInput):
    features = np.array([[input.sepal_length, input.sepal_width,
                          input.petal_length, input.petal_width]])
    prediction = get_model().predict(features)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)