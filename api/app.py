from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model (use the best one from Part 2)
model = joblib.load("models/random_forest_model.pkl")

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input: IrisInput):
    features = np.array([[input.sepal_length, input.sepal_width,
                          input.petal_length, input.petal_width]])
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
