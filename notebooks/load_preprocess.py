from sklearn.datasets import load_iris
import pandas as pd
import os

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Save to CSV in data/ folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/Iris.csv", index=False)

print("Saved preprocessed dataset to data/iris.csv")
