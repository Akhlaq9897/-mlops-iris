import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os

# Load dataset
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models/ folder if needed
os.makedirs("models", exist_ok=True)

mlflow.set_experiment("iris_experiment")

# Train Logistic Regression
with mlflow.start_run(run_name="LogisticRegression"):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    joblib.dump(lr, "models/logistic_model.pkl")
    mlflow.sklearn.log_model(lr, "logistic_model")

# Train Random Forest
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)

    joblib.dump(rf, "models/random_forest_model.pkl")
    mlflow.sklearn.log_model(rf, "random_forest_model")
