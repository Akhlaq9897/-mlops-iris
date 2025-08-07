import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models folder
os.makedirs("models", exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("iris_experiment")

# -----------------------------
#     LightGBM Classifier
# -----------------------------
with mlflow.start_run(run_name="LightGBM"):
    lgbm = lgb.LGBMClassifier(random_state=42)

    # Define hyperparameter space
    lgbm_param_grid = {
        'num_leaves': [15, 31, 63],
        'max_depth': [-1, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'subsample': [0.6, 0.8, 1.0]
    }

    # RandomizedSearchCV for tuning
    lgbm_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=lgbm_param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    lgbm_search.fit(X_train, y_train)
    best_lgbm = lgbm_search.best_estimator_

    preds = best_lgbm.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log results
    mlflow.log_param("model", "LightGBMClassifier")
    mlflow.log_params(lgbm_search.best_params_)
    mlflow.log_metric("accuracy", acc)

    joblib.dump(best_lgbm, "models/lightgbm_model.pkl")
    mlflow.sklearn.log_model(best_lgbm, "lightgbm_model")


# -----------------------------
#    Random Forest Classifier
# -----------------------------
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier(random_state=42)

    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    preds = best_rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log results
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_params(rf_search.best_params_)
    mlflow.log_metric("accuracy", acc)

    joblib.dump(best_rf, "models/random_forest_model.pkl")
    mlflow.sklearn.log_model(best_rf, "random_forest_model")

