import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import warnings

# Suppress LightGBM warnings for cleaner output
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# Load data
df = pd.read_csv("data/Iris.csv")

# Clean column names by removing whitespace and special characters
df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for col in df.columns]

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
    # Configure LightGBM with better defaults for small datasets
    lgbm = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,  # Suppress warnings
        force_col_wise=True,  # Force column-wise to avoid overhead warnings
        min_child_samples=20,  # Prevent overfitting on small dataset
        min_split_gain=0.1,  # Minimum gain to make split
    )

    # Define more reasonable hyperparameter space for iris dataset
    lgbm_param_grid = {
        'num_leaves': [10, 20, 31],  # Reduced for small dataset
        'max_depth': [3, 5, 7],  # More reasonable depths, removed -1
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],  # Simplified
        'min_child_samples': [10, 20, 30]  # Added to prevent overfitting
    }

    # RandomizedSearchCV for tuning with better configuration
    lgbm_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=lgbm_param_grid,
        n_iter=5,  # Reduced iterations for faster training
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=1,  # Use single job to avoid conflicts
        verbose=0  # Suppress verbose output
    )

    try:
        print("Training LightGBM model...")
        lgbm_search.fit(X_train, y_train)
        best_lgbm = lgbm_search.best_estimator_

        preds = best_lgbm.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"LightGBM best params: {lgbm_search.best_params_}")
        print(f"LightGBM accuracy: {acc:.4f}")

        # Log results
        mlflow.log_param("model", "LightGBMClassifier")
        mlflow.log_params(lgbm_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        joblib.dump(best_lgbm, "models/lightgbm_model.pkl")
        
        # Log model with signature and input example
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, best_lgbm.predict(X_train))
        mlflow.sklearn.log_model(
            best_lgbm, 
            name="lightgbm_model",  # Use name parameter instead of deprecated artifact_path
            signature=signature,
            input_example=X_train.iloc[:5]  # Provide input example
        )
        
    except Exception as e:
        print(f"Error training LightGBM model: {str(e)}")
        raise


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
        n_iter=5,  # Reduced for faster training
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=1,  # Use single job for consistency
        verbose=0
    )

    try:
        print("Training Random Forest model...")
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_

        preds = best_rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Random Forest best params: {rf_search.best_params_}")
        print(f"Random Forest accuracy: {acc:.4f}")

        # Log results
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_params(rf_search.best_params_)
        mlflow.log_metric("accuracy", acc)

        joblib.dump(best_rf, "models/random_forest_model.pkl")
        
        # Log model with signature and input example
        signature = infer_signature(X_train, best_rf.predict(X_train))
        mlflow.sklearn.log_model(
            best_rf,
            name="random_forest_model",  # Use name parameter instead of deprecated artifact_path
            signature=signature,
            input_example=X_train.iloc[:5]  # Provide input example
        )
        
    except Exception as e:
        print(f"Error training Random Forest model: {str(e)}")
        raise

print("Training completed successfully!")

