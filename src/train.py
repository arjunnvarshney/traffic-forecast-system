import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from mlflow.models.signature import infer_signature
import joblib
import optuna
from optuna.samplers import TPESampler

def train_model(data_path: str, model_dir: str):
    if not os.path.exists(data_path):
        print(f"Error: Cleaned data not found at {data_path}.")
        return

    os.makedirs(model_dir, exist_ok=True)

    print("Loading cleaned dataset...")
    df = pd.read_csv(data_path)
    
    features = ['hour', 'temperature', 'rain_1h', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']
    target = 'traffic_volume'
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save test set to disk to prevent data leakage in evaluate.py
    base_dir = Path(data_path).parent
    X_test.to_csv(os.path.join(base_dir, "test_features.csv"), index=False)
    y_test.to_csv(os.path.join(base_dir, "test_target.csv"), index=False)
    print("Test datasets saved for isolated evaluation.")

    print("Training Anomaly Detection Engine (Isolation Forest)...")
    from sklearn.ensemble import IsolationForest
    # Assuming 1% of our historic data represents true extreme anomalies
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    iso_path = os.path.join(model_dir, "isolation_forest.pkl")
    joblib.dump(iso_forest, iso_path)
    print(f"Anomaly detector saved to {iso_path}")

    print("Running Hyperparameter Tuning with Optuna...")
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        
        model = XGBRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=50) 
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("traffic_volume_prediction")

    with mlflow.start_run() as run:
        mlflow.log_params(best_params)

        print("Training Optimized XGBRegressor...")
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print("--- Evaluation Metrics ---")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        signature = infer_signature(X_train, model.predict(X_train))

        print("Registering model in MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model", # Keep path name compatible
            signature=signature,
            registered_model_name="TrafficVolumePredictor"
        )
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "rf_model.pkl") # Keep file name compatible
        joblib.dump(model, model_path)
        print(f"Model successfully saved locally at {model_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_traffic.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.chdir(BASE_DIR)
    train_model(CLEANED_DATA_PATH, MODEL_DIR)
