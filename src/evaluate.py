import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys
from pathlib import Path

def evaluate_model(model_path: str, test_features_path: str, test_target_path: str):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return
        
    if not os.path.exists(test_features_path) or not os.path.exists(test_target_path):
        print(f"Error: Isolated test data not found. Please run train.py first to generate test splits.")
        return

    print("Loading model and isolated test data...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_target_path)

    # Ensure y_test is a 1D array/series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("--- Model Evaluation on Test Data ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print("-------------------------------------")

    # CI/CD Quality Gate: Fail pipeline if R2 drops below a certain threshold
    MIN_R2_THRESHOLD = float(os.getenv("MIN_R2_THRESHOLD", "-1.0"))
    if r2 < MIN_R2_THRESHOLD:
        print(f"🚨 QUALITY GATE FAILED: R2 Score {r2:.2f} is below acceptable threshold of {MIN_R2_THRESHOLD}")
        sys.exit(1)
    else:
        print("✅ Quality Gate Passed.")

    # Champion vs Challenger MLflow Promotion Logic
    try:
        from mlflow.tracking import MlflowClient
        import mlflow.sklearn
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        
        client = MlflowClient()
        model_name = "TrafficVolumePredictor"
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if len(versions) > 1:
            # Sort by version number to get the champion (second to last)
            versions = sorted(versions, key=lambda v: int(v.version))
            previous_version = versions[-2].version
            
            champion_uri = f"models:/{model_name}/{previous_version}"
            print(f"\nFetching Champion Model (Version {previous_version}) from MLflow Registry...")
            champion_model = mlflow.sklearn.load_model(champion_uri)
            
            champion_preds = champion_model.predict(X_test)
            champion_r2 = r2_score(y_test, champion_preds)
            
            print("--- Champion/Challenger Comparison ---")
            print(f"🏆 Champion R2 (v{previous_version}): {champion_r2:.4f}")
            print(f"⚔️ Challenger R2 (Latest): {r2:.4f}")
            
            if r2 <= champion_r2:
                print("❌ Challenger failed to outperform Champion. Rejecting promotion.")
                # Rejecting by failing the pipeline so CD doesn't deploy it
                sys.exit(1)
            else:
                print("✅ Challenger wins! Promotion to Production confirmed.")
        else:
            print("\nFirst model version registered. Establishing as initial Champion.")
            
    except Exception as e:
        print(f"Champion comparison skipped or failed: {e}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    TEST_FEATURES_PATH = os.path.join(BASE_DIR, "data", "test_features.csv")
    TEST_TARGET_PATH = os.path.join(BASE_DIR, "data", "test_target.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
    evaluate_model(MODEL_PATH, TEST_FEATURES_PATH, TEST_TARGET_PATH)
