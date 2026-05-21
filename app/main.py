from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import mlflow.sklearn
import joblib
import os
from pathlib import Path
from mlflow.tracking import MlflowClient
import shap

app = FastAPI(title="Traffic Volume Prediction API", version="1.0.0")

class TrafficInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    temperature: float = Field(..., ge=-50, description="Temperature in Kelvin or Celsius")
    rain: float = Field(..., ge=0, description="Amount of rain in mm")
    day_of_week: int = Field(0, ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")

model = None
explainer = None
iso_forest = None

@app.on_event("startup")
def load_model():
    global model, explainer, iso_forest
    BASE_DIR = Path(__file__).resolve().parent.parent
    os.chdir(BASE_DIR) 
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = "TrafficVolumePredictor"
    
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(name=model_name)
        if latest_versions:
            latest_version = latest_versions[-1].version
            model_uri = f"models:/{model_name}/{latest_version}"
            model = mlflow.sklearn.load_model(model_uri)
            print("Model loaded successfully from MLflow registry.")
    except Exception as e:
        print(f"Failed to load from MLflow registry: {e}. Falling back to local model.")

    if model is None:
        local_model_path = os.path.join(BASE_DIR, "models", "rf_model.pkl")
        if os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
            print("Local model loaded successfully.")
        else:
            print("Warning: No model found! Please train the model first.")
            return
            
    # FIX: Initialize SHAP explainer once at startup to avoid per-request latency overhead
    try:
        explainer = shap.TreeExplainer(model)
        print("SHAP TreeExplainer initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize SHAP Explainer: {e}")
        
    # Load Anomaly Detector
    iso_path = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
    if os.path.exists(iso_path):
        iso_forest = joblib.load(iso_path)
        print("Anomaly Detector (IsolationForest) loaded successfully.")

@app.post("/predict")
def predict_traffic(data: TrafficInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    is_weekend = 1 if data.day_of_week >= 5 else 0
    hour_sin = np.sin(2 * np.pi * data.hour / 24)
    hour_cos = np.cos(2 * np.pi * data.hour / 24)

    input_df = pd.DataFrame([{
        "hour": data.hour,
        "temperature": data.temperature,
        "rain_1h": data.rain,
        "day_of_week": data.day_of_week,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos
    }])

    try:
        prediction = model.predict(input_df)[0]
        
        # Calculate Explainable AI (SHAP)
        explanation = {}
        if explainer is not None:
            try:
                shap_vals = explainer.shap_values(input_df)[0]
                base_val = explainer.expected_value
                if isinstance(base_val, np.ndarray):
                    base_val = base_val[0]
                
                # Index mapping:
                # 0: hour, 1: temperature, 2: rain_1h, 3: day_of_week, 4: is_weekend, 5: hour_sin, 6: hour_cos
                # Aggregate temporal features into hour_contribution for UI compatibility
                time_contrib = shap_vals[0] + shap_vals[3] + shap_vals[4] + shap_vals[5] + shap_vals[6]
                
                explanation = {
                    "hour_contribution": round(float(time_contrib), 2),
                    "temperature_contribution": round(float(shap_vals[1]), 2),
                    "rain_contribution": round(float(shap_vals[2]), 2),
                    "base_value": round(float(base_val), 2)
                }
            except Exception as e:
                print(f"SHAP Error during prediction: {e}")

        # Confidence Score Calculation
        confidence_score = 95.0
        if hasattr(model, 'estimators_'):
            try:
                # Get prediction from each individual tree (RandomForest fallback)
                tree_preds = [est.predict(input_df.values)[0] for est in model.estimators_]
                std_dev = float(np.std(tree_preds))
                confidence_score = max(20.0, min(99.9, 100.0 - (std_dev / 15.0)))
            except Exception as e:
                print(f"Confidence calc error: {e}")
        elif hasattr(model, 'get_booster'):
            try:
                # XGBoost specific confidence calculation:
                # Predict using standard deviation of predictions of subsets of estimators
                preds_stages = []
                for i in range(1, 6):
                    limit = max(1, int(model.n_estimators * (i / 5.0)))
                    preds_stages.append(model.predict(input_df, iteration_range=(0, limit))[0])
                std_dev = float(np.std(preds_stages))
                confidence_score = max(20.0, min(99.9, 100.0 - (std_dev / 15.0)))
            except Exception as e:
                print(f"XGBoost confidence calc error: {e}")
                confidence_score = 95.0

        # Anomaly Detection
        is_anomaly = False
        if iso_forest is not None:
            try:
                outlier_pred = iso_forest.predict(input_df)[0]
                if outlier_pred == -1:
                    is_anomaly = True
                    # Force confidence to drop massively if anomaly
                    confidence_score = min(confidence_score, 35.0)
            except Exception as e:
                print(f"Anomaly detection error: {e}")

        return {
            "input": {
                "hour": data.hour,
                "temperature": data.temperature,
                "rain": data.rain,
                "day_of_week": data.day_of_week
            },
            "predicted_traffic_volume": round(float(prediction), 2),
            "explanation": explanation,
            "is_anomaly": is_anomaly,
            "confidence_score": round(confidence_score, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def root():
    BASE_DIR = Path(__file__).resolve().parent
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()
