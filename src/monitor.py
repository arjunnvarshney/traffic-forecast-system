import pandas as pd
from scipy.stats import ks_2samp
import os
from pathlib import Path

def detect_drift(reference_data_path, current_data_path):
    print("Running Data Drift Detection...")
    if not os.path.exists(reference_data_path) or not os.path.exists(current_data_path):
        print("Error: Data files missing for drift detection.")
        return

    ref_df = pd.read_csv(reference_data_path)
    cur_df = pd.read_csv(current_data_path)

    features = ['hour', 'temperature', 'rain_1h', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']
    drift_detected = False

    for feature in features:
        if feature in ref_df.columns and feature in cur_df.columns:
            # Skip cyclic / temporal features as their variations are normal and expected
            if feature in ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']:
                continue
                
            # Kolmogorov-Smirnov test to compare distributions
            # Threshold p_value < 0.05 indicates 95% confidence that distributions differ significantly
            stat, p_value = ks_2samp(ref_df[feature], cur_df[feature])
            if p_value < 0.05:
                print(f"⚠️ Drift detected in feature '{feature}' (p-value: {p_value:.4f})")
                drift_detected = True
            else:
                print(f"✅ No drift in feature '{feature}' (p-value: {p_value:.4f})")

    if drift_detected:
        print("🚨 Data drift detected! Automated retraining workflow should be triggered.")
    else:
        print("✅ Data distribution is stable.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    # In a real scenario, CUR_DATA would be the newest data batch from production
    # For demonstration, we compare the cleaned data to itself (which will show no drift)
    REF_DATA = os.path.join(BASE_DIR, "data", "cleaned_traffic.csv")
    CUR_DATA = os.path.join(BASE_DIR, "data", "cleaned_traffic.csv") 
    
    detect_drift(REF_DATA, CUR_DATA)
