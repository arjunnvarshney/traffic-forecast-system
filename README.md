# 🚦 Enterprise Traffic Volume Prediction & MLOps Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.95%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-v2.0%2B-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-v20%2B-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

An end-to-end, production-grade MLOps ecosystem designed for high-precision urban mobility forecasting. This project moves beyond simple modeling into **Enterprise MLOps**, featuring automated hyperparameter optimization, Explainable AI (XAI), and a robust Champion/Challenger model registry strategy.

---

##  Key Enterprise Features

###  1. Explainable AI (XAI) with SHAP
Don't just predict numbers—understand the "why." Our engine integrates **SHAP (SHapley Additive exPlanations)** to break down exactly how much the time of day, temperature, or rainfall contributed to each specific traffic prediction.

###  2. Anomaly Detection Guardrail
A secondary **Isolation Forest** model acts as a security guard. If a user inputs data that is statistically out-of-distribution (e.g., impossible weather for the time of day), the system flags it as an anomaly and alerts the operator.

###  3. Champion vs. Challenger Registry
A strict model promotion strategy. New models (**Challengers**) must mathematically outperform the current production model (**Champion**) on an isolated test set before they are allowed to be promoted to "Production" status in the MLflow Registry.

###  4. Automated Hyperparameter Tuning
Integrated **Optuna** engine that runs dozens of trials to find the perfect architecture for the RandomForest ensemble, optimizing for $R^2$ score and generalization.

###  5. Real-time Monitoring & Dashboard
A high-fidelity, cyberpunk-themed dashboard featuring:
- **Live Terminal Logs**: Real-time backend system status.
- **Dynamic Chart.js Forecasts**: Visualizing the next 7 hours of traffic.
- **Confidence Scoring**: Measuring the mathematical variance of the forest ensemble.

---

##  System Architecture

1.  **Data Ingestion**: A custom physics-based synthetic data generator mimics real-world traffic patterns (rush hours, weather correlations) to provide high-quality $R^2$ scores (~96%).
2.  **Training Pipeline**: Optuna-driven hyperparameter search logged via MLflow Tracking.
3.  **Evaluation Gate**: Automated comparison of Challenger vs. Champion models.
4.  **Serving Layer**: FastAPI backend with cached TreeExplainers for low-latency inference.
5.  **Monitoring**: Drift detection and anomaly monitoring on every request.

---

##  Project Structure

```text
traffic-forecast-engine/
├── .github/workflows/    # CI/CD - Automated MLOps Pipeline
├── app/                  # Frontend & Serving Layer
│   ├── index.html        # High-Fidelity Cyberpunk Dashboard
│   └── main.py          # FastAPI Application Logic (XAI + Anomaly)
├── data/                 # Raw/Cleaned Dataset Artifacts
├── models/               # Serialized ML Artifacts (RF, IsolationForest)
├── src/                  # Core MLOps Logic
│   ├── preprocess.py     # Physics-based Data Generation & Cleaning
│   ├── train.py          # Optuna Training & MLflow Logging
│   ├── evaluate.py       # Champion/Challenger Promotion Logic
│   └── monitor.py        # Statistical Drift Detection
├── requirements.txt      # System Dependencies
└── Dockerfile            # Containerization Config
```

---

##  Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/[YOUR_USERNAME]/traffic-forecast-system.git
cd traffic-forecast-system

# Set up virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the MLOps Pipeline
Execute the full automated workflow from ingestion to promotion:
```bash
python run_pipeline.py
```

### 3. Launch the Dashboard
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000` to interact with the engine.

---


