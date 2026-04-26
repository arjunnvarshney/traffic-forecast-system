# Enterprise MLOps: Traffic Volume Prediction Pipeline

## 🚀 Project Overview
This project is a production-grade, end-to-end MLOps pipeline designed to predict hourly traffic volume based on temporal and weather features. It is built to simulate a FAANG-level machine learning system, featuring robust data validation, hyperparameter tuning, model registry integration, statistical data drift monitoring, and Explainable AI (XAI).

## 🧠 Architecture & Tech Stack
* **Machine Learning Engine:** `scikit-learn` (RandomForestRegressor & IsolationForest)
* **Hyperparameter Optimization:** `optuna` (TPESampler)
* **Experiment Tracking & Model Registry:** `mlflow` (SQLite backend, configurable via environment variables)
* **Explainable AI:** `shap` (TreeExplainer for real-time feature contribution analysis)
* **API Serving:** `fastapi` & `uvicorn`
* **Frontend UI:** Vanilla JS + `Chart.js` (Glassmorphism Cyberpunk Dashboard)
* **Orchestration & Scheduling:** Python `subprocess` & `schedule` (with PID-based file locking)
* **CI/CD & DevOps:** GitHub Actions & Docker (`.dockerignore` enforced statelessness)

---

## 🛠️ Core Components & Features

### 1. Robust Data Preprocessing & Validation (`src/preprocess.py`)
* Extracts relevant features (`hour`, `temperature`, `rain_1h`) and target variable.
* Drops missing values to ensure data integrity.
* **Schema Enforcement:** Strict assertion rules guarantee that temperature is physically possible (>= -50), rainfall is non-negative, and hours remain within 0-23 bounds.

### 2. Automated Hyperparameter Tuning & Training (`src/train.py`)
* **Optuna Integration:** Conducts 50 trials using the mathematically superior `TPESampler` to hunt for the absolute best `n_estimators`, `max_depth`, and `min_samples_split`.
* **Out-of-Distribution Safety (Anomaly Engine):** Trains an `IsolationForest` on the training distribution (assuming 1% historic anomaly rate) to act as a security guard against crazy inputs in production.
* **MLflow Tracking:** Logs all parameters, metrics, and models to a centralized tracking server.
* **Data Leakage Prevention:** Strictly isolates and saves `X_test` and `y_test` to disk rather than passing them in memory, guaranteeing the evaluator never touches training logic.

### 3. Model Evaluation & Champion/Challenger Promotion (`src/evaluate.py`)
* Loads the rigorously isolated test dataset from disk to guarantee objective grading.
* Evaluates using MAE, RMSE, and R² Score.
* **Champion vs Challenger Promotion:** Actively queries the MLflow Registry for the previously deployed "Production" model (Champion). Evaluates both the Champion and the newly trained model (Challenger) on the *exact same* unseen test set. If the Challenger's R² is not strictly greater than the Champion's, the promotion is forcefully rejected and the pipeline aborts.
* **Quality Gate:** Reads a configurable `MIN_R2_THRESHOLD` from the environment. If the model degrades below this threshold, it triggers a `sys.exit(1)`, instantly blocking any CI/CD GitHub Action from deploying the regressed model.

### 4. Data Drift Monitoring (`src/monitor.py`)
* Uses the **Kolmogorov-Smirnov (KS) Test** to statistically compare the distribution of incoming live data against the original reference training data.
* Intelligently ignores cyclic features (like `hour`) which naturally fluctuate daily, focusing only on genuine shifts in weather or volume behavior.

### 5. Self-Healing Orchestration (`run_pipeline.py` & `scheduler.py`)
* Orchestrates the entire lifecycle: Preprocess ➡️ Train ➡️ Evaluate ➡️ Monitor.
* Runs on a cron-like schedule (Every 12 hours / 2:00 AM).
* **Fault Tolerance:** Uses `psutil` to write and verify Process ID (PID) locks. If the server hard-crashes mid-training, the scheduler dynamically detects the stale PID, deletes the lock file, and resumes operations seamlessly.

---

## 🌐 Production Serving & Frontend (`app/main.py` & `app/index.html`)

### The API (FastAPI)
* Loads the latest registered `TrafficVolumePredictor` model directly from the MLflow Registry.
* **Memory Optimization:** Instantiates the heavy `shap.TreeExplainer` exactly once during the server `@startup` event, allowing per-request predictions to execute in milliseconds.
* **Anomaly Guard:** Passes incoming user requests through the `IsolationForest`. If the input is mathematically absurd, it flags it as an anomaly.

### The Mega-Dashboard (Frontend UI)
* **Real-Time Forecasting:** Uses `Chart.js` to shoot 7 parallel background API requests, building a dynamic visual forecast of the next 7 hours.
* **Natural Language AI Insights:** Instead of just showing raw SHAP values (+47), the JavaScript analyzes the SHAP math and generates a plain-English explanation (e.g., *"At 4:00, traffic is significantly lower than average because it is an off-peak or late-night hour."*).
* **Live System Logs:** A hacker-style terminal streams the actual JavaScript/API execution logs in real-time.
* **Congestion Badge:** Dynamically lights up Green (Flowing), Orange (Moderate), or Red (Severe) based on exact predicted volume.
* **Anomaly Alert System:** Listens for the API's anomaly flag and flashes a massive red warning banner if Out-Of-Distribution data is detected.

---

## 📦 Containerization & Deployment
* **Stateless Dockerization:** A strict `.dockerignore` prevents local `models/` and `mlflow.db` from being baked into the image.
* **Environment Configuration:** Relies heavily on `os.getenv()` allowing the MLflow Tracking URI and Quality Gate thresholds to be dynamically injected by Cloud Providers (AWS, Render, GCP) without editing Python code.
