# Traffic Volume Prediction MLOps

A complete end-to-end MLOps project for predicting traffic volume based on time and weather features. 

## Project Overview

This project demonstrates a production-style machine learning pipeline using:
- **Scikit-learn** for training a Random Forest Regressor model.
- **MLflow** for experiment tracking and model registry.
- **FastAPI** for real-time model deployment.
- **Docker** for containerizing the complete application.

## Architecture

The workflow consists of the following stages:
1. **Data Preprocessing** (`src/preprocess.py`): Loads raw CSV data, cleans it, handles missing values, extracts the `hour` feature from the datetime column, and selects relevant features (`hour`, `temperature`, `rain_1h`).
2. **Model Training** (`src/train.py`): Splits data into train and test sets, trains a `RandomForestRegressor`, logs hyperparameters and metrics (MAE, RMSE, R2) to MLflow, and registers the trained model.
3. **Model Evaluation** (`src/evaluate.py`): Loads the saved model and evaluates it against the test dataset to print accuracy metrics.
4. **Model Deployment** (`app/main.py`): A FastAPI application that loads the latest version of the model directly from the MLflow model registry (with local file fallback) and exposes a `/predict` endpoint.

## Tech Stack
- **Languages:** Python 3.9
- **Machine Learning:** Scikit-learn, Pandas, Numpy
- **MLOps:** MLflow
- **Web Framework:** FastAPI, Uvicorn
- **Containerization:** Docker

## Folder Structure
```text
traffic_prediction_mlops/
│
├── data/                  # Raw and cleaned data
├── models/                # Locally saved model artifacts
├── src/                   # Source scripts
│   ├── preprocess.py      
│   ├── train.py           
│   └── evaluate.py        
├── app/                   # FastAPI application
│   └── main.py            
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## How to Run Locally

### 1. Set Up Environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data and Train Model
Run the pipeline scripts in order from the project root:
```bash
# 1. Preprocess data (will create dummy data if raw traffic data is missing)
python src/preprocess.py

# 2. Train model and log to MLflow
python src/train.py

# 3. Evaluate the saved model
python src/evaluate.py
```

### 3. MLflow Tracking & Model Registry
To view the MLflow UI and check your experiments, metrics, and registered models, run:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000` in your browser. Here you can see your experiments, logged metrics, parameters, and the registered model versions under the **Models** tab.

### 4. Run FastAPI Server
Start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Visit `http://127.0.0.1:8000/docs` to test the `/predict` API using the Swagger interactive documentation interface.

## How to Run with Docker

To containerize the application and run it anywhere:

1. **Build the Docker Image:**
```bash
docker build -t traffic-mlops .
```

2. **Run the Docker Container:**
```bash
docker run -p 8000:8000 traffic-mlops
```

Once running, the API will be available at `http://localhost:8000/docs`. 

*Note: Since the MLflow sqlite database and local model files are created during the local training process, ensure you train the model locally (`python src/train.py`) before building the Docker image so the trained artifacts get copied inside the container.*
