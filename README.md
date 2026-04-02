# Local MLOps Platform Implementation

This repository hosts an end-to-end local Machine Learning Operations (MLOps) platform tailored for demonstrating robust engineering patterns. Built around a simulated credit card fraud detection use case, this project progressively implements critical MLOps capabilities, graduating from basic "naive" approaches to a production-ready ecosystem featuring data versioning, feature stores, experiment tracking, model registries, data validation, and monitoring pipelines.

## 🏗️ Architecture & Stack

The platform integrates several industry-standard open-source tools:

- **MLflow**: Experiment Tracking & Model Registry.
- **DVC (Data Version Control)**: Artifact storage and dataset versioning.
- **Feast**: Offline and Online Feature Store for consistent feature engineering.
- **Great Expectations**: Data validation and pipeline guards.
- **Evidently AI**: Data drift and model performance monitoring.
- **Docker Compose**: Containerized backend infrastructure (Postgres + MinIO S3).
- **FastAPI**: Low-latency model serving framework.

## 📂 Repository Structure

The core codebase is located inside the `ml-platform-tutorial` directory.

```text
Mlops_local_platform/
├── ml-platform-tutorial/
│   ├── .dvc/                   # DVC configuration
│   ├── data/                   # Datasets (Tracked by DVC, ignored by Git)
│   ├── feature_repo/           # Feast feature store definitions
│   ├── models/                 # Serialized basic models
│   ├── src/                    # Main logic and scripts
│   │   ├── generate_data.py            # Generates synthetic transaction data
│   │   ├── train_naive.py              # Baseline local training
│   │   ├── train_mlflow.py             # Advanced training logging to MLflow
│   │   ├── prepare_feast_features.py   # Feast online/offline processing
│   │   ├── data_validation.py          # Data quality checks
│   │   ├── monitoring.py               # Evidently AI drift detection
│   │   ├── serve_naive.py              # Basic FastAPI model server
│   │   ├── serve_mlflow.py             # Server pulling from MLflow Registry
│   │   └── serve_validated.py          # Server with strict input validation
│   └── tests/                  # Pytest unit and integration tests
└── README.md
```

## 🚀 Environment Setup

### 1. MLflow & Storage Infrastructure (Docker)
This repository includes a fully self-contained Dockerized infrastructure (`mlflow-docker-infra/`), composed of a PostgreSQL database strictly for experiment metadata and a MinIO (S3-compatible) container for model artifact storage.

To launch the backend infrastructure:
```bash
cd mlflow-docker-infra
docker-compose up -d
```
*Wait locally for the MLflow UI (`localhost:5001`) and MinIO Console (`localhost:9001`) to come online.*

### 2. Python Environment

Create a virtual environment and install the dependencies.

```bash
cd ml-platform-tutorial
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. DVC Integration (Data Version Control)
The datasets for this repository are tracked via a separate standalone DVC local remote. To fetch the data into your workspace locally:
```bash
cd ml-platform-tutorial
dvc pull
```

### 4. Client Environment Variables
Set your environment variables to point your training scripts to the running backend infrastructure:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=mlflow_password
```

## 🧠 Workflows

### Data Generation & Feature Engineering
1. **Generate Raw Data**: `python src/generate_data.py`
2. **Compute & Ingest Features**: `python src/prepare_feast_features.py` (Outputs to Parquet and Feast's local Registry/Online DB).

### Model Training
You can run a naive training setup, or use the integrated MLflow training script:
```bash
# Naive (Saves simple .pkl locally)
python src/train_naive.py

# MLflow (Logs parameters, metrics, and models to our Docker backend)
python src/train_mlflow.py
```

### Model Serving
Spin up the FastAPI server to serve predictions.
```bash
# Serves the model registered directly in MLflow
python src/serve_mlflow.py
# OR load with strict input validation
python src/serve_validated.py
```

### Validation & Monitoring
1. **Validate schema & incoming structures**: `python src/data_validation.py`
2. **Check for data drift**: `python src/monitoring.py` (Generates `drift_report.html`)

## 💡 Key Takeaways
- **No Model Silos**: The combination of MinIO + Postgres effectively removes the bottleneck of local filesystem sqlite `mlruns`.
- **Reproducibility**: DVC tracks `train.csv`, `test.csv`, and `merchant_features.parquet` providing absolute control over training history.
- **Consistent Inference**: Feast guarantees that features calculated offline during training are served synchronously online, banishing offline-online skew.