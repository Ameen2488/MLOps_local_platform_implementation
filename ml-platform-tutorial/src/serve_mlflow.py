# src/serve_mlflow.py
"""
Serve fraud detection model from MLflow Registry.

This version loads the champion model from MLflow Model Registry
instead of from a local pickle file.
"""
import mlflow
import mlflow.sklearn
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Configure MLflow
MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "fraud-detection-model"
ALIAS = "champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Loading model from MLflow Registry...")

try:
    # Load model from registry using the champion alias
    # If champion alias doesn't exist, load the latest version
    try:
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from alias: {ALIAS}")
    except Exception:
        # Fallback to latest version
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
        model_uri = f"models:/{MODEL_NAME}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model version: {latest_version}")
    
    # Load the encoder from the same run's artifacts
    # Get the latest run's encoder
    client = mlflow.MlflowClient()
    latest_run = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    encoder_path = client.download_artifacts(latest_run.run_id, "encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    encoder = None

app = FastAPI(
    title="Fraud Detection API (MLflow)",
    description="API to predict credit card fraud using model from MLflow Registry",
    version="2.0.0"
)

class TransactionRequest(BaseModel):
    amount: float = Field(..., example=120.50, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, example=14, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="Day (0=Mon, 6=Sun)")
    merchant_category: str = Field(..., example="online", description="Category of merchant")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str

@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API (MLflow Version)",
        "links": {
            "documentation": "/docs",
            "health_check": "/health"
        }
    }

@app.get("/health")
def health_check():
    if model is None or encoder is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "healthy", "source": "MLflow Registry", "model": MODEL_NAME}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    
    try:
        merchant_encoded = encoder.transform([transaction.merchant_category])[0]
    except ValueError:
        merchant_encoded = 0
    
    features = [[
        transaction.amount,
        transaction.hour,
        transaction.day_of_week,
        merchant_encoded
    ]]
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Get model version
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    
    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4),
        model_version=latest.version
    )

if __name__ == "__main__":
    print("\nVisit http://127.0.0.1:8001/docs to test the API via Swagger UI!\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)
