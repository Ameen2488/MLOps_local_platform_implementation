# src/serve_naive.py
"""
Serve fraud detection model as a REST API - NAIVE VERSION.

This script uses FastAPI to create a web server that:
1. Loads the trained model (and encoder) at startup.
2. Accepts transaction data via a POST request to '/predict'.
3. Returns the fraud prediction and probability.

This is a "naive" version because:
- It lacks robust input validation (beyond basic types).
- It doesn't log requests/predictions to a database.
- It doesn't have monitoring for model drift or performance.
- It loads the model from a local file instead of a model registry.
"""

import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# --- 1. Load Model Artifacts ---
# We load the model and encoder once when the script starts.
# This is much more efficient than loading them for every request.
print("Starting API Server...")
print("Loading model and encoder from models/model.pkl...")

try:
    with open("models/model.pkl", "rb") as f:
        # The training script saved both as a tuple
        model, encoder = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: models/model.pkl not found. Please run src/train_naive.py first.")
    model, encoder = None, None

# --- 2. Define API Application ---
app = FastAPI(
    title="Fraud Detection API (Naive)",
    description="A simple API to predict credit card fraud using a Random Forest model.",
    version="1.0.0"
)

# --- 3. Define Data Schemas ---
# Pydantic models define the structure of our JSON requests and responses.
# This gives us automatic validation and generates interactive Swagger docs.

class TransactionRequest(BaseModel):
    """Schema for incoming transaction data."""
    amount: float = Field(..., example=120.50, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, example=14, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="Day (0=Mon, 6=Sun)")
    merchant_category: str = Field(..., example="online", description="Category of merchant")

class PredictionResponse(BaseModel):
    """Schema for the API's fraud prediction response."""
    is_fraud: bool = Field(..., description="True if transaction is predicted as fraud")
    fraud_probability: float = Field(..., description="Probability score between 0.0 and 1.0")
    status: str = Field("success", description="Indicates if the prediction was processed")

# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    """Welcome endpoint with basic API info."""
    return {
        "message": "Welcome to the Fraud Detection API (Naive Version)",
        "links": {
            "documentation": "/docs",
            "health_check": "/health"
        }
    }

@app.get("/health")
def health_check():
    """
    Standard health check endpoint.
    Used by monitoring tools to verify the service is alive and the model is loaded.
    """
    if model is None or encoder is None:
        return {"status": "unhealthy", "error": "Model artifacts not loaded"}
    return {"status": "healthy", "model": "RandomForest", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud for a single transaction.
    
    Steps:
    1. Parse the JSON request into a Pydantic object.
    2. Encode the categorical 'merchant_category' using the saved LabelEncoder.
    3. Construct a feature vector for the model.
    4. Run inference and return results.
    """
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    # A. Feature Preparation
    # We must replicate the exact feature engineering steps from training.
    try:
        # LabelEncoder.transform expects a list/array
        merchant_encoded = encoder.transform([transaction.merchant_category])[0]
    except ValueError:
        # If we see a category we didn't train on, we default to a safe value (or could error)
        # In a real app, 'Unknown' categories should be handled more formally.
        print(f"Warning: Unknown merchant category '{transaction.merchant_category}'. Defaulting to 'retail' encoding.")
        # Defaulting to index 0 (usually 'grocery' or similar) as a fallback
        merchant_encoded = 0

    # B. Inference
    # Model expects a 2D array: [[feature1, feature2, ...]]
    features = [[
        transaction.amount,
        transaction.hour,
        transaction.day_of_week,
        merchant_encoded
    ]]
    
    # Get binary prediction (0 or 1)
    prediction = model.predict(features)[0]
    
    # Get probability mapping for class 1 (fraud)
    # predict_proba returns [[prob_legit, prob_fraud]]
    probability = model.predict_proba(features)[0][1]

    # C. Return Response
    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4)
    )

if __name__ == "__main__":
    # Standard entry point to run the server locally.
    # Note: Use 'uvicorn src.serve_naive:app --reload' for development auto-reloads.
    print("\nVisit http://127.0.0.1:8000/docs to test the API via Swagger UI!\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
