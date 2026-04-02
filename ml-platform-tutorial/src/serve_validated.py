"""
FastAPI service with integrated data validation for fraud detection.

This version adds a critical safety layer - ALL inputs are validated BEFORE
they reach the model. This prevents:
- Garbage predictions from invalid input
- Model errors from malformed data
- Silent failures that are hard to debug

Key improvements over naive version:
1. Input validation using data_validation.py
2. HTTP 400 response with clear error messages for invalid input
3. Validation flag in response to confirm validation passed
4. Better error messages that help clients fix their requests

This follows the "fail fast" principle - reject bad input early rather than
making bad predictions.
"""
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Import our validation function
# This is the same validation logic we tested in data_validation.py
from src.data_validation import validate_transaction

# Load the trained model and label encoder
# In production, you'd load from MLflow Model Registry (see serve_mlflow.py)
with open("models/model.pkl", "rb") as f:
    model, encoder = pickle.load(f)

# Create FastAPI application with descriptive metadata
app = FastAPI(
    title="Fraud Detection API (Validated)",
    description="""
    Fraud detection API with built-in input validation.
    
    SECURITY: All inputs are validated BEFORE prediction.
    Invalid inputs are rejected with HTTP 400 and detailed error messages.
    
    Validation rules:
    - amount: Must be positive and below $50,000
    - hour: Must be integer between 0-23
    - day_of_week: Must be integer between 0-6 (Monday to Sunday)
    - merchant_category: Must be one of: grocery, restaurant, retail, online, travel
    
    Why validate?
    - Prevents garbage predictions from bad input
    - Catches errors early with actionable messages
    - Makes debugging easier with explicit error reporting
    """,
    version="3.0.0"
)


class Transaction(BaseModel):
    """
    Input schema for transaction prediction.
    
    Uses Pydantic for automatic validation and documentation.
    FastAPI will automatically validate:
    - Field types (float, int, str)
    - Required fields (all fields have ...)
    - Default values if any
    
    The 'example' values appear in the OpenAPI/Swagger documentation.
    """
    amount: float = Field(
        ..., 
        description="Transaction amount in dollars (must be positive)",
        example=150.00
    )
    hour: int = Field(
        ..., 
        description="Hour of day (0-23)",
        example=14
    )
    day_of_week: int = Field(
        ..., 
        description="Day of week (0=Monday, 6=Sunday)",
        example=3
    )
    merchant_category: str = Field(
        ..., 
        description="Type of merchant (grocery, restaurant, retail, online, travel)",
        example="online"
    )


class PredictionResponse(BaseModel):
    """
    Response schema for fraud prediction.
    
    Includes:
    - is_fraud: Boolean prediction
    - fraud_probability: Probability score (0.0 to 1.0)
    - validation_passed: Confirms validation was performed (always True here)
    """
    is_fraud: bool
    fraud_probability: float
    validation_passed: bool = True


class ValidationErrorResponse(BaseModel):
    """
    Response schema for validation errors.
    
    Includes:
    - message: Summary of what went wrong
    - errors: List of specific validation errors
    - input: The original input (for debugging)
    """
    detail: dict


@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    responses={
        400: {"model": ValidationErrorResponse, "description": "Validation failed"}
    }
)
def predict(tx: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Pipeline:
    1. Pydantic validates basic types (if this fails, FastAPI returns 422)
    2. Our validate_transaction() checks business rules
    3. If invalid, return HTTP 400 with detailed errors
    4. If valid, make prediction and return result
    
    This "defense in depth" ensures only valid data reaches the model.
    
    Args:
        tx: Transaction data from request body
        
    Returns:
        PredictionResponse with fraud prediction
        
    Raises:
        HTTPException: 400 if validation fails
    """
    # Convert Pydantic model to dictionary for validation
    data = tx.dict()
    
    # Run our custom validation (business rules beyond basic types)
    validation = validate_transaction(data)
    
    # If validation failed, return HTTP 400 with clear error messages
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Validation failed - input rejected",
                "errors": validation["errors"],
                "hint": "Fix the errors above and retry",
                "input": data  # Include original input for debugging
            }
        )
    
    # Validation passed - proceed with prediction
    
    # Encode merchant category using the same encoder from training
    # This ensures consistency between training and serving
    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    
    # Prepare features in the same order as training
    # Feature order: amount, hour, day_of_week, merchant_encoded
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    # Get prediction and probability from model
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    # Return prediction result
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        validation_passed=True  # Confirms validation was applied
    )


@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Returns service status - useful for:
    - Load balancer health checks
    - Kubernetes liveness probes
    - Monitoring systems
    """
    return {
        "status": "healthy", 
        "validation": "enabled",
        "model": "loaded"
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API (Validated)",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "validation": "enabled"
    }


# Run with: uvicorn src.serve_validated:app --reload --host 0.0.0.0 --port 8000