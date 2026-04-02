"""
Tests for the FastAPI prediction service.

These tests ensure the API:
1. Returns correct responses for valid inputs
2. Rejects invalid inputs with proper error messages
3. Health check works properly

Why test the API in CI/CD?
- Catches breaking changes before deployment
- Ensures validation works correctly
- Verifies API contract (inputs/outputs)
- Documents expected behavior

Run with: pytest tests/test_api.py -v
Note: Requires the API to be running on localhost:8000
Start API with: uvicorn src.serve_validated:app --reload --host 0.0.0.0 --port 8000
"""
import pytest
import httpx

# Base URL for the API - change if running on different port/host
BASE_URL = "http://localhost:8000"


class TestPredictionEndpoint:
    """
    Tests for the /predict endpoint.
    
    These tests verify:
    - Valid inputs return correct predictions
    - Invalid inputs are rejected with proper error codes
    - Response format matches expected schema
    """
    
    def test_valid_prediction_returns_200(self):
        """
        Test that a valid transaction returns a successful response.
        
        This is the happy path - a normal transaction should:
        - Return HTTP 200
        - Include is_fraud (bool) in response
        - Include fraud_probability (float 0-1) in response
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1
    
    def test_high_risk_transaction(self):
        """
        Test that high-risk transactions are processed correctly.
        
        High-risk characteristics:
        - High amount ($500)
        - Late night (hour 3)
        - Online merchant (historically higher fraud rate)
        
        The model should still return a valid response.
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 500.0,
            "hour": 3,
            "day_of_week": 1,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        # High risk transactions should have higher fraud probability
        # (but we just verify it's in valid range, not exact value)
        assert data["fraud_probability"] >= 0.0
    
    def test_negative_amount_rejected(self):
        """
        Test that negative amounts are rejected with HTTP 400.
        
        Negative amounts are:
        - Physically impossible (can't have negative transactions)
        - A clear validation error
        - Should return clear error message
        
        The API should return 400 (not 200) for invalid input.
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": -100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
        assert "errors" in response.json()["detail"]
    
    def test_invalid_hour_rejected(self):
        """
        Test that invalid hours are rejected with HTTP 400.
        
        Hours must be 0-23. Values outside this range indicate:
        - Client-side validation failed
        - Data corruption in transit
        - Client is using wrong format
        
        Should return 400, not 200.
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 25,  # Invalid: hours must be 0-23
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_invalid_merchant_rejected(self):
        """
        Test that unknown merchant categories are rejected.
        
        Merchant categories must be one of the known values:
        grocery, restaurant, retail, online, travel
        
        Unknown categories would cause:
        - Encoder failure in model preprocessing
        - Potential model errors
        - Inconsistent behavior vs training
        
        Should return 400 with clear error message.
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "unknown_category"  # Invalid
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_missing_field_rejected(self):
        """
        Test that missing required fields are rejected.
        
        Required fields: amount, hour, day_of_week, merchant_category
        
        Missing fields should be caught by Pydantic validation
        and return HTTP 422 (Unprocessable Entity).
        
        This is FastAPI's built-in validation, not our custom validation.
        """
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14
            # Missing: day_of_week, merchant_category
        }, timeout=10)
        
        assert response.status_code == 422


class TestHealthEndpoint:
    """
    Tests for the /health endpoint.
    
    The health endpoint is used by:
    - Load balancers to check if service is alive
    - Kubernetes liveness/readiness probes
    - Monitoring systems
    
    It should be simple and fast, returning just service status.
    """
    
    def test_health_returns_200(self):
        """
        Test that health endpoint returns HTTP 200.
        
        Even if there are issues, the health endpoint should
        respond so load balancers know the service is running.
        """
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self):
        """
        Test that health endpoint returns healthy status.
        
        The response should indicate the service is operational.
        """
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        assert data["status"] == "healthy"