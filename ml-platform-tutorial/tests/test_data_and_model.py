"""
Tests for data quality and model performance.

These tests run in CI/CD to ensure:
1. Data meets quality requirements (no bad data in, no garbage out)
2. Model meets performance thresholds (detects fraud effectively)
3. No regressions are introduced (new changes don't break existing functionality)

Why test in CI/CD?
- Catches problems before they reach production
- Ensures consistency across environments
- Provides confidence for deployments
- Creates documentation of expected behavior

Run with: pytest tests/test_data_and_model.py -v
"""
import pandas as pd
import pickle
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TestDataQuality:
    """
    Tests to ensure data quality meets requirements.
    
    These tests verify that:
    - Data files exist and are readable
    - Required columns are present
    - Values are within valid ranges
    - No data corruption or missing values
    
    Running these tests in CI/CD prevents bad data from causing
    model training issues or production problems.
    """
    
    @pytest.fixture
    def train_data(self):
        """Load training data for tests."""
        return pd.read_csv("data/train.csv")
    
    @pytest.fixture
    def test_data(self):
        """Load test data for tests."""
        return pd.read_csv("data/test.csv")
    
    def test_train_data_has_expected_columns(self, train_data):
        """
        Verify training data has all required columns.
        
        Why: Missing columns would cause training failures or
        use wrong features, leading to poor model performance.
        """
        required_columns = {"amount", "hour", "day_of_week", "merchant_category", "is_fraud"}
        actual_columns = set(train_data.columns)
        missing = required_columns - actual_columns
        assert not missing, f"Missing columns: {missing}"
    
    def test_train_data_not_empty(self, train_data):
        """
        Verify training data has sufficient samples.
        
        Why: Empty or too-small datasets lead to:
        - Training failures
        - Poor model generalization
        - Unreliable evaluation
        """
        assert len(train_data) > 0, "Training data is empty"
        assert len(train_data) >= 1000, f"Training data too small: {len(train_data)} rows"
    
    def test_no_negative_amounts(self, train_data):
        """
        Verify transaction amounts are non-negative.
        
        Why: Negative amounts are physically impossible and indicate
        data corruption that would cause model to learn wrong patterns.
        """
        negative_count = (train_data["amount"] < 0).sum()
        assert negative_count == 0, f"Found {negative_count} negative amounts"
    
    def test_amounts_reasonable(self, train_data):
        """
        Verify transaction amounts are within reasonable bounds.
        
        Why: Extremely high amounts (>$100k) are unusual and might
        indicate data issues or require special handling.
        """
        max_amount = train_data["amount"].max()
        assert max_amount <= 100000, f"Max amount {max_amount} exceeds reasonable limit"
    
    def test_hours_valid(self, train_data):
        """
        Verify hour values are valid (0-23).
        
        Why: Invalid hours would cause encoding issues and
        potentially corrupt model predictions.
        """
        invalid = train_data[(train_data["hour"] < 0) | (train_data["hour"] > 23)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid hours"
    
    def test_days_valid(self, train_data):
        """
        Verify day_of_week values are valid (0-6).
        
        Why: Days outside 0-6 indicate data corruption that
        could affect model training and inference.
        """
        invalid = train_data[(train_data["day_of_week"] < 0) | (train_data["day_of_week"] > 6)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid days"
    
    def test_merchant_categories_valid(self, train_data):
        """
        Verify merchant categories match expected values.
        
        Why: Unknown categories would fail during encoding in
        production, causing prediction failures.
        """
        valid_categories = {"grocery", "restaurant", "retail", "online", "travel"}
        actual_categories = set(train_data["merchant_category"].unique())
        invalid = actual_categories - valid_categories
        assert not invalid, f"Invalid merchant categories: {invalid}"
    
    def test_fraud_ratio_reasonable(self, train_data):
        """
        Verify fraud ratio is within realistic bounds.
        
        Why: Extremely high or low fraud ratios indicate data
        issues and would affect model training strategy.
        
        Real fraud rates typically range from 0.1% to 50%.
        """
        fraud_ratio = train_data["is_fraud"].mean()
        assert 0.001 <= fraud_ratio <= 0.5, f"Fraud ratio {fraud_ratio:.2%} is unrealistic"
    
    def test_no_nulls_in_critical_columns(self, train_data):
        """
        Verify no null values in critical columns.
        
        Why: Null values cause training failures and require
        imputation, which might introduce bias.
        """
        critical = ["amount", "hour", "day_of_week", "merchant_category", "is_fraud"]
        for col in critical:
            null_count = train_data[col].isnull().sum()
            assert null_count == 0, f"Column {col} has {null_count} null values"


class TestModelPerformance:
    """
    Tests to ensure model meets performance requirements.
    
    These tests verify that:
    - Model loads successfully
    - Model can make predictions
    - Performance metrics meet minimum thresholds
    
    Running these in CI/CD catches:
    - Corrupted model files
    - Performance regressions from retraining
    - Breaking changes from code updates
    """
    
    @pytest.fixture
    def model_and_encoder(self):
        """Load trained model and label encoder."""
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_data(self):
        """Load test data."""
        return pd.read_csv("data/test.csv")
    
    def test_model_loads_successfully(self, model_and_encoder):
        """
        Verify model and encoder load without errors.
        
        Why: Corrupted pickle files would cause production failures.
        This is a basic sanity check before more complex tests.
        """
        model, encoder = model_and_encoder
        assert model is not None, "Model is None"
        assert encoder is not None, "Encoder is None"
    
    def test_model_can_predict(self, model_and_encoder, test_data):
        """
        Verify model can make predictions on test data.
        
        Why: Models that fail silently or throw errors would
        cause production failures and missing predictions.
        """
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        predictions = model.predict(X)
        assert len(predictions) == len(X), "Prediction count mismatch"
    
    def test_accuracy_threshold(self, model_and_encoder, test_data):
        """
        Verify model accuracy meets minimum threshold (90%).
        
        Why: Low accuracy indicates model isn't learning properly.
        This catches major regressions that would affect many predictions.
        
        Note: For imbalanced fraud data, accuracy can be misleading,
        so we also test F1-score which handles imbalance better.
        """
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        accuracy = model.score(X, y)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below 90% threshold"
    
    def test_f1_threshold(self, model_and_encoder, test_data):
        """
        Verify model F1-score meets minimum threshold (0.3).
        
        Why: F1-score is more appropriate for imbalanced data
        (like fraud detection with ~2% positive class). It balances
        precision and recall, which is critical for fraud detection.
        
        - Low F1 means model isn't catching enough fraud OR
          is too many false positives
        """
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred)
        assert f1 >= 0.3, f"F1-score {f1:.2f} below 0.3 threshold"
    
    def test_precision_not_zero(self, model_and_encoder, test_data):
        """
        Verify model has non-zero precision.
        
        Why: Zero precision means the model predicts everyone as fraud,
        which would cause massive false positives in production.
        At least some predictions should be correct.
        """
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        assert precision > 0, "Model has zero precision (predicts no fraud)"
    
    def test_recall_not_zero(self, model_and_encoder, test_data):
        """
        Verify model has non-zero recall.
        
        Why: Zero recall means the model never predicts fraud,
        defeating the entire purpose of the fraud detection system.
        At least some actual fraud should be detected.
        """
        model, encoder = model_and_encoder
        test_data["merchant_encoded"] = encoder.transform(test_data["merchant_category"])
        X = test_data[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data["is_fraud"]
        y_pred = model.predict(X)
        recall = recall_score(y, y_pred, zero_division=0)
        assert recall > 0, "Model has zero recall (misses all fraud)"