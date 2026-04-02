"""
Data validation for fraud detection API.

This module provides data validation functions that act as a safety net for the ML pipeline:
- validate_transaction(): Validates a single transaction before prediction
- validate_batch(): Validates batch data using Great Expectations

WHY THIS MATTERS: 
In production, invalid data can cause:
- Garbage predictions that look valid but are wrong
- Model errors or crashes
- Silent failures that are hard to debug

By validating BEFORE prediction, we:
- Catch bad data early with clear error messages
- Prevent invalid predictions
- Make debugging easier with explicit error reporting

This is a critical part of the "defense in depth" strategy for ML systems.
"""
import pandas as pd
from typing import Dict, List, Any, Optional

# Valid merchant categories - anything else is invalid
VALID_CATEGORIES = ["grocery", "restaurant", "retail", "online", "travel"]

# Business rules for validation
MAX_TRANSACTION_AMOUNT = 50000  # Maximum allowed transaction amount in dollars


def validate_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single transaction before making a fraud prediction.
    
    This function acts as the first line of defense - it checks ALL business rules
    and data quality requirements BEFORE the transaction reaches the model.
    
    Each field is checked for:
    1. Presence (required fields must exist)
    2. Type (correct data type)
    3. Range/Value (within acceptable bounds)
    4. Format (for categorical fields, must be valid category)
    
    Why validate at API time?
    - User errors happen (typos, wrong values)
    - Upstream systems can send bad data
    - Catching errors early prevents downstream issues
    
    Args:
        data: Dictionary containing transaction fields with keys:
            - amount (float): Transaction amount in dollars
            - hour (int): Hour of day (0-23)
            - day_of_week (int): Day of week (0=Monday to 6=Sunday)
            - merchant_category (str): Type of merchant
        
    Returns:
        Dictionary with validation results:
            - valid (bool): True if all checks pass
            - errors (list): List of error messages if validation fails
            
    Example:
        >>> validate_transaction({"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"})
        {"valid": True, "errors": []}
        
        >>> validate_transaction({"amount": -100, "hour": 25, ...})
        {"valid": False, "errors": ["amount must be positive", "hour must be between 0 and 23"]}
    """
    errors = []
    
    # Validate amount - must be positive and within reasonable bounds
    # Why? Negative amounts are physically impossible, and extremely high
    # amounts likely indicate errors or fraud attempts
    amount = data.get("amount")
    if amount is None:
        errors.append("amount is required")
    elif not isinstance(amount, (int, float)):
        errors.append(f"amount must be a number (got {type(amount).__name__})")
    elif amount <= 0:
        errors.append("amount must be positive")
    elif amount > MAX_TRANSACTION_AMOUNT:
        errors.append(f"amount exceeds maximum allowed value of $50,000 (got ${amount:,.2f})")
    
    # Validate hour - must be valid hour of day
    # Why? Hours outside 0-23 are invalid and would cause model issues
    hour = data.get("hour")
    if hour is None:
        errors.append("hour is required")
    elif not isinstance(hour, int):
        errors.append(f"hour must be an integer (got {type(hour).__name__})")
    elif not (0 <= hour <= 23):
        errors.append(f"hour must be between 0 and 23 (got {hour})")
    
    # Validate day_of_week - must be valid day of week
    # Why? Days outside 0-6 are invalid
    day = data.get("day_of_week")
    if day is None:
        errors.append("day_of_week is required")
    elif not isinstance(day, int):
        errors.append(f"day_of_week must be an integer (got {type(day).__name__})")
    elif not (0 <= day <= 6):
        errors.append(f"day_of_week must be between 0 (Monday) and 6 (Sunday) (got {day})")
    
    # Validate merchant_category - must be a known category
    # Why? Unknown categories would cause issues during feature encoding,
    # and we need to ensure consistency with training data
    category = data.get("merchant_category")
    if category is None:
        errors.append("merchant_category is required")
    elif not isinstance(category, str):
        errors.append(f"merchant_category must be a string (got {type(category).__name__})")
    elif category not in VALID_CATEGORIES:
        errors.append(
            f"merchant_category must be one of {VALID_CATEGORIES} (got '{category}')"
        )
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_batch(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a batch of transactions using Great Expectations.
    
    This is more sophisticated than single-transaction validation and is used for:
    - Validating training data before model training
    - Validating batch prediction requests
    - Data quality monitoring
    
    Great Expectations provides:
    - Declarative data contracts (define expectations, not validation code)
    - Detailed validation reports
    - Data profiling and documentation
    
    Why use Great Expectations?
    - More expressive than manual validation
    - Self-documenting expectations
    - Great for data pipeline quality gates
    
    Args:
        df: DataFrame with transaction data containing columns:
            - amount: Transaction amount
            - hour: Hour of day
            - day_of_week: Day of week
            - merchant_category: Merchant category
        
    Returns:
        Dictionary with comprehensive validation results:
            - success (bool): True if all expectations pass
            - passed (int): Number of passed checks
            - total (int): Total number of checks
            - pass_rate (float): Percentage of passed checks
            - details (dict): Detailed results for each check
    """
    import great_expectations as gx
    
    # Convert pandas DataFrame to Great Expectations dataset
    ge_df = gx.from_pandas(df)
    
    results = []
    
    # Expect amount to be in valid range (0.01 to 50000)
    # mostly=0.99 allows 1% of values to fail without failing the check
    r = ge_df.expect_column_values_to_be_between(
        'amount', min_value=0.01, max_value=50000, mostly=0.99
    )
    results.append(('amount_range', r.success, r.result))
    
    # Expect hour to be in valid range (0-23)
    r = ge_df.expect_column_values_to_be_between(
        'hour', min_value=0, max_value=23
    )
    results.append(('hour_range', r.success, r.result))
    
    # Expect day_of_week to be in valid range (0-6)
    r = ge_df.expect_column_values_to_be_between(
        'day_of_week', min_value=0, max_value=6
    )
    results.append(('day_range', r.success, r.result))
    
    # Expect merchant_category to be in valid set
    r = ge_df.expect_column_values_to_be_in_set(
        'merchant_category', VALID_CATEGORIES
    )
    results.append(('category_valid', r.success, r.result))
    
    # Check that no required columns have null values
    for col in ['amount', 'hour', 'day_of_week', 'merchant_category']:
        r = ge_df.expect_column_values_to_not_be_null(col)
        results.append((f'{col}_not_null', r.success, r.result))
    
    # Calculate summary statistics
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    return {
        'success': passed == total,
        'passed': passed,
        'total': total,
        'pass_rate': passed / total,
        'details': {name: {'passed': success, 'result': result} 
                   for name, success, result in results}
    }


# Test/demonstration code - runs when script is executed directly
if __name__ == "__main__":
    print("="*60)
    print("TESTING DATA VALIDATION")
    print("="*60)
    
    # Test 1: Single transaction validation
    print("\n1. Single Transaction Validation")
    print("-"*40)
    
    test_cases = [
        {
            "name": "Valid transaction",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Negative amount",
            "data": {"amount": -100.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Invalid hour",
            "data": {"amount": 50.0, "hour": 25, "day_of_week": 3, "merchant_category": "grocery"}
        },
        {
            "name": "Unknown merchant",
            "data": {"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "unknown"}
        },
        {
            "name": "Everything wrong",
            "data": {"amount": -999, "hour": 99, "day_of_week": 15, "merchant_category": "fake"}
        },
    ]
    
    for tc in test_cases:
        result = validate_transaction(tc["data"])
        status = "PASS" if result["valid"] else "FAIL"
        print(f"\n{tc['name']}: {status}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"  - {error}")
    
    # Test 2: Batch validation with Great Expectations
    print("\n\n2. Batch Validation with Great Expectations")
    print("-"*40)
    
    train_df = pd.read_csv('data/train.csv')
    results = validate_batch(train_df)
    
    print(f"\nTraining data validation: {results['passed']}/{results['total']} checks passed")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    
    if not results['success']:
        print("\nFailed checks:")
        for name, detail in results['details'].items():
            if not detail['passed']:
                print(f"  - {name}")