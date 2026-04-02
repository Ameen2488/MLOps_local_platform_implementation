"""
Prepare feature data for Feast.

This script orchestrates the entire feature preparation pipeline:
1. Load raw transaction data from CSV
2. Compute aggregated merchant-level features (the feature engineering step)
3. Save features in Parquet format (Feast's preferred offline storage format)
4. Register feature definitions with Feast (apply)
5. Materialize features to the online store for low-latency serving

Why this matters: Running this script ensures your feature store is synchronized
with your training data. It should be run whenever:
- You have new training data
- You want to refresh features in the online store
- Before deploying to production

The key insight: All feature computation happens HERE in one place.
Training and serving code simply retrieve pre-computed features from Feast,
ensuring consistency between offline (training) and online (serving) environments.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import os


def compute_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated features grouped by merchant category.
    
    THIS FUNCTION IS THE SINGLE SOURCE OF TRUTH FOR FEATURE COMPUTATION.
    
    This computes historical statistics that the model uses as features:
    - avg_amount: How much do customers typically spend at this merchant type?
    - transaction_count: How popular is this merchant category?
    - fraud_rate: What's the historical fraud rate? (KEY for fraud detection!)
    
    By computing these once and storing in Feast, we avoid:
    - Recalculating during inference (slow)
    - Different logic in training vs serving (causes skew)
    - Storing raw data and computing on-the-fly (error-prone)
    
    The fraud_rate is especially important - it tells the model that
    'online' merchants have higher fraud historically, so a high-value
    online transaction is more likely to be fraudulent.
    
    Args:
        df: Transaction DataFrame with columns: amount, merchant_category, is_fraud
        
    Returns:
        DataFrame with computed features per merchant category, with columns:
        - merchant_category: The entity key for lookup
        - avg_amount: Float32 average transaction amount
        - transaction_count: Int64 total transactions
        - fraud_rate: Float32 historical fraud proportion (0.0 to 1.0)
        - event_timestamp: datetime for Feast to track freshness
    """
    print("Computing merchant-level features...")
    
    # Group by merchant category and compute aggregations
    # - 'mean' on 'amount' gives average transaction amount
    # - 'count' on 'amount' gives total transaction count  
    # - 'mean' on 'is_fraud' gives fraud rate (proportion of fraud)
    stats = df.groupby('merchant_category').agg({
        'amount': ['mean', 'count'],
        'is_fraud': 'mean'
    }).reset_index()
    
    # Flatten column names from multi-index (amount_mean -> avg_amount)
    stats.columns = ['merchant_category', 'avg_amount', 'transaction_count', 'fraud_rate']
    
    # Add timestamp for Feast to track when these features were computed
    # This helps Feast understand data freshness and apply TTL correctly
    stats['event_timestamp'] = datetime.now()
    
    # Cast to specific dtypes for Feast schema compliance
    # Feast requires exact dtype matching with the FeatureView schema
    stats['avg_amount'] = stats['avg_amount'].astype('float32')
    stats['transaction_count'] = stats['transaction_count'].astype('int64')
    stats['fraud_rate'] = stats['fraud_rate'].astype('float32')
    
    return stats


def main():
    """
    Main orchestration function for the Feast feature preparation pipeline.
    
    Steps:
    1. Load raw training data from CSV
    2. Compute aggregated merchant features
    3. Save to Parquet (offline store format)
    4. Apply Feast feature definitions (register in registry)
    5. Materialize to online store (for low-latency serving)
    
    After this completes, features are available for:
    - Training: get_historical_features() - batch retrieval
    - Serving: get_online_features() - real-time retrieval
    """
    print("="*60)
    print("FEAST FEATURE PREPARATION")
    print("="*60)
    
    # Step 1: Load raw transaction data
    # This is the same data used for model training
    print("\n1. Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"   Loaded {len(train_df):,} transactions")
    
    # Step 2: Compute feature aggregations
    # This is the feature engineering step - turning raw transactions into features
    print("\n2. Computing merchant features...")
    merchant_features = compute_merchant_features(train_df)
    
    # Display computed features for verification
    print("\n   Computed features:")
    print(merchant_features.to_string(index=False))
    
    # Step 3: Save to Parquet format
    # Parquet is efficient for analytics and is Feast's preferred format
    # for offline batch data
    print("\n3. Saving features to Parquet...")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/merchant_features.parquet'
    merchant_features.to_parquet(output_path, index=False)
    print(f"   Saved to {output_path}")
    
    # Step 4: Register feature definitions with Feast
    # 'feast apply' reads features.py and creates/updates the registry
    # The registry is the central catalog of all features
    print("\n4. Applying Feast feature definitions...")
    try:
        result = subprocess.run(
            ['feast', 'apply'],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Feature definitions applied successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error applying Feast: {e.stderr}")
        raise
    
    # Step 5: Materialize features to online store
    # This loads features from the offline store (Parquet) into the online store
    # (SQLite in our case) for low-latency retrieval at inference time
    print("\n5. Materializing features to online store...")
    try:
        result = subprocess.run(
            ['feast', 'materialize-incremental', datetime.now().isoformat()],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Features materialized successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error materializing: {e.stderr}")
        raise
    
    print("\n" + "="*60)
    print("FEAST FEATURE PREPARATION COMPLETE!")
    print("="*60)
    print("\nYou can now:")
    print("  - Retrieve features for training: get_training_features()")
    print("  - Retrieve features for serving: get_online_features()")
    print("  - View feature stats: feast feature-views list")


if __name__ == "__main__":
    main()