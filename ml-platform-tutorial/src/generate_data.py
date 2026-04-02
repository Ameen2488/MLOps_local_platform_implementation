"""
Generate synthetic fraud detection dataset.

This script is the first step in building our local MLOps platform.
It simulates transaction data with specific fraudulent patterns:
1. High transaction amounts: Fraudulent transactions are typically much larger on average
   than legitimate ones (simulated with a higher log-normal mean).
2. Late-night timing: Fraud often occurs during hours with lower scrutiny (0-5, 23).
3. Online/Travel Merchant Categories: Certain categories are more prone to fraud.

The purpose of this script is to provide a consistent, reproducible dataset
without requiring external data downloads.
"""

import pandas as pd
import numpy as np

def generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=42):
    """
    Generate synthetic fraud detection dataset with realistic-looking patterns.
    
    Args:
        n_samples (int): Total number of transactions (rows) to generate.
        fraud_ratio (float): Targeted proportion of fraudulent transactions (e.g., 0.02 for 2%).
        seed (int): Seed for reproducibility of random data.
        
    Returns:
        pd.DataFrame: A shuffled DataFrame containing transaction features and labels.
        
    Features generated:
    - amount (float): Transaction value in dollars.
    - hour (int): Hour of day (0-23).
    - day_of_week (int): Day of the week (0=Mon, 6=Sun).
    - merchant_category (string): 'grocery', 'restaurant', 'retail', 'online', or 'travel'.
    - is_fraud (binary): 1 for fraud, 0 for legitimate.
    """
    
    # Set seed for reproducible results across different runs
    np.random.seed(seed)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Generate Legitimate Transactions ---
    # Patterns: 
    # - Small to moderate amounts (log-normal distribution)
    # - Spread out normally across all hours
    # - Mostly grocery/retail shopping
    legit = pd.DataFrame({
        "amount": np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit),  # Average ~$33
        "hour": np.random.randint(0, 24, size=n_legit),
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_legit,
            p=[0.30, 0.25, 0.25, 0.15, 0.05]  # Frequent everyday categories
        ),
        "is_fraud": 0
    })

    # --- Generate Fraudulent Transactions ---
    # Patterns:
    # - Much higher amounts than normal (fraudsters wanting large payouts)
    # - Higher likelihood of late-night hours (e.g., 1 AM to 4 AM)
    # - Heavily skewed toward Online and Travel categories
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),  # Average ~$245
        "hour": np.random.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud),   # Late-night hours
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_fraud,
            p=[0.05, 0.05, 0.10, 0.60, 0.20]  # Skewed toward high-risk categories
        ),
        "is_fraud": 1
    })

    # Concatenate the two groups and shuffle them thoroughly
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # --- Execute Script ---
    print("Generating synthetic fraud detection dataset...")
    
    # Generate 10,000 samples for our initial training
    df = generate_transactions(n_samples=10000, fraud_ratio=0.02)

    # --- Split Into Train and Test sets ---
    # We use a 80/20 train-test split for model evaluation
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # --- Save To CSV Files ---
    # These files will be used by our training script
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # --- Summary Metrics ---
    # Let's print some stats to verify our simulated patterns
    print(f"\nDataset generated successfully!")
    print(f"Training set row count: {len(train_df):,} transactions")
    print(f"Test set row count    : {len(test_df):,} transactions")
    print(f"Overall fraud ratio   : {df['is_fraud'].mean():.2%}")
    
    # Verify the simulated "amount" gap
    print(f"\nAverage Transaction Amount:")
    print(f"  - Legitimate: ${df[df['is_fraud']==0]['amount'].mean():.2f}")
    print(f"  - Fraudulent: ${df[df['is_fraud']==1]['amount'].mean():.2f}")

    # Verify the "merchant_category" distribution for fraud
    print(f"\nMerchant category distribution for Fraudulent transactions:")
    print(df[df['is_fraud']==1]['merchant_category'].value_counts(normalize=True))
