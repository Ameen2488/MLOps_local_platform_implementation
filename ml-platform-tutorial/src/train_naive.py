# src/train_naive.py
"""
Train a fraud detection model - NAIVE VERSION.

This script demonstrates the "quick and dirty" approach to ML:
- No experiment tracking (results aren't logged to a central server)
- No model versioning (overwrites model.pkl every time)
- No data versioning (uses whatever is in data/train.csv)

The goal of this script is to establish a baseline model using a 
Random Forest Classifier and save it for inference.
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def main():
    # --- 1. Load Data ---
    # We load the synthetic data generated in the previous step.
    print("Loading data...")
    try:
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Error: data/train.csv or data/test.csv not found. Please run src/generate_data.py first.")
        return

    print(f"Dataset Loaded:")
    print(f"  - Training samples: {len(train_df):,}")
    print(f"  - Test samples    : {len(test_df):,}")
    print(f"  - Fraud ratio (%) : {train_df['is_fraud'].mean():.2%}")

    # --- 2. Feature Engineering (Naive) ---
    # Our model needs numeric inputs. We use LabelEncoder to convert the 
    # 'merchant_category' string into numbers (0, 1, 2, etc.).
    # IMPORTANT: We must save this encoder to use the exact same mapping 
    # during real-time inference (the 'serve' step).
    print("\nEncoding categorical features...")
    encoder = LabelEncoder()
    train_df["merchant_encoded"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_encoded"] = encoder.transform(test_df["merchant_category"])
    
    # Showcase the mapping for transparency
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"Merchant category mapping: {mapping}")

    # --- 3. Prepare Feature Matrices ---
    # Select our features (X) and target (y).
    # 'day_of_week' and 'hour' are cyclic, but here we treat them as simple integers.
    feature_cols = ["amount", "hour", "day_of_week", "merchant_encoded"]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]

    # --- 4. Model Training ---
    # We use Random Forest because it handles non-linear relationships well
    # and provides feature importance out-of-the-box.
    print("\nTraining Random Forest model (baseline)...")
    model = RandomForestClassifier(
        n_estimators=200,  # 100 trees in the forest
        max_depth=20,      # Constrain depth to prevent over-fitting
        random_state=42,   # Fixed seed for same results every time
        n_jobs=-1          # Use all available CPU cores for speed
    )
    model.fit(X_train, y_train)
    print("Training complete!")

    # --- 5. Model Evaluation ---
    # Since our data is imbalanced (only 2% fraud), Accuracy is a poor metric.
    # We focus on Precision (of flagged fraud, how many were real?) and 
    # Recall (of all real fraud, how many did we catch?).
    print("\n" + "="*50)
    print("MODEL PERFORMANCE REPORT")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f} (Can be misleading!)")
    print(f"Precision: {precision_score(y_test, y_pred):.4f} (Minimizing false alarms)")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f} (Catching actual fraud)")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f} (Balanced harmonic mean)")

    # Confusion Matrix gives a raw count of correct vs wrong predictions
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    # cm[row][col] -> [Actual][Predicted]
    print(f"  True Negatives : {cm[0][0]:,} (Legit transactions identified correctly)")
    print(f"  False Positives: {cm[0][1]:,} (Legit transactions wrongly flagged as fraud)")
    print(f"  False Negatives: {cm[1][0]:,} (Fraudulent transactions MISSED - High Risk!)")
    print(f"  True Positives : {cm[1][1]:,} (Fraudulent transactions caught effectively)")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    # --- 6. Feature Importance ---
    # Check which features influenced the model the most.
    print("\nFeature Impact (Importance):")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for name, score in importances:
        print(f"  - {name}: {score:.4f}")

    # --- 7. Save Model Artifacts ---
    # We save both the model AND the encoder into a single pickle file.
    # This prevents 'Training-Serving Skew' where the API might interpret 
    # category '0' differently than the training script did.
    print("\nSaving artifacts to models/model.pkl...")
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, encoder), f)

    print("\nDone! Baseline model is ready for the 'serve' step.")
    print("Note: In a mature MLOps pipeline, we would log these metrics to MLflow.")

if __name__ == "__main__":
    main()
