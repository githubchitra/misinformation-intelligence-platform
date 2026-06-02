# baseline.py
"""
Baseline Model Script
Trains a TF-IDF + Logistic Regression model on the processed dataset as a baseline,
evaluates it, and saves the baseline model.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def train_baseline():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Processed datasets not found! Please run 'python preprocess.py' first.")
        return

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Fill any empty cells
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")

    X_train, y_train = train_df["cleaned_text"], train_df["binary_label"]
    X_test, y_test = test_df["cleaned_text"], test_df["binary_label"]

    print("Fitting TF-IDF + Logistic Regression baseline model...")
    # Using TF-IDF with unigrams/bigrams and sublinear TF scaling
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Save baseline model
    os.makedirs("models", exist_ok=True)
    baseline_path = "models/baseline_model.joblib"
    joblib.dump(pipeline, baseline_path)
    print(f"Saved baseline model to {baseline_path}")

    # Predict and evaluate
    print("\n--- Baseline Model Evaluation ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
    print("\nClassification Report:")
    print(report)
    
    # Save metrics to file
    with open("models/baseline_metrics.txt", "w") as f:
        f.write("TF-IDF + Logistic Regression Baseline Metrics\n")
        f.write("==============================================\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}\n")
    print("Saved baseline metrics to models/baseline_metrics.txt")

if __name__ == "__main__":
    train_baseline()
