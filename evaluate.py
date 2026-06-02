# evaluate.py
"""
Evaluation and Comparison Script
Loads the fine-tuned DistilBERT model and the baseline model, evaluates both on the
test dataset, prints classification reports, and outputs a comparison summary.
"""

import os
import argparse
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def evaluate_baseline(test_df):
    baseline_path = "models/baseline_model.joblib"
    if not os.path.exists(baseline_path):
        print("Baseline model not found. Skipping baseline evaluation...")
        return None
    
    print("Evaluating TF-IDF + Logistic Regression Baseline...")
    pipeline = joblib.load(baseline_path)
    X_test = test_df["cleaned_text"].fillna("")
    y_test = test_df["binary_label"]
    
    y_pred = pipeline.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
    
    return {
        "predictions": y_pred,
        "confusion_matrix": cm,
        "report": report
    }

def evaluate_distilbert(test_df):
    model_path = "models/distilbert_fake_news"
    if not os.path.exists(model_path):
        print("DistilBERT model not found. Skipping DistilBERT evaluation...")
        return None
    
    print("Evaluating DistilBERT Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    texts = test_df["cleaned_text"].fillna("").tolist()
    y_test = test_df["binary_label"].tolist()
    
    preds = []
    # Process in batches for safety
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
            
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Fake", "Real"], output_dict=True)
    
    return {
        "predictions": preds,
        "confusion_matrix": cm,
        "report": report
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="data/processed/test.csv", help="Path to processed test csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.test_data):
        print(f"Test data file not found at {args.test_data}. Please run preprocess.py first.")
        return
        
    test_df = pd.read_csv(args.test_data)
    y_test = test_df["binary_label"].values
    
    baseline_res = evaluate_baseline(test_df)
    distilbert_res = evaluate_distilbert(test_df)
    
    print("\n=======================================================")
    print("                 MODEL COMPARISON REPORT               ")
    print("=======================================================")
    
    comparison_data = []
    
    if baseline_res:
        b_rep = baseline_res["report"]
        comparison_data.append({
            "Model": "Baseline (TF-IDF + LR)",
            "Accuracy": b_rep["accuracy"],
            "Precision (Fake)": b_rep["Fake"]["precision"],
            "Recall (Fake)": b_rep["Fake"]["recall"],
            "F1-Score (Fake)": b_rep["Fake"]["f1-score"],
            "Precision (Real)": b_rep["Real"]["precision"],
            "Recall (Real)": b_rep["Real"]["recall"],
            "F1-Score (Real)": b_rep["Real"]["f1-score"]
        })
        
        print("\n--- Baseline Model Confusion Matrix ---")
        print(baseline_res["confusion_matrix"])
        print("\n--- Baseline Model Classification Report ---")
        print(classification_report(y_test, baseline_res["predictions"], target_names=["Fake", "Real"]))
        
    if distilbert_res:
        d_rep = distilbert_res["report"]
        comparison_data.append({
            "Model": "DistilBERT (Fine-Tuned)",
            "Accuracy": d_rep["accuracy"],
            "Precision (Fake)": d_rep["Fake"]["precision"],
            "Recall (Fake)": d_rep["Fake"]["recall"],
            "F1-Score (Fake)": d_rep["Fake"]["f1-score"],
            "Precision (Real)": d_rep["Real"]["precision"],
            "Recall (Real)": d_rep["Real"]["recall"],
            "F1-Score (Real)": d_rep["Real"]["f1-score"]
        })
        
        print("\n--- DistilBERT Model Confusion Matrix ---")
        print(distilbert_res["confusion_matrix"])
        print("\n--- DistilBERT Model Classification Report ---")
        print(classification_report(y_test, distilbert_res["predictions"], target_names=["Fake", "Real"]))

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data).set_index("Model")
        print("\n--- Summary Performance Table ---")
        print(comp_df.to_string())
        
        # Save comparison report
        comp_df.to_csv("models/comparison_report.csv")
        print("\nSaved comparison report table to models/comparison_report.csv")

if __name__ == "__main__":
    main()
