import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Load model and tokenizer
    model_path = 'models/best_model'
    if not os.path.exists(model_path):
        print("Model not found. Please run train.py first.")
        return

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    statements = test_df.statement.tolist()
    labels = test_df.label.tolist()
    
    all_preds = []
    
    print("Generating predictions for test set...")
    with torch.no_grad():
        for i in range(0, len(statements), 16):
            batch_statements = statements[i:i+16]
            encodings = tokenizer(batch_statements, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # Metrics
    print("\nClassification Report:")
    report = classification_report(labels, all_preds, target_names=['Fake', 'Real'])
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
