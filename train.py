# train.py
"""
Model Training Script
Fine-tunes a DistilBERT model on the processed LIAR dataset using a two-stage training strategy,
logs hyperparameters and metrics to MLflow, and saves the final model.
"""

import os
import argparse
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define custom Dataset class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_stage1", type=int, default=2, help="Epochs for Stage 1 (Backbone frozen)")
    parser.add_argument("--epochs_stage2", type=int, default=3, help="Epochs for Stage 2 (Unfrozen last 2 layers)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr_stage1", type=float, default=1e-3, help="Learning rate for Stage 1")
    parser.add_argument("--lr_stage2", type=float, default=2e-5, help="Learning rate for Stage 2")
    parser.add_argument("--quick_run", action="store_true", help="Run with a tiny subset of data for debugging")
    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Fake_News_Detection_DistilBERT")

    # Load data
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError("Processed datasets not found! Run preprocess.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # In case of quick run (for debugging/testing)
    if args.quick_run:
        print("--- RUNNING QUICK RUN MODE ---")
        train_df = train_df.sample(n=min(50, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=min(20, len(val_df)), random_state=42).reset_index(drop=True)

    # Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print("Tokenizing datasets...")
    train_encodings = tokenizer(train_df["cleaned_text"].tolist(), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_df["cleaned_text"].tolist(), truncation=True, padding=True, max_length=128)

    train_dataset = NewsDataset(train_encodings, train_df["binary_label"].tolist())
    val_dataset = NewsDataset(val_encodings, val_df["binary_label"].tolist())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Load class weights if available
    class_weights = None
    if os.path.exists("data/processed/class_weights.txt"):
        class_weights = np.loadtxt("data/processed/class_weights.txt")
        print(f"Loaded class weights: {class_weights}")
        # Note: Class weights can be passed to a custom Trainer if needed, 
        # but for simplicity in two-step training we'll rely on balanced training.

    with mlflow.start_run() as run:
        # Log Hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs_stage1", args.epochs_stage1)
        mlflow.log_param("epochs_stage2", args.epochs_stage2)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr_stage1", args.lr_stage1)
        mlflow.log_param("lr_stage2", args.lr_stage2)
        mlflow.log_param("device", device)

        # ----------------------------------------------------
        # STAGE 1: Freeze backbone, train classification head
        # ----------------------------------------------------
        print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
        
        # Freeze DistilBERT backbone parameters
        for name, param in model.named_parameters():
            if "classifier" not in name and "pre_classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        stage1_args = TrainingArguments(
            output_dir="./results/stage1",
            num_train_epochs=args.epochs_stage1,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr_stage1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs/stage1",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none"  # Prevent default Wandb or external logging
        )

        trainer_stage1 = Trainer(
            model=model,
            args=stage1_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer_stage1.train()
        eval_results_s1 = trainer_stage1.evaluate()
        print("Stage 1 Evaluation Results:", eval_results_s1)
        
        for k, v in eval_results_s1.items():
            mlflow.log_metric(f"stage1_{k}", v)

        # ----------------------------------------------------
        # STAGE 2: Unfreeze last 2 layers, fine-tune whole model
        # ----------------------------------------------------
        print("\n--- STAGE 2: Fine-Tuning Last 2 Transformer Layers ---")
        
        # Unfreeze classifier and last 2 layers of DistilBERT transformer
        # DistilBERT has 6 layers (indexed 0 to 5) in distilbert.transformer.layer
        for name, param in model.named_parameters():
            if "distilbert.transformer.layer.4" in name or "distilbert.transformer.layer.5" in name:
                param.requires_grad = True
            elif "classifier" in name or "pre_classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        stage2_args = TrainingArguments(
            output_dir="./results/stage2",
            num_train_epochs=args.epochs_stage2,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr_stage2,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs/stage2",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none"
        )

        trainer_stage2 = Trainer(
            model=model,
            args=stage2_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer_stage2.train()
        eval_results_s2 = trainer_stage2.evaluate()
        print("Stage 2 Evaluation Results:", eval_results_s2)
        
        for k, v in eval_results_s2.items():
            mlflow.log_metric(f"stage2_{k}", v)
            mlflow.log_metric(k, v) # Final metrics

        # Save model and tokenizer
        output_dir = "models/distilbert_fake_news"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved best DistilBERT model and tokenizer to {output_dir}")

        # Log model in mlflow
        mlflow.pytorch.log_model(model, "model")
        print("Log model and parameters to MLflow run: ", run.info.run_id)

if __name__ == "__main__":
    main()
