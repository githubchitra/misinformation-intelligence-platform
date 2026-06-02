import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch

# Constants
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS_STEP1 = 2
EPOCHS_STEP2 = 3
LR_STEP1 = 1e-3
LR_STEP2 = 2e-5

class NewsDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_len):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, item):
        statement = str(self.statements[item])
        label = self.labels[item]

        # Use modern __call__ instead of encode_plus for better compatibility
        encoding = self.tokenizer(
            statement,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'statement_text': statement,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, sum(losses) / len(data_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/val.csv'
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run preprocess.py first.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Use AutoTokenizer for standard modern implementation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = NewsDataset(train_df.statement.to_numpy(), train_df.label.to_numpy(), tokenizer, MAX_LEN)
    val_ds = NewsDataset(val_df.statement.to_numpy(), val_df.label.to_numpy(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Use AutoModel for better flexibility
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # MLflow Setup
    mlflow.set_experiment("Fake News Detection")
    
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "epochs_step1": EPOCHS_STEP1,
            "epochs_step2": EPOCHS_STEP2,
            "lr_step1": LR_STEP1,
            "lr_step2": LR_STEP2
        })

        # STEP 1: Freeze backbone, train head
        print("Step 1: Training classification head...")
        # Access distilbert backbone dynamically
        backbone = getattr(model, model.config.model_type)
        for param in backbone.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=LR_STEP1)
        total_steps = len(train_loader) * EPOCHS_STEP1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(EPOCHS_STEP1):
            train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
            val_acc, val_pre, val_rec, val_f1, val_loss = eval_model(model, val_loader, device)
            print(f"Step 1 Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            mlflow.log_metrics({"val_acc_step1": val_acc, "val_f1_step1": val_f1}, step=epoch)

        # STEP 2: Unfreeze last layers, fine-tune
        print("Step 2: Fine-tuning last layers...")
        # Unfreeze the last 2 layers of the transformer encoder
        if hasattr(backbone, 'transformer'):
            for param in backbone.transformer.layer[-2:].parameters():
                param.requires_grad = True
        
        optimizer = AdamW(model.parameters(), lr=LR_STEP2)
        total_steps = len(train_loader) * EPOCHS_STEP2
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_f1 = 0
        for epoch in range(EPOCHS_STEP2):
            train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
            val_acc, val_pre, val_rec, val_f1, val_loss = eval_model(model, val_loader, device)
            print(f"Step 2 Epoch {epoch+1}: Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            mlflow.log_metrics({"val_acc_step2": val_acc, "val_f1_step2": val_f1}, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save model
                save_dir = 'models/best_model'
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                mlflow.pytorch.log_model(model, "model")

    print("Training complete.")

if __name__ == "__main__":
    main()
