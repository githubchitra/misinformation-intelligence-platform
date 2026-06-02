# preprocess.py
"""
Data Preprocessing Script
Downloads the LIAR dataset, performs text cleaning, maps multi-class labels 
to binary ('real' vs 'fake'), computes class weights, and saves raw & processed data.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("omw-1.4")

def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Cleans input text by removing special characters, lowercasing, and lemmatizing.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags/URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    
    # Remove special characters and numbers (keep only words and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        words = text.split()
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        text = " ".join(words)
        
    return text

def map_liar_label(label_idx: int) -> int:
    """
    Maps 6-class LIAR label index to binary:
    Original:
    0: false, 1: half-true, 2: mostly-true, 3: true, 4: barely-true, 5: pants-fire
    
    Mapping:
    - 1 ("real"): true (3), mostly-true (2), half-true (1)
    - 0 ("fake"): false (0), barely-true (4), pants-fire (5)
    """
    # 1, 2, 3 -> real (1)
    # 0, 4, 5 -> fake (0)
    if label_idx in [1, 2, 3]:
        return 1
    return 0

def preprocess_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemmatize", action="store_true", default=True, help="Use lemmatization")
    args = parser.parse_args()

    print("Step 1: Downloading LIAR dataset using Hugging Face datasets...")
    # Using 'liar' dataset
    try:
        dataset = load_dataset("liar")
    except Exception as e:
        print(f"Error downloading LIAR dataset: {e}")
        print("Creating a mock/dummy dataset for demonstration...")
        dataset = generate_dummy_dataset()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    splits = ["train", "validation", "test"]
    processed_dfs = {}

    for split in splits:
        df = pd.DataFrame(dataset[split])
        
        # Save raw data
        raw_path = f"data/raw/raw_{split}.csv"
        df.to_csv(raw_path, index=False)
        print(f"Saved raw {split} dataset to {raw_path}")

        # Process labels and text
        # The LIAR dataset column for text is 'statement' and label is 'label'
        df["cleaned_text"] = df["statement"].apply(lambda x: clean_text(x, lemmatize=args.lemmatize))
        df["binary_label"] = df["label"].apply(map_liar_label)
        
        # Filter out empty statements after cleaning
        df = df[df["cleaned_text"] != ""]
        
        # Save processed data
        processed_path = f"data/processed/{split}.csv"
        df[["cleaned_text", "binary_label"]].to_csv(processed_path, index=False)
        print(f"Saved processed {split} dataset to {processed_path}")
        processed_dfs[split] = df

    # Calculate class weights on the training set
    train_labels = processed_dfs["train"]["binary_label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    print("\n--- Preprocessing Summary ---")
    print(f"Train samples: {len(processed_dfs['train'])}")
    print(f"Val samples: {len(processed_dfs['validation'])}")
    print(f"Test samples: {len(processed_dfs['test'])}")
    print(f"Class imbalance on Train: {np.bincount(train_labels)}")
    print(f"Calculated Class Weights (balanced): {class_weights}")
    
    # Save class weights to a text file for the trainer
    np.savetxt("data/processed/class_weights.txt", class_weights)
    print("Saved class weights to data/processed/class_weights.txt")

def generate_dummy_dataset():
    """
    Generates dummy dataset in the LIAR dataset schema for fallback / offline execution.
    """
    dummy_data = {
        "train": {
            "statement": [
                "The economy grew by 5% last quarter, a record high.",
                "Aliens landed in New York yesterday and bought bagels.",
                "Vaccine mandates have saved thousands of lives nationwide.",
                "Drinking bleach cures the common cold in two hours.",
                "The governor signed a bipartisan tax cut bill yesterday.",
                "A local candidate stole millions from a children charity fund."
            ],
            "label": [3, 0, 2, 5, 1, 4] # Mix of real and fake index labels
        },
        "validation": {
            "statement": [
                "Unemployment fell to a 50-year low of 3.5 percent.",
                "The mayor was spotted flying a UFO over city hall."
            ],
            "label": [3, 5]
        },
        "test": {
            "statement": [
                "Congress passed a landmark infrastructure investment act.",
                "The moon is actually made of Swiss cheese."
            ],
            "label": [3, 0]
        }
    }
    return dummy_data

if __name__ == "__main__":
    preprocess_pipeline()
