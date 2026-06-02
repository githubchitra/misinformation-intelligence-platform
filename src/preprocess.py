import pandas as pd
import re
import string
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Cleans text by removing special characters, lowercasing, and stripping whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data():
    print("Loading reliable fake news dataset from Hugging Face...")
    
    # We use 'GonzaloA/fake_news' as it is a stable, Parquet-based dataset 
    # that does not rely on deprecated loading scripts.
    # It contains columns: 'title', 'text', 'label'
    # Labels: 0 = fake, 1 = true
    try:
        dataset = load_dataset("GonzaloA/fake_news", trust_remote_code=True)
        df = pd.DataFrame(dataset['train'])
    except Exception as e:
        print(f"Error loading primary dataset: {e}")
        print("Attempting alternative stable dataset...")
        # Fallback to another very stable dataset if GonzaloA is down
        dataset = load_dataset("mosharaf2k/fake-news-dataset", trust_remote_code=True)
        df = pd.DataFrame(dataset['train'])
        # Rename columns if necessary to match our expected format
        if 'text' not in df.columns and 'statement' in df.columns:
            df = df.rename(columns={'statement': 'text'})

    print(f"Dataset loaded. Total rows: {len(df)}")

    # 1. Clean and Prepare
    # We combine title and text for better context, or just use text.
    # Here we use 'text' as the primary statement.
    df = df.dropna(subset=['text', 'label'])
    df['statement'] = df['text'].apply(clean_text)
    
    # Ensure label is binary (0 = fake, 1 = real)
    # GonzaloA/fake_news is already 0: fake, 1: true
    df['label'] = df['label'].astype(int)

    # 2. Split Data (70% Train, 15% Val, 15% Test)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    # 3. Save to data/processed/
    os.makedirs('data/processed', exist_ok=True)
    
    cols = ['statement', 'label']
    train_df[cols].to_csv('data/processed/train.csv', index=False)
    val_df[cols].to_csv('data/processed/val.csv', index=False)
    test_df[cols].to_csv('data/processed/test.csv', index=False)
    
    print(f"Preprocessing complete!")
    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    print("Files saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()
