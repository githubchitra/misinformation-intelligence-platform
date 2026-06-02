import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def run_baseline():
    print("Running TF-IDF + Logistic Regression Baseline...")
    
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_df['statement'])
    X_test = tfidf.transform(test_df['statement'])
    
    y_train = train_df['label']
    y_test = test_df['label']
    
    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Eval
    preds = model.predict(X_test)
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))
    
    # Save baseline
    os.makedirs('models/baseline', exist_ok=True)
    joblib.dump(model, 'models/baseline/model.joblib')
    joblib.dump(tfidf, 'models/baseline/tfidf.joblib')
    print("Baseline model saved to models/baseline/")

if __name__ == "__main__":
    run_baseline()
