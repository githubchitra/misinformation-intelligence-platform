# explain.py
"""
Explainability (XAI) Module
Provides functions for local explanation using LIME and local & global explanations using SHAP.
Generates matplotlib plots and HTML representations.
"""

import os
import io
import base64
import matplotlib
# Use non-interactive Agg backend to avoid GUI threads in server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import shap

def get_predict_proba_fn(model, tokenizer, device):
    """
    Returns a prediction probability function suitable for LIME and SHAP text explainers.
    It takes a list of raw strings and returns a numpy array of shape (N, 2).
    """
    def predict_proba(texts):
        model.eval()
        # Handle empty inputs
        if len(texts) == 0:
            return np.array([])
            
        probs = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # DistilBERT preprocessing
                encodings = tokenizer(
                    batch_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
                encodings = {k: v.to(device) for k, v in encodings.items()}
                outputs = model(**encodings)
                batch_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                probs.append(batch_probs)
        return np.vstack(probs)
    return predict_proba

def explain_with_lime(text: str, model, tokenizer, device, num_features: int = 10):
    """
    Generates a local explanation using LIME.
    Returns:
        prediction_label (str): "Real" or "Fake"
        probability (float): confidence score of prediction
        html_str (str): LIME explanation exported as HTML
        list_explanation (list): Word-weight tuples
    """
    predict_proba = get_predict_proba_fn(model, tokenizer, device)
    
    # Get direct prediction info
    probs = predict_proba([text])[0]
    fake_prob, real_prob = probs[0], probs[1]
    pred_label = "Real" if real_prob >= 0.5 else "Fake"
    confidence = real_prob if real_prob >= 0.5 else fake_prob

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    
    # Generate explanation
    exp = explainer.explain_instance(
        text, 
        predict_proba, 
        num_features=num_features,
        labels=[1]  # Explain class "Real"
    )
    
    html_str = exp.as_html()
    list_explanation = exp.as_list(label=1)
    
    return pred_label, float(confidence), html_str, list_explanation

def explain_with_shap(text: str, model, tokenizer, device):
    """
    Generates a local explanation using SHAP.
    Returns a matplotlib figure base64 string showing token contributions.
    """
    predict_proba = get_predict_proba_fn(model, tokenizer, device)
    
    # SHAP explainer for text
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker)
    
    # Compute SHAP values
    shap_values = explainer([text])
    
    # Extract tokens and values for class 1 (Real)
    # shap_values.values has shape (1, num_tokens, 2)
    tokens = shap_values.data[0]
    # For class 'Real' (index 1)
    values = shap_values.values[0, :, 1]
    
    # Filter out empty tokens or padding
    valid_indices = [i for i, t in enumerate(tokens) if t.strip() != ""]
    tokens = [tokens[i] for i in valid_indices]
    values = [values[i] for i in valid_indices]

    # Create a nice horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.4)))
    
    colors = ["#3182bd" if val >= 0 else "#e34a33" for val in values] # Blue for Real, Red for Fake
    y_pos = np.arange(len(tokens))
    
    ax.barh(y_pos, values, align="center", color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("SHAP Value (Impact on 'Real' Prediction)", fontsize=11)
    ax.set_title(f"SHAP Local Explanation\nRed: Contributes to 'Fake' | Blue: Contributes to 'Real'", fontsize=12, pad=15)
    
    plt.tight_layout()
    
    # Save figure to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return img_b64

def explain_global_shap(val_texts: list, model, tokenizer, device, num_samples: int = 15):
    """
    Generates a global feature importance plot using SHAP across multiple texts.
    Returns a matplotlib figure base64 string.
    """
    predict_proba = get_predict_proba_fn(model, tokenizer, device)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker)
    
    # Run SHAP on the validation subset
    subset = val_texts[:num_samples]
    shap_values = explainer(subset)
    
    # Sum absolute SHAP values for each unique token to get global importance
    token_importance = {}
    for sample_idx in range(len(subset)):
        tokens = shap_values.data[sample_idx]
        # Get values for class 1 (Real)
        values = shap_values.values[sample_idx, :, 1]
        
        for token, val in zip(tokens, values):
            token = token.strip().lower()
            if token == "" or len(token) < 2:  # Skip empty or single-character noise
                continue
            token_importance[token] = token_importance.get(token, 0.0) + abs(val)
            
    # Sort and take top 15
    sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    if not sorted_tokens:
        sorted_tokens = [("dummy", 0.0)]
        
    top_tokens, top_values = zip(*sorted_tokens)
    
    # Plot global feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(top_tokens))
    
    ax.barh(y_pos, top_values, align="center", color="#7570b3", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute SHAP Value (Global Impact)", fontsize=11)
    ax.set_title("SHAP Global Feature Importance (Top Words)", fontsize=13, pad=15)
    
    plt.tight_layout()
    
    # Save figure to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return img_b64
