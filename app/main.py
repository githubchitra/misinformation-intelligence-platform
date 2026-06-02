# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import sys

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.explainability import XAIExplainer
from src.claim_extractor import ClaimExtractor
from src.web_search import WebSearcher
from src.fact_check import FactChecker

app = FastAPI(title="Fake News Detection API")

# Model path
MODEL_PATH = "models/models/best_model"

# Global variables
model = None
tokenizer = None
explainer = None
claim_extractor = None
searcher = None
fact_checker = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    global model, tokenizer, explainer, claim_extractor, searcher, fact_checker
    print(f"Starting model load from {MODEL_PATH}...")
    
    if os.path.exists(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        explainer = XAIExplainer(model, tokenizer, device)
    
    claim_extractor = ClaimExtractor()
    searcher = WebSearcher()
    fact_checker = FactChecker()
    print("All components loaded successfully.")

class NewsRequest(BaseModel):
    text: str
    threshold: float = 0.8
    always_factcheck: bool = True

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class SourceDetail(BaseModel):
    title: str
    link: str
    snippet: str

class FactCheckDetail(BaseModel):
    extracted_claim: str
    verdict: str
    confidence: float
    sources: List[SourceDetail]

class HybridResponse(BaseModel):
    original_prediction: PredictionResponse
    fact_check: Optional[FactCheckDetail] = None
    final_prediction: str
    override_reason: Optional[str] = None
    fact_check_triggered: bool = True
    threshold_note: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewsRequest):
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return PredictionResponse(label="Real" if pred.item() == 1 else "Fake", confidence=conf.item())

@app.post("/factcheck", response_model=HybridResponse)
def fact_check_endpoint(request: NewsRequest):
    # 1. DistilBERT First Pass
    orig_pred = predict(request)
    
    # 2. Always run fact-check logic as requested
    print(f"Running fact-check for: {request.text}")
    claim = claim_extractor.extract(request.text)
    evidence = searcher.run_search(claim)
    check_result = fact_checker.check_claim(claim, evidence)
    
    # 3. Determine if we should override based on threshold
    final_label = orig_pred.label
    reason = None
    note = None
    
    # Logic: Fact-check result can override the AI prediction.
    # If Fact-check says Contradicted, we label as Fake.
    # If Fact-check says Supported, we label as Real.
    # We apply this override if AI confidence is high enough OR if AI is uncertain.
    
    if check_result['verdict'] == "Contradicted":
        if orig_pred.label == "Real" and orig_pred.confidence >= request.threshold:
            final_label = "Fake"
            reason = f"External sources contradict the claim (Evidence confidence: {check_result['confidence']:.2f})."
        elif orig_pred.label == "Real":
            note = f"Fact-check suggests contradiction, but AI confidence ({orig_pred.confidence:.2f}) is below override threshold ({request.threshold:.2f})."
    
    elif check_result['verdict'] == "Supported":
        if orig_pred.label == "Fake" and orig_pred.confidence >= request.threshold:
            final_label = "Real"
            reason = f"External sources support the claim (Evidence confidence: {check_result['confidence']:.2f})."
        elif orig_pred.label == "Fake":
            note = f"Fact-check suggests support, but AI confidence ({orig_pred.confidence:.2f}) is below override threshold ({request.threshold:.2f})."

    return HybridResponse(
        original_prediction=orig_pred,
        fact_check=FactCheckDetail(
            extracted_claim=claim,
            verdict=check_result['verdict'],
            confidence=check_result['confidence'],
            sources=[SourceDetail(**e) for e in evidence]
        ),
        final_prediction=final_label,
        override_reason=reason,
        fact_check_triggered=True,
        threshold_note=note
    )

@app.post("/explain", response_model=Dict)
def explain(request: NewsRequest):
    if not model or not explainer: raise HTTPException(status_code=500, detail="Model not loaded")
    lime_html = explainer.explain_with_lime(request.text)
    shap_base64 = explainer.explain_with_shap(request.text)
    return {"lime_html": lime_html, "shap_base64": shap_base64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
