# app.py
"""
FastAPI Server
Exposes REST endpoints for prediction, LIME & SHAP explanation, and fact-checking.
Incorporates Redis caching (with in-memory fallback) and proper error handlers.
"""

import os
import hashlib
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Import explain and fact_check modules
import explain
import fact_check

# Cache configurations
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Global variables
model = None
tokenizer = None
device = "cpu"
redis_client = None
in_memory_cache = {}  # Fallback prediction cache
fact_checker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler for startup/shutdown.
    Ensures heavy models are loaded once.
    """
    global model, tokenizer, device, redis_client, fact_checker
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Server starting. Using device: {device}")
    
    # Initialize Redis with a timeout to prevent startup hangs
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        redis_client.ping()
        print("Connected to Redis successfully.")
    except Exception as e:
        print(f"Redis is unavailable: {e}. Falling back to in-memory caching.")
        redis_client = None

    # Load model and tokenizer
    model_path = "models/distilbert_fake_news"
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}...")
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}. Loading base model as fallback.")
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    else:
        print("Fine-tuned model directory not found. Loading base DistilBERT for demo.")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
    model.to(device)
    model.eval()
    
    # Initialize RAG FactChecker
    fact_checker = fact_check.FactChecker()
    
    yield
    # Shutdown logic (close redis if needed)
    if redis_client:
        redis_client.close()

app = FastAPI(
    title="Fake News Detection API",
    description="A production-ready REST API for fake news classification and ML explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Chrome Extension and Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class ArticleRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=5000, 
        description="The news article text to verify. Must be between 10 and 5000 characters."
    )

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: dict
    cached: bool

class ExplanationResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    lime_html: str
    shap_plot_b64: str

class FactCheckResponse(BaseModel):
    text: str
    extracted_claim: str
    search_results: list
    reasoning: str
    verdict: str

# Helper functions
def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_prediction(text_hash: str) -> dict:
    if redis_client:
        try:
            cached_val = redis_client.get(f"pred:{text_hash}")
            if cached_val:
                return json.loads(cached_val)
        except Exception:
            pass
    return in_memory_cache.get(text_hash)

def set_cached_prediction(text_hash: str, data: dict):
    if redis_client:
        try:
            # Cache for 24 hours
            redis_client.setex(f"pred:{text_hash}", 86400, json.dumps(data))
            return
        except Exception:
            pass
    in_memory_cache[text_hash] = data

# Endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Service health check.
    """
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None,
        "redis_connected": redis_client is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ArticleRequest):
    """
    Predicts whether a news article is Real or Fake.
    Results are cached in Redis.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is currently loading or failed to initialize."
        )
        
    text_hash = get_text_hash(request.text)
    cached_data = get_cached_prediction(text_hash)
    
    if cached_data:
        cached_data["cached"] = True
        return cached_data

    try:
        # Run inference
        encodings = tokenizer(
            request.text, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        pred_label = "Real" if real_prob >= 0.5 else "Fake"
        confidence = real_prob if real_prob >= 0.5 else fake_prob
        
        res_data = {
            "text": request.text,
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": {
                "Fake": fake_prob,
                "Real": real_prob
            },
            "cached": False
        }
        
        # Save to cache
        set_cached_prediction(text_hash, res_data)
        return res_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}"
        )

@app.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(request: ArticleRequest):
    """
    Runs prediction and returns word-level visualisations (LIME HTML and SHAP plot).
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded."
        )

    try:
        # 1. Run LIME explanation
        pred_label, confidence, lime_html, _ = explain.explain_with_lime(
            request.text, model, tokenizer, device
        )
        
        # 2. Run SHAP explanation
        shap_plot_b64 = explain.explain_with_shap(
            request.text, model, tokenizer, device
        )
        
        return {
            "text": request.text,
            "prediction": pred_label,
            "confidence": confidence,
            "lime_html": lime_html,
            "shap_plot_b64": shap_plot_b64
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability module error: {str(e)}"
        )

@app.post("/factcheck", response_model=FactCheckResponse)
def factcheck_endpoint(request: ArticleRequest):
    """
    Runs RAG fact-checking: claim extraction, web search retrieval, and Chain-of-Thought verification.
    """
    if fact_checker is None:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Fact-checking module is not initialized."
        )
    try:
        res = fact_checker.run(request.text)
        return {
            "text": request.text,
            "extracted_claim": res["claim"],
            "search_results": res["search_results"],
            "reasoning": res["reasoning"],
            "verdict": res["verdict"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact-checking module failure: {str(e)}"
        )
