# ui.py
"""
Streamlit Web Interface
Connects to the FastAPI backend to display predictions, interactive explanations,
and Chain-of-Thought RAG fact-checking results.
"""

import base64
import requests
import streamlit as st
import streamlit.components.v1 as components

# API base URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FakeNews AI - Detection & XAI Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    h1, h2, h3 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .card-real {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .card-fake {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .card-neutral {
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to convert base64 image string to bytes
def get_image_bytes(b64_str):
    return base64.b64decode(b64_str)

# Cached API Calls to prevent redundant network overhead
@st.cache_data(show_spinner=False)
def call_predict_api(text: str):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server. Please make sure it is running on http://localhost:8000")
    return None

@st.cache_data(show_spinner=False)
def call_explain_api(text: str):
    try:
        response = requests.post(f"{API_URL}/explain", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server.")
    return None

@st.cache_data(show_spinner=False)
def call_factcheck_api(text: str):
    try:
        response = requests.post(f"{API_URL}/factcheck", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server.")
    return None

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/fingerprint.png", width=100)
    st.title("FakeNews AI Control Panel")
    st.markdown("---")
    st.markdown("### System Health Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success("🟢 API Server Online")
        st.caption(f"Device: {health_resp['device'].upper()}")
        st.caption(f"Model Loaded: {'Yes' if health_resp['model_loaded'] else 'No'}")
        st.caption(f"Redis Cache: {'Active' if health_resp['redis_connected'] else 'Inactive (Fallback active)'}")
    except Exception:
        st.error("🔴 API Server Offline")
    st.markdown("---")
    st.markdown("### Architecture Specs")
    st.caption("**Backbone**: DistilBERT-base-uncased")
    st.caption("**XAI**: LIME Text Explainer & SHAP Explainer")
    st.caption("**Serving**: FastAPI + Redis Cache")
    st.caption("**Search Retriever**: Serper Google Search API")

# Main Page Header
st.title("🔍 Fake News Detection & Explainability Platform")
st.markdown("Assess the truthfulness of statements, visualize model explanations, and verify facts in real-time.")

tab1, tab2 = st.tabs(["⚡ AI Classifier & Explainer", "🧠 RAG Real-Time Fact-Checker"])

# TAB 1: CLASSIFIER & EXPLANATION
with tab1:
    st.subheader("Statement Classification")
    text_input = st.text_area(
        "Enter a news headline, quote, or claim to evaluate:",
        value="The economy grew by 5% last quarter, a record high.",
        height=100,
        help="Text length must be between 10 and 5000 characters."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        predict_clicked = st.button("Predict Verdict")
    with col2:
        explain_clicked = st.button("Generate LIME/SHAP Explanations")
        
    if predict_clicked or explain_clicked:
        if len(text_input) < 10:
            st.warning("Please enter at least 10 characters to perform predictions.")
        else:
            with st.spinner("Analyzing statement..."):
                pred_result = call_predict_api(text_input)
                
            if pred_result:
                label = pred_result["prediction"]
                conf = pred_result["confidence"]
                cached = pred_result["cached"]
                
                # Card display based on verdict
                if label == "Real":
                    st.markdown(f"""
                        <div class="card-real">
                            <h3>✅ Verdict: REAL ({conf*100:.2f}% Confidence)</h3>
                            <p>This statement shows patterns aligned with verified factual reporting.</p>
                            <small>{'⚡ Retrieved from cache' if cached else '🤖 Computed in real-time'}</small>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="card-fake">
                            <h3>🚨 Verdict: FAKE ({conf*100:.2f}% Confidence)</h3>
                            <p>This statement contains text patterns indicative of mis/disinformation.</p>
                            <small>{'⚡ Retrieved from cache' if cached else '🤖 Computed in real-time'}</small>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show breakdown
                st.write("**Probability Distribution:**")
                st.progress(pred_result["probabilities"]["Real"])
                st.caption(f"Real: {pred_result['probabilities']['Real']*100:.1f}% | Fake: {pred_result['probabilities']['Fake']*100:.1f}%")
                
            # If explain button is clicked, trigger API /explain
            if explain_clicked:
                with st.spinner("Generating local word-level explanations..."):
                    explain_result = call_explain_api(text_input)
                    
                if explain_result:
                    st.markdown("---")
                    st.subheader("Model Interpretability Analysis")
                    st.markdown("This section explains *why* the model made this prediction. We run two distinct Explainable AI (XAI) models: LIME and SHAP.")
                    
                    # Split into columns for explanations
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        st.markdown("### LIME Word Highlights")
                        st.markdown("LIME trains local surrogate models by perturbing individual words. Words highlighted in **orange/red** pull towards *Fake*, and **blue** towards *Real*.")
                        components.html(explain_result["lime_html"], height=420, scrolling=True)
                        
                    with exp_col2:
                        st.markdown("### SHAP Token Impacts")
                        st.markdown("SHAP computes Shapley values to assign mathematical contributions of each word token to the final probability output.")
                        try:
                            img_bytes = get_image_bytes(explain_result["shap_plot_b64"])
                            st.image(img_bytes, caption="SHAP value token breakdown (impact on predicting 'Real')", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not render SHAP plot: {e}")

# TAB 2: RAG REAL-TIME FACT-CHECKER
with tab2:
    st.subheader("Retrieval-Augmented Fact Verification")
    st.markdown("This module parses your text, extracts its main factual claim, searches the web for live context, and conducts Chain-of-Thought verification.")
    
    fact_input = st.text_area(
        "Paste the full article content to fact-check:",
        value="The governor signed a bipartisan tax cut bill yesterday.",
        height=150,
        help="Provide the context or paragraph containing statements to verify."
    )
    
    check_clicked = st.button("Fact Check via RAG")
    
    if check_clicked:
        if len(fact_input) < 10:
            st.warning("Please enter at least 10 characters to run fact-checking.")
        else:
            with st.spinner("Processing RAG verification pipeline..."):
                fc_res = call_factcheck_api(fact_input)
                
            if fc_res:
                verdict = fc_res["verdict"]
                claim = fc_res["extracted_claim"]
                snippets = fc_res["search_results"]
                reasoning = fc_res["reasoning"]
                
                # Style verdict card
                if verdict == "Real":
                    card_class = "card-real"
                    verdict_icon = "✅"
                elif verdict == "Fake":
                    card_class = "card-fake"
                    verdict_icon = "🚨"
                else:
                    card_class = "card-neutral"
                    verdict_icon = "❓"
                    
                st.markdown(f"""
                    <div class="{card_class}">
                        <h3>{verdict_icon} RAG Verdict: {verdict.upper()}</h3>
                        <p><strong>Extracted Core Claim:</strong> {claim}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### Retrieved Web Context")
                    st.markdown("Top matching search snippets used as ground truth verification:")
                    for idx, snippet in enumerate(snippets):
                        st.info(f"**Result {idx+1}:** {snippet}")
                        
                with col_right:
                    st.markdown("### Chain-of-Thought Reasoning")
                    st.markdown("Step-by-step audit showing how the system cross-referenced statements:")
                    st.write(reasoning)
