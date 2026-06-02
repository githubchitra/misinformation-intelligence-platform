# ui/app.py
import streamlit as st
import requests
import streamlit.components.v1 as components
import base64

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("🛡️ Hybrid AI Fake News Detection & Fact-Checker")
st.markdown("""
This system combines a **DistilBERT classifier** with **Retrieval-Augmented Fact-Checking**.
1. **Pass 1**: DistilBERT classifies the text based on linguistic patterns.
2. **Pass 2**: A web search is **always** performed to find external evidence and verify the claim.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
threshold = st.sidebar.slider("Override Threshold", 0.5, 1.0, 0.8, 
                             help="If AI confidence is above this, fact-check results can override the prediction.")

# Main Input
text_input = st.text_area("Enter news article or statement:", height=150, placeholder="Type or paste news text here...")

if st.button("Analyze & Fact-Check"):
    if text_input:
        with st.spinner("Analyzing with AI and searching the web..."):
            try:
                # Call Hybrid Fact-Check endpoint (always run both now)
                res = requests.post(f"{api_url}/factcheck", json={"text": text_input, "threshold": threshold})
                if res.status_code == 200:
                    data = res.json()
                    
                    orig = data['original_prediction']
                    fc = data.get('fact_check')
                    final_label = data['final_prediction']
                    
                    # 1. Show Original Prediction
                    st.subheader("1. AI Classifier Result")
                    if orig['label'] == "Real":
                        st.success(f"DistilBERT Prediction: **{orig['label']}** (Confidence: {orig['confidence']:.2%})")
                    else:
                        st.error(f"DistilBERT Prediction: **{orig['label']}** (Confidence: {orig['confidence']:.2%})")

                    # 2. Show Fact-Check Result (Always displayed now)
                    st.subheader("2. Web Fact-Check Result")
                    if fc:
                        st.write(f"**Extracted Claim:** {fc['extracted_claim']}")
                        
                        if fc['verdict'] == "Contradicted":
                            st.error(f"Verdict: **{fc['verdict']}** (Evidence Confidence: {fc['confidence']:.2%})")
                        elif fc['verdict'] == "Supported":
                            st.success(f"Verdict: **{fc['verdict']}** (Evidence Confidence: {fc['confidence']:.2%})")
                        else:
                            st.info(f"Verdict: **{fc['verdict']}**")
                            
                        if data.get('threshold_note'):
                            st.info(data['threshold_note'])

                        with st.expander("View Sources", expanded=True):
                            if fc.get('sources') and len(fc['sources']) > 0:
                                for source in fc['sources']:
                                    st.markdown(f"### [{source['title']}]({source['link']})")
                                    st.write(source['snippet'])
                                    st.divider()
                            else:
                                st.write("No sources found. Try a more specific claim or check your search API.")
                    else:
                        st.write("Fact-check data unavailable.")

                    # 3. Final Verdict
                    st.divider()
                    st.subheader("Final Verdict")
                    if final_label == "Real":
                        st.success(f"Result: **{final_label}**")
                    else:
                        st.error(f"Result: **{final_label}**")
                    
                    if data.get('override_reason'):
                        st.warning(f"**Override Applied:** {data['override_reason']}")
                    
                    st.session_state['text'] = text_input
                else:
                    st.error("Error connecting to the API.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text to analyze.")

if 'text' in st.session_state:
    if st.button("Explain with LIME/SHAP"):
        with st.spinner("Generating explanations (this may take a moment)..."):
            try:
                res = requests.post(f"{api_url}/explain", json={"text": st.session_state['text']})
                if res.status_code == 200:
                    data = res.json()
                    st.divider()
                    st.subheader("Explainability Reports")
                    tab1, tab2 = st.tabs(["LIME (Word Importance)", "SHAP (Token Impact)"])
                    with tab1:
                        st.markdown("### LIME Explanation")
                        components.html(data['lime_html'], height=500, scrolling=True)
                    with tab2:
                        st.markdown("### SHAP Explanation")
                        img_bytes = base64.b64decode(data['shap_base64'])
                        st.image(img_bytes, caption="SHAP Text Explanation", use_column_width=True)
                else:
                    st.error("Error generating explanation.")
            except Exception as e:
                st.error(f"Error: {e}")
