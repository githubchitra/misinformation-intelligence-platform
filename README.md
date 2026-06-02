# 🛡️ Misinformation Intelligence Platform (MIP)

This platform provides an end-to-end suite for detecting, explaining, and fact-checking news misinformation. It combines deep learning (DistilBERT), explainable AI (LIME/SHAP), and real-time retrieval-augmented fact-checking (RAG).

**Live Demo**: [https://misinformation-intelligence-platform.streamlit.app/](https://misinformation-intelligence-platform.streamlit.app/)

## 🚀 Architecture
- **Model**: DistilBERT fine-tuned on the LIAR dataset.
- **Explainability**: LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations).
- **Backend**: FastAPI for high-performance REST endpoints.
- **Frontend**: Streamlit for an interactive user interface.
- **Ops**: MLflow for experiment tracking, DVC for data versioning, and Docker for containerization.
- **Advanced**: RAG-based fact-checking using LangChain and OpenAI.

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/githubchitra/misinformation-intelligence-platform.git
cd FNDM

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation & Versioning
```bash
# Initialize DVC
dvc init

# Download and preprocess data
python src/preprocess.py

# Version data with DVC
dvc add data/processed
git add data/processed.dvc .gitignore
```

### 3. Model Training (Local or Colab)

#### Local Training
```bash
# Launch MLflow UI in a separate terminal
mlflow ui

# Run training (2-step fine-tuning)
python src/train.py
```

#### GPU Training with Google Colab (Recommended for speed)
If you don't have a local GPU, you can use Google Colab:
1. Upload the `notebooks/colab_training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU: `Runtime -> Change runtime type -> T4 GPU`.
3. Upload the `src/` folder to the Colab workspace.
4. Run all cells to preprocess, train, and download the model.
5. Extract the downloaded `best_model.zip` into your local `models/` directory.

### 4. Running with Docker (Recommended)
```bash
# Set your OpenAI API key for RAG
export OPENAI_API_KEY="your_key_here"

# Build and start services
docker-compose up --build
```
- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501
- **MLflow**: http://localhost:5000

## 📊 Evaluation
To compare the transformer model with the baseline:
```bash
# Run baseline
python src/baseline.py

# Run transformer evaluation
python src/evaluate.py
```

## 🧩 Browser Extension
1. Open Chrome and navigate to `chrome://extensions`.
2. Enable "Developer mode".
3. Click "Load unpacked" and select the `extension/` folder.
4. Select text on any news site and click "Check News".

## 📝 Design Decisions
- **DistilBERT**: Chosen over BERT-base for 40% smaller size and 60% faster inference while retaining 97% of the performance.
- **LIME vs SHAP**: LIME provides fast, word-level local explanations, while SHAP offers a more theoretically grounded approach to feature importance.
- **FastAPI**: Used for its asynchronous support and automatic OpenAPI documentation.
- **RAG**: Implemented as a stretch goal to provide real-time evidence verification from the web.
