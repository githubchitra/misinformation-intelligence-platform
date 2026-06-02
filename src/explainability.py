import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class XAIExplainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.class_names = ['Fake', 'Real']

    def predictor(self, texts):
        """
        Predictor function for LIME.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

    def explain_with_lime(self, text):
        explainer = LimeTextExplainer(class_names=self.class_names)
        exp = explainer.explain_instance(text, self.predictor, num_features=10)
        
        # Save to HTML
        html_content = exp.as_html()
        return html_content

    def explain_with_shap(self, text):
        """
        SHAP explanation using the text explainer.
        """
        # Define a prediction function for SHAP
        def f(x):
            tv = self.tokenizer(x.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            outputs = self.model(**tv)[0]
            scores = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
            return scores

        explainer = shap.Explainer(f, self.tokenizer)
        shap_values = explainer([text])
        
        # Plotting SHAP values
        plt.figure(figsize=(10, 5))
        shap.plots.text(shap_values[0])
        
        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
