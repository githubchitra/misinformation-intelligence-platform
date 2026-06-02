import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ClaimExtractor:
    def __init__(self, model_name="google/flan-t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def extract(self, text):
        """
        Extracts a concise claim from the given text using FLAN-T5.
        """
        prompt = f"Extract the primary factual claim from this text as a single concise sentence: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50, 
                num_beams=4, 
                early_stopping=True
            )
        
        extracted_claim = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Fallback if extraction is too short or failed
        if len(extracted_claim.split()) < 3:
            return text
            
        return extracted_claim

if __name__ == "__main__":
    extractor = ClaimExtractor()
    test_text = "Virat Kohli is the world chess champion"
    print(f"Original: {test_text}")
    print(f"Claim: {extractor.extract(test_text)}")
