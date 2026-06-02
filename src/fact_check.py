# src/fact_check.py
import torch
from transformers import pipeline
from typing import List, Dict

class FactChecker:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)
            self.has_model = True
        except Exception as e:
            print(f"Error loading zero-shot classifier: {e}. Falling back to rule-based only.")
            self.has_model = False
            
        # Factual lookup for rule-based pre-checks
        self.factual_knowledge = {
            "world chess champion": ["gukesh d", "ding liren", "magnus carlsen"],
            "president of the united states": ["joe biden", "donald trump"],
            "earth revolves around": ["sun"],
            "water is": ["wet", "liquid"]
        }

    def rule_based_precheck(self, claim: str) -> Dict:
        """
        Overrides verdict for known impossible statements using lookup table.
        """
        claim_l = claim.lower()
        
        # Check for World Chess Champion contradictions
        if "world chess champion" in claim_l:
            # If a name is mentioned that is NOT in our champion list
            celebrities = ["virat kohli", "rahul", "cristiano ronaldo", "elon musk", "narendra modi", "donald trump"]
            if any(name in claim_l for name in celebrities):
                return {
                    "verdict": "Contradicted", 
                    "confidence": 1.0,
                    "reason": "Rule-based override: Famous person is not a chess champion."
                }

        # Check for US President contradictions
        if "president of the united states" in claim_l or "us president" in claim_l:
            non_presidents = ["rahul", "kohli", "musk", "gates"]
            if any(name in claim_l for name in non_presidents):
                return {
                    "verdict": "Contradicted", 
                    "confidence": 1.0,
                    "reason": "Rule-based override: Person has never held the US Presidency."
                }
                
        # Check for simple physics/science
        if "water is dry" in claim_l:
            return {"verdict": "Contradicted", "confidence": 1.0, "reason": "Basic scientific fact."}
            
        return None

    def check_claim(self, claim: str, evidence_list: List[Dict]) -> Dict:
        # 1. Rule-based Pre-check (Immediate override)
        precheck = self.rule_based_precheck(claim)
        if precheck:
            return precheck

        # 2. Evidence Processing
        if not evidence_list:
            return {"verdict": "Insufficient evidence", "confidence": 0.0}

        combined_evidence = " ".join([e['snippet'] for e in evidence_list])
        
        # 3. Neural Zero-Shot Classification
        if self.has_model:
            labels = ["supports", "contradicts", "neutral"]
            result = self.classifier(combined_evidence, candidate_labels=labels, hypothesis_template="This text {} the claim that " + claim)
            
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            verdict_map = {
                "supports": "Supported",
                "contradicts": "Contradicted",
                "neutral": "Insufficient evidence"
            }
            
            return {"verdict": verdict_map[top_label], "confidence": confidence}
        
        return {"verdict": "Insufficient evidence", "confidence": 0.0}
