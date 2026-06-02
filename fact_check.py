# fact_check.py
"""
RAG-Based Fact-Checking Module
Extracts claims, fetches search results from web APIs (with mock fallback), 
and uses Chain-of-Thought reasoning to verify statements.
"""

import os
import requests
from transformers import pipeline

class FactChecker:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initializes a fact-checking pipeline. We use 'google/flan-t5-small' by default
        for memory efficiency in Docker container settings (runnable on CPU).
        """
        self.model_name = model_name
        self._llm = None
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

    @property
    def llm(self):
        if self._llm is None:
            print(f"Loading LLM '{self.model_name}' for claim extraction and CoT...")
            try:
                # Use text2text-generation pipeline for Flan-T5
                self._llm = pipeline("text2text-generation", model=self.model_name, device=-1) # Run on CPU
            except Exception as e:
                print(f"Error loading local LLM: {e}. Falling back to rule-based mock solver.")
                self._llm = "mock"
        return self._llm

    def extract_claims(self, text: str) -> str:
        """
        Uses LLM to extract the main claim from an article.
        """
        prompt = f"Extract the single most testable factual claim from this article: '{text}'. Claim:"
        
        if self.llm == "mock":
            # Simple rule-based mock: return first sentence
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
            return sentences[0] if sentences else text
            
        try:
            res = self.llm(prompt, max_length=64, num_return_sequences=1)
            claim = res[0]["generated_text"].strip()
            return claim
        except Exception as e:
            print(f"Error during claim extraction: {e}")
            return text

    def search_web(self, query: str) -> list:
        """
        Retrieves search results from Serper API (or falls back to mock search).
        """
        if not self.serper_api_key:
            print("No SERPER_API_KEY found. Using mock search engine...")
            return self._mock_search(query)
            
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        data = {"q": query, "num": 5}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                results = response.json()
                snippets = []
                # Extract organic search snippets
                for item in results.get("organic", []):
                    snippets.append(f"{item.get('title')}: {item.get('snippet')}")
                return snippets
            else:
                return self._mock_search(query)
        except Exception as e:
            print(f"Search API error: {e}. Falling back to mock search.")
            return self._mock_search(query)

    def _mock_search(self, query: str) -> list:
        """
        A rule-based mock search retriever for offline validation.
        """
        query_lower = query.lower()
        if "tax" in query_lower:
            return [
                "Treasury Report: Tax rates remained steady during the June session.",
                "Tax Legislation Fact-sheet: The state tax cuts bill was signed, decreasing rates by 1%.",
                "Reuters News: Bipartisan consensus leads to tax overhaul, avoiding raises."
            ]
        elif "economy" in query_lower:
            return [
                "Bureau of Economic Analysis: GDP increased at an annual rate of 5.2% in the last quarter.",
                "Wall Street Journal: Strong consumer spending fuels 5% economic growth.",
                "Bloomberg: Inflation cools as GDP exceeds expectations with 5% rise."
            ]
        elif "alien" in query_lower or "ufo" in query_lower or "cheese" in query_lower:
            return [
                "NASA Astronomy FAQ: There is no scientific evidence of extraterrestrial life visiting Earth.",
                "Lunar Exploration Group: The moon's crust consists of silicate rock and basalt, not dairy.",
                "Local News: Reports of a UFO sighting over City Hall were debunked as high-altitude drones."
            ]
        else:
            return [
                f"Web Search Archive: Found discussions regarding: '{query}'",
                "FactCheck.org: Investigating viral social media statements.",
                "Associated Press: Official statements and verified records regarding public claims."
            ]

    def verify_claim(self, article: str, claim: str, search_results: list) -> dict:
        """
        Performs Chain-of-Thought (CoT) verification combining the claim and snippets.
        """
        snippets_text = "\n".join([f"- {s}" for s in search_results])
        
        prompt = (
            f"Context: {snippets_text}\n"
            f"Claim to verify: {claim}\n"
            "Task: Verify if the claim is supported by the context. "
            "Write a step-by-step reasoning (Chain of Thought), then end with a final line: "
            "Verdict: [Real / Fake / Unverified]\n\n"
            "Reasoning:"
        )

        if self.llm == "mock":
            # Simple mock evaluation
            if "alien" in claim.lower() or "cheese" in claim.lower() or "bleach" in claim.lower():
                reasoning = "1. The context clearly states UFO reports are drones and the moon is rock, not cheese.\n2. Therefore, the claim contradicts scientific facts."
                verdict = "Fake"
            elif "tax" in claim.lower() or "economy" in claim.lower() or "grew" in claim.lower():
                reasoning = "1. Official government agency and economic reports state the economy grew 5%.\n2. The context supports the claim directly."
                verdict = "Real"
            else:
                reasoning = "1. Available context mentions discussions but lacks direct confirmation.\n2. Verification requires official transcripts."
                verdict = "Unverified"
        else:
            try:
                res = self.llm(prompt, max_length=200, num_return_sequences=1)
                reasoning = res[0]["generated_text"].strip()
                
                # Extract verdict from generated text
                verdict = "Unverified"
                if "verdict: real" in reasoning.lower():
                    verdict = "Real"
                elif "verdict: fake" in reasoning.lower():
                    verdict = "Fake"
                elif "verdict: unverified" in reasoning.lower():
                    verdict = "Unverified"
                else:
                    # Simple heuristic
                    if "real" in reasoning.lower().split()[-10:]:
                        verdict = "Real"
                    elif "fake" in reasoning.lower().split()[-10:]:
                        verdict = "Fake"
            except Exception as e:
                print(f"Error during CoT verification: {e}")
                reasoning = "System error during text inference."
                verdict = "Unverified"
                
        return {
            "claim": claim,
            "search_results": search_results,
            "reasoning": reasoning,
            "verdict": verdict
        }

    def run(self, text: str) -> dict:
        """
        Executes the full fact-checking pipeline.
        """
        claim = self.extract_claims(text)
        snippets = self.search_web(claim)
        result = self.verify_claim(text, claim, snippets)
        return result

if __name__ == "__main__":
    checker = FactChecker()
    # Test fake news claim
    test_article = "Doctors recommend drinking bleach because it cures the common cold in two hours."
    res = checker.run(test_article)
    print("\n=== Fact-Check Result ===")
    print("Claim:", res["claim"])
    print("Verdict:", res["verdict"])
    print("Reasoning:\n", res["reasoning"])
