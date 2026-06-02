import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googlesearch import search
import requests
from bs4 import BeautifulSoup

class RAGFactChecker:
    def __init__(self, openai_api_key=None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0, 
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )

    def extract_claims(self, text):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract the main factual claims from the following news text. Return them as a numbered list. Text: {text}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(text)

    def search_evidence(self, claims):
        # Just search the first few claims for brevity
        top_claims = claims.split('\n')[:2]
        evidence = []
        for claim in top_claims:
            if not claim.strip(): continue
            print(f"Searching evidence for: {claim}")
            search_results = search(claim, num_results=3)
            for url in search_results:
                try:
                    res = requests.get(url, timeout=5)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    text = ' '.join([p.text for p in soup.find_all('p')[:3]]) # Get first 3 paragraphs
                    evidence.append({"url": url, "text": text})
                except:
                    continue
        return evidence

    def verify_fact(self, text, evidence):
        evidence_str = "\n".join([f"Source {i}: {e['text']}" for i, e in enumerate(evidence)])
        prompt = PromptTemplate(
            input_variables=["text", "evidence"],
            template="""
            News Text: {text}
            
            Evidence retrieved from web search:
            {evidence}
            
            Based on the evidence above, evaluate the truthfulness of the News Text. 
            Provide a verdict (Real, Fake, or Unverified) and a detailed reasoning.
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(text=text, evidence=evidence_str)

    def run(self, text):
        claims = self.extract_claims(text)
        evidence = self.search_evidence(claims)
        verdict = self.verify_fact(text, evidence)
        return {"claims": claims, "evidence": evidence, "verdict": verdict}
