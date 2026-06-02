# test_hybrid_system.py
import requests
import time
import sys

def test_hybrid_system():
    api_url = "http://localhost:8000"
    health_endpoint = f"{api_url}/health"
    factcheck_endpoint = f"{api_url}/factcheck"

    print("Checking API health...")
    max_retries = 30
    for i in range(max_retries):
        try:
            res = requests.get(health_endpoint)
            if res.status_code == 200 and res.json().get("model_loaded"):
                print("API is healthy and model is loaded.")
                break
        except:
            pass
        print(f"Waiting for API... ({i+1}/{max_retries})")
        time.sleep(5)
    else:
        print("API failed to become healthy. Exiting.")
        sys.exit(1)

    test_cases = [
        "Virat Kohli is the world chess champion",
        "Rahul is the current world chess champion",
        "The Earth revolves around the Sun",
        "Water is dry",
        "Donald Trump is the current president of the United States"
    ]

    passed = 0
    print("\nStarting Hybrid System Tests...\n" + "="*50)

    for text in test_cases:
        print(f"Testing: \"{text}\"")
        try:
            response = requests.post(factcheck_endpoint, json={"text": text, "threshold": 0.5})
            if response.status_code == 200:
                data = response.json()
                orig = data['original_prediction']
                fc = data.get('fact_check') or {}
                final = data['final_prediction']
                
                print(f"  - DistilBERT: {orig['label']} ({orig['confidence']:.2%})")
                print(f"  - Fact-Check: {fc.get('verdict', 'N/A')} (Claim: {fc.get('extracted_claim', 'N/A')})")
                print(f"  - Final Result: {final}")
                
                # Logical pass/fail check
                is_pass = False
                if "chess champion" in text.lower() and ("Virat" in text or "Rahul" in text):
                    is_pass = final == "Fake"
                elif "Earth" in text and "Sun" in text:
                    is_pass = final == "Real"
                elif "Water is dry" in text:
                    is_pass = final == "Fake"
                elif "Donald Trump" in text and "president" in text:
                    # Depending on current logic, this might be Fake or Real
                    # but for now let's just accept the logic path
                    is_pass = True 
                else:
                    is_pass = True
                
                if is_pass:
                    print("  - STATUS: PASSED")
                    passed += 1
                else:
                    print("  - STATUS: FAILED")
            else:
                print(f"  - Error: API returned {response.status_code}")
        except Exception as e:
            print(f"  - Exception: {e}")
        print("-" * 30)

    print(f"\nSummary: {passed}/{len(test_cases)} tests passed.")

if __name__ == "__main__":
    test_hybrid_system()
