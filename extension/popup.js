// popup.js
/**
 * Chrome Extension Toolbar Popup Logic
 * Calls the local FastAPI server and renders the result in popup.html.
 */

document.getElementById("verify-btn").addEventListener("click", () => {
  const text = document.getElementById("text-input").value.trim();
  const resultDiv = document.getElementById("result");
  const origVerdictDiv = document.getElementById("orig-verdict");
  const fcVerdictDiv = document.getElementById("fc-verdict");
  const fcClaimDiv = document.getElementById("fc-claim");
  const finalVerdictDiv = document.getElementById("final-verdict");
  const overrideReasonDiv = document.getElementById("override-reason");
  
  if (text.length < 10) {
    resultDiv.style.display = "block";
    finalVerdictDiv.className = "verdict-box error";
    finalVerdictDiv.innerText = "Error: Input too short";
    return;
  }
  
  // Reset UI
  finalVerdictDiv.className = "verdict-box";
  finalVerdictDiv.innerText = "Analyzing...";
  origVerdictDiv.innerText = "";
  fcVerdictDiv.innerText = "";
  fcClaimDiv.innerText = "";
  overrideReasonDiv.innerText = "";
  resultDiv.style.display = "block";
  
  fetch("http://localhost:8000/factcheck", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text, threshold: 0.8 })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`Server status ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    const orig = data.original_prediction;
    const fc = data.fact_check;
    const finalLabel = data.final_prediction;

    // 1. AI Classifier
    origVerdictDiv.className = `verdict-box ${orig.label === "Real" ? "real" : "fake"}`;
    origVerdictDiv.innerText = `${orig.label} (${(orig.confidence * 100).toFixed(1)}%)`;

    // 2. Fact Check
    let fcClass = "info";
    if (fc.verdict === "Supported") fcClass = "real";
    if (fc.verdict === "Contradicted") fcClass = "fake";
    
    fcVerdictDiv.className = `verdict-box ${fcClass}`;
    fcVerdictDiv.innerText = `${fc.verdict} (${(fc.confidence * 100).toFixed(1)}%)`;
    fcClaimDiv.innerText = `Claim: ${fc.extracted_claim}`;

    // 3. Final
    finalVerdictDiv.className = `verdict-box ${finalLabel === "Real" ? "real" : "fake"}`;
    finalVerdictDiv.innerText = finalLabel;
    
    if (data.override_reason) {
      overrideReasonDiv.innerText = data.override_reason;
    } else if (data.threshold_note) {
      overrideReasonDiv.innerText = data.threshold_note;
      overrideReasonDiv.style.color = "#94a3b8";
    }
  })
  .catch(error => {
    finalVerdictDiv.className = "verdict-box error";
    finalVerdictDiv.innerText = "Connection Failed";
    fcClaimDiv.innerText = "FastAPI server may be offline. Start the server on port 8000.";
  });
});
