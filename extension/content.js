// Create a floating button when text is selected
document.addEventListener('mouseup', function(e) {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText.length > 20) {
        let btn = document.getElementById('fndm-check-btn');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'fndm-check-btn';
            btn.innerText = '🔍 Check News';
            btn.style.position = 'absolute';
            btn.style.zIndex = '9999';
            btn.style.backgroundColor = '#4CAF50';
            btn.style.color = 'white';
            btn.style.border = 'none';
            btn.style.borderRadius = '5px';
            btn.style.padding = '5px 10px';
            btn.style.cursor = 'pointer';
            document.body.appendChild(btn);
        }
        btn.style.display = 'block';
        btn.style.top = (e.pageY + 10) + 'px';
        btn.style.left = (e.pageX + 10) + 'px';
        
        btn.onclick = function() {
            checkNews(selectedText);
        };
    } else {
        const btn = document.getElementById('fndm-check-btn');
        if (btn) btn.style.display = 'none';
    }
});

function checkNews(text) {
    // Replace with your actual deployed API URL
    const API_URL = 'http://localhost:8000/predict';
    
    fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        alert(`Prediction: ${data.label}\nConfidence: ${(data.confidence * 100).toFixed(2)}%`);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Could not connect to Fake News Detection API.');
    });
}
