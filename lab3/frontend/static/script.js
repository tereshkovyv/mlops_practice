document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const resultsDiv = document.getElementById('results');
    
    if (!fileInput.files[0]) {
        resultsDiv.innerHTML = '<p>Please select a file.</p>';
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    fetch('http://localhost:5000/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            let html = '<h2>Predictions:</h2><ul>';
            data.predictions.forEach(pred => {
                html += `<li>${pred.class}: ${(pred.probability * 100).toFixed(2)}%</li>`;
            });
            html += '</ul>';
            resultsDiv.innerHTML = html;
        }
    })
    .catch(error => {
        resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    });
});