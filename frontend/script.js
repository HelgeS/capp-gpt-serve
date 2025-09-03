document.addEventListener('DOMContentLoaded', () => {
    const apiBaseUrl = 'http://localhost:8000';
    const formFields = document.getElementById('form-fields');
    const predictionForm = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');

    // Update temperature display
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });

    // Fetch tokens and build form
    fetch(`${apiBaseUrl}/tokens`)
        .then(response => response.json())
        .then(data => {
            Object.keys(data).forEach(key => {
                if (key !== 'process_chains') {
                    const label = document.createElement('label');
                    label.setAttribute('for', key);
                    label.textContent = `${key.replace('_', ' ')}:`;

                    const select = document.createElement('select');
                    select.id = key;
                    select.name = key;

                    data[key].forEach(optionValue => {
                        const option = document.createElement('option');
                        option.value = optionValue;
                        option.textContent = optionValue.replace(/_/g, ' ');
                        select.appendChild(option);
                    });

                    const formGroup = document.createElement('div');
                    formGroup.className = 'form-group';
                    formGroup.appendChild(label);
                    formGroup.appendChild(select);
                    formFields.appendChild(formGroup);
                }
            });
        })
        .catch(error => {
            console.error('Error fetching tokens:', error);
            resultsContainer.innerHTML = '<p class="error">Could not load form fields. Is the server running?</p>';
        });

    // Handle form submission
    predictionForm.addEventListener('submit', event => {
        event.preventDefault();
        resultsContainer.innerHTML = '<p>Loading...</p>';

        const formData = new FormData(predictionForm);
        const partCharacteristics = {};
        for (const [key, value] of formData.entries()) {
            if (key !== 'max_processes' && key !== 'temperature') {
                partCharacteristics[key] = value;
            }
        }

        const requestBody = {
            part_characteristics: partCharacteristics,
            max_processes: parseInt(formData.get('max_processes')),
            temperature: parseFloat(formData.get('temperature')),
            include_confidence: true
        };

        fetch(`${apiBaseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            let html = '<ul>';
            data.process_chains.forEach((process, index) => {
                const confidence = data.confidence_scores ? `(${(data.confidence_scores[index] * 100).toFixed(2)}%)` : '';
                html += `<li>${process} ${confidence}</li>`;
            });
            html += '</ul>';
            resultsContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error during prediction:', error);
            resultsContainer.innerHTML = `<p class="error">Prediction failed. Check the console for details.</p>`;
        });
    });
});
