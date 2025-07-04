async function runNeuralNetwork() {
    const inputFields = document.querySelectorAll('.inputFields input');
    const inputObj = {};

    inputFields.forEach(input => {
        const val = input.value.trim();
        inputObj[input.name] = isNaN(val) ? val : parseFloat(val);
    });

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: inputObj })
        });

        if (!response.ok) throw new Error("Prediction request failed.");

        const result = await response.json();
        document.getElementById("prediction").innerText = `Predicted Cost: $${result.prediction[0].toFixed(2)}`;
    } catch (err) {
        alert("Error: " + err.message);
    }
}
