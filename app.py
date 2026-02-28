# app.py
from flask import Flask, request, render_template_string
import joblib
import numpy as np

# Load saved model and preprocessing tools
model = joblib.load("crop_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🌾 Crop Recommendation System</title>
    <style>
        body { font-family: Arial; background: #e8f5e9; padding: 40px; }
        h1 { color: #2e7d32; }
        form { background: white; padding: 25px; border-radius: 10px; width: 350px; }
        input { margin: 8px 0; padding: 8px; width: 100%; border-radius: 5px; border: 1px solid #ccc; }
        input[type=submit] { background: #2e7d32; color: white; border: none; cursor: pointer; }
        input[type=submit]:hover { background: #388e3c; }
        h2 { color: #1b5e20; }
    </style>
</head>
<body>
    <h1>🌱 Crop Recommendation System</h1>
    <form method="POST">
        <input type="number" step="any" name="N" placeholder="Nitrogen (N)" required>
        <input type="number" step="any" name="P" placeholder="Phosphorous (P)" required>
        <input type="number" step="any" name="K" placeholder="Potassium (K)" required>
        <input type="number" step="any" name="temperature" placeholder="Temperature (°C)" required>
        <input type="number" step="any" name="humidity" placeholder="Humidity (%)" required>
        <input type="number" step="any" name="ph" placeholder="Soil pH" required>
        <input type="number" step="any" name="rainfall" placeholder="Rainfall (mm)" required>
        <input type="submit" value="Predict Crop">
    </form>
    {% if prediction %}
    <h2>Recommended Crop: 🌾 {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Collect inputs
            vals = [float(request.form[v]) for v in ["N","P","K","temperature","humidity","ph","rainfall"]]
            vals_scaled = scaler.transform([vals])
            pred = model.predict(vals_scaled)
            prediction = le.inverse_transform(pred)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

