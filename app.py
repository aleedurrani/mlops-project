from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/model.pkl")

# Load the scaler used during preprocessing
scaler = StandardScaler()
df = pd.read_csv("data/processed_data.csv")
scaler.fit(df[["humidity", "wind_speed"]])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        humidity = data.get("humidity")
        wind_speed = data.get("wind_speed")

        # Validate inputs
        if humidity is None or wind_speed is None:
            return jsonify({"error": "Missing humidity or wind_speed"}), 400

        # Prepare input data
        input_data = pd.DataFrame([[humidity, wind_speed]], columns=["humidity", "wind_speed"])
       
        # Scale the input data
        input_scaled = scaler.transform(input_data)
       
        # Make prediction
        prediction = model.predict(input_scaled)[0]
       
        return jsonify({"predicted_temperature": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)