import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load the model
model = joblib.load("/app/model.pkl")

# Start Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        instances = np.array(data["instances"])
        predictions = model.predict(instances).tolist()
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
