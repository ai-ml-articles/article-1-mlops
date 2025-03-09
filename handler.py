import os
import tarfile
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)
model = None

def load_model(artifact_uri):
    global model
    try:
        tar_path = os.path.join(artifact_uri, "model.tar.gz").replace("gs://", "")
        bucket_name, blob_name = tar_path.split("/", 1)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename("model.tar.gz")
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall()
        model = joblib.load("model.pkl")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return "Model not loaded", 500
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json()
    instances = data.get("instances", [])
    predictions = model.predict(instances).tolist()
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    artifact_uri = os.getenv("AIP_STORAGE_URI", "gs://mlops-housing/models/")
    load_model(artifact_uri)
    app.run(host="0.0.0.0", port=8080)
