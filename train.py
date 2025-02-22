import argparse
import joblib
import os
from google.cloud import storage
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_and_upload_model(bucket_name, model_filename):
    # Load dataset
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure the directory exists before saving
    local_model_path = os.path.join("model", model_filename)
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # Save the model locally
    joblib.dump(model, local_model_path)
    print(f"✅ Model trained and saved locally at: {local_model_path}")

    # Upload model to Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(local_model_path)

    print(f"🚀 Model uploaded to: gs://{bucket_name}/models/{model_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", type=str, required=True, help="Google Cloud Storage bucket name")
    parser.add_argument("--model-filename", type=str, default="model.pkl", help="Filename for saved model")
    
    args = parser.parse_args()

    train_and_upload_model(args.bucket_name, args.model_filename)
