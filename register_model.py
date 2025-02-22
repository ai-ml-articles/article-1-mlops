from google.cloud import aiplatform

# Replace these values with your actual project and bucket name
PROJECT_ID = "mlops-housing-project"
REGION = "us-central1"
BUCKET_NAME = "mlops-housing-bucket"
MODEL_NAME = "housing-price-model-v1"

# Initialize Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION)

# Upload model to Vertex AI Model Registry
model = aiplatform.Model.upload(
    display_name=MODEL_NAME,
    artifact_uri=f"gs://{BUCKET_NAME}/models/",  # Path to model artifacts in GCS
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    labels={"version": "v1-0"}
)

print(f"✅ Model registered successfully! Model ID: {model.resource_name}")
