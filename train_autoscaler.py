from google.cloud import aiplatform

# Define project and model details
PROJECT_ID = "mlops-housing-project"  # Replace with your project ID
REGION = "us-central1"
MODEL_ID = "6060557570324561920"  # Replace with actual model ID
ENDPOINT_DISPLAY_NAME = "housing-scale-endpoint"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

# Retrieve the model
model = aiplatform.Model(model_name=f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}")

# Deploy the model to an endpoint
endpoint = model.deploy(
    deployed_model_display_name=ENDPOINT_DISPLAY_NAME,
    machine_type="n1-standard-4",
    min_replica_count=2,  # Minimum number of instances
    max_replica_count=10,  # Maximum number of instances
    traffic_split={"0": 100},  # Direct all traffic to the deployed model
)

# Print endpoint details
print(f"✅ Model deployed successfully at endpoint: {endpoint.resource_name}")

# Now apply the auto-scaling policy separately
endpoint.update_deployed_model(
    deployed_model_id=endpoint.deployed_models[0].id,
    autoscaling_metric_specs=[
        {"metricName": "aiplatform.googleapis.com/prediction/requests", "target": 10},  # Scale based on request count
        {"metricName": "aiplatform.googleapis.com/prediction/cpu/utilization", "target": 0.6},  # Scale based on CPU usage
    ],
)

print(f"✅ Auto-scaling policy applied to endpoint {ENDPOINT_DISPLAY_NAME}")
