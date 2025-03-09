from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='mlops-housing-project', location='us-central1')

# Upload the model with custom container
model = aiplatform.Model.upload(
    display_name='california-housing-model',
    artifact_uri='gs://mlops-housing/models/',
    serving_container_image_uri='us-central1-docker.pkg.dev/mlops-housing-project/housing-repo/housing-model:latest',
    serving_container_predict_route='/predict',
    serving_container_health_route='/health',
)

# Create an endpoint
endpoint = aiplatform.Endpoint.create(display_name='california-housing-endpoint')

# Deploy the model to the endpoint
model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=1,
)

print(f"Endpoint created with ID: {endpoint.resource_name}")
