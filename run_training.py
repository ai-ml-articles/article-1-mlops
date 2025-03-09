from google.cloud import aiplatform

# Initialize the AI Platform with your project and location
aiplatform.init(project="mlops-housing-project", location="us-central1")

# Define the worker pool specifications for distributed training
worker_pool_specs = [
    {  # Master pool (replica_count must be 1)
        "machine_spec": {
            "machine_type": "n1-highmem-16",
            "accelerator_type": "NVIDIA_TESLA_T4",  # Specify GPU type
            "accelerator_count": 1,  # Number of GPUs per instance
        },
        "replica_count": 1,  # Master must have exactly 1 replica
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310",
            "command": ["python", "train_distributed.py"],
        },
    },
    {  # Worker pool (additional replicas for distributed training)
        "machine_spec": {
            "machine_type": "n1-highmem-16",
            "accelerator_type": "NVIDIA_TESLA_T4",  # Specify GPU type
            "accelerator_count": 1,  # Number of GPUs per instance
        },
        "replica_count": 3,  # Additional workers (total 4 with master)
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310",
            "command": ["python", "train_distributed.py"],
        },
    }
]
# Create the custom job
job = aiplatform.CustomJob(
    display_name="housing-distributed-training",
    worker_pool_specs=worker_pool_specs,
    staging_bucket="gs://mlops-housing-bucket-us-central1"  # Updated to regional bucket
)
# Run the job
job.run()
