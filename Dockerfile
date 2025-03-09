# Use Vertex AI PyTorch GPU base image (PyTorch 1.13, Python 3)
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest

# Install any additional Python dependencies
# e.g., cloudml-hypertune for reporting metrics in hyperparameter tuning
RUN pip install --no-cache-dir cloudml-hypertune

# Set working directory
WORKDIR /app

# Copy the training script into the container
COPY train_distributed.py /app/train_distributed.py

# (Optional) Copy any other required files, e.g., data or config files
# COPY data/ /app/data/

# Set entrypoint to run the training script
ENTRYPOINT ["python", "train_distributed.py"]
