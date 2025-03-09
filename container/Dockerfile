# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas scikit-learn joblib flask gunicorn google-cloud-storage

# Copy the model into the container
COPY model.pkl /app/model.pkl

# Copy the inference script
COPY predictor.py /app/predictor.py

# Expose the port for the Flask app
EXPOSE 8080

# Define the entry point for serving predictions
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "predictor:app"]
