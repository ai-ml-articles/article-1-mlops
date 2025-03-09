import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torchvision.transforms as transforms

# (Optional) import hypertune for reporting metrics in hyperparameter tuning
try:
    import hypertune
except ImportError:
    hypertune = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main():
    # ===== Distributed setup =====
    # Initialize the default process group. Vertex AI will launch one process per GPU per node.
    dist.init_process_group(backend="nccl", init_method="env://")  # use NCCL for GPU, or "gloo" for CPU
    rank = dist.get_rank()              # Global rank of this process among all processes
    world_size = dist.get_world_size()  # Total number of processes (GPUs) across all nodes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Local rank on the current node (set by torchrun)

    # Set the device for this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    torch.cuda.set_device(device)  # If using GPUs, bind the current process to its local GPU

    # ===== Hyperparameters (can be passed in or use defaults) =====
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per process")
    parser.add_argument("--hidden_units", type=int, default=128, help="Hidden units in the model")
    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    hidden_units = args.hidden_units

    if rank == 0:
        logger.info(f"Launching DDP training job with world_size={world_size}, epochs={epochs}, lr={learning_rate}")

    # ===== Data preparation =====
    # Example: create a synthetic dataset (10000 samples, 10 features, binary labels) 
    # In practice, load your real dataset here (and ensure all replicas can access it, e.g., from GCS)
    num_samples = 10000
    num_features = 10
    num_classes = 2
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)

    # Use DistributedSampler to split the dataset among DDP processes
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # ===== Model, Loss, Optimizer setup =====
    # Simple model example: a two-layer neural network
    model = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_classes)
    ).to(device)
    # Wrap the model with DistributedDataParallel for synchronized training
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)

    # ===== Training Loop =====
    ddp_model.train()
    for epoch in range(1, epochs + 1):
        # Set epoch for sampler for shuffling
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)

        # Only the main process (rank 0) logs the metrics to avoid duplicate logs
        if rank == 0:
            logger.info(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")

    # ===== Post-training: evaluation and saving model =====
    if rank == 0:
        # (Optional) Evaluate training accuracy on the full dataset (using the model on CPU or single GPU)
        ddp_model.eval()
        correct = 0
        total = 0
        # Create a DataLoader without DistributedSampler to go over full dataset on rank 0
        eval_loader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for features, labels in eval_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = ddp_model(features)  # DDP model on rank 0 still wraps the underlying model
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Training completed. Accuracy on full dataset: {accuracy:.4f}")

        # Save the model (only on rank 0 to avoid multiple writes)
        model_filename = "model.pth"
        # Use model.module if using DDP to get the underlying model weights
        torch.save(ddp_model.module.state_dict(), model_filename)
        logger.info(f"Model saved to {model_filename}")

        # If running on Vertex AI, save model to the output directory (AIP_MODEL_DIR) for persistent storage
        model_dir = os.environ.get("AIP_MODEL_DIR")
        if model_dir:
            # Upload the model file to the Cloud Storage path in AIP_MODEL_DIR
            model_path = os.path.join(model_dir, model_filename)
            try:
                from google.cloud import storage
                storage_client = storage.Client()
                blob = storage.Blob.from_string(model_path, client=storage_client)
                blob.upload_from_filename(model_filename)
                logger.info(f"Model uploaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to upload model to {model_dir}: {e}")

        # Report metric for hyperparameter tuning if hypertune is available
        if hypertune:
            hpt = hypertune.HyperTune()
            # Report the final accuracy (Vertex hyperparameter tuner will use this)
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='accuracy',
                metric_value=accuracy
            )

def run():
    """Entry point for torch.multiprocessing (if used)."""
    main()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log any exception that caused training to fail
        logger.exception("Training failed due to an error: %s", str(e))
        # Propagate the exception to ensure the job is marked as failed
        raise
