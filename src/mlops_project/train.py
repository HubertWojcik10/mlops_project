import torch
import wandb
from pathlib import Path
from data import get_train_loaders, download_and_preprocess
from model import get_model
from loguru import logger
import sys
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from google.cloud import storage
import os

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, loss_fn, optimizer, epoch):
    """
    Train the model on Fashion MNIST.
    """
    model.train()
    train_loss = 0
    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            wandb.log({'iter': i, 'loss': loss.item()})
            logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
            logger.debug(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for img, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            y_pred = model(img)
            val_loss += loss_fn(y_pred, label).item()
    val_loss /= len(val_loader)
    logger.info(f"Epoch {epoch}, val_loss: {val_loss}")
    train_loss /= len(train_loader)
    return train_loss, val_loss

def compare_models_val_loss(current_val, new_val):
    return current_val > new_val

def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def train_sweep(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)

    with wandb.init(project="sweep_project", config=config_dict) as run:
        wb_config = wandb.config
        run_name = f"sweep_lr_{wb_config.lr}_batch_{wb_config.batch_size}"
        run.name = run_name
        logger.info(f"Run name: {run_name}")
        logger.info("Training started")

        # Model initialization
        model = get_model(config)
        model.to(DEVICE)
        logger.info("Model added to device")

        # Data loading
        train_loader, val_loader = get_train_loaders(wb_config.paths['processed_dir'], wb_config.batch_size)
        logger.info("Data loaders initialized")

        # Loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wb_config.lr)
        logger.info("Loss function and optimizer initialized")

        # Track the best model
        current_val_loss = float('inf')

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
        for epoch in range(wb_config.experiment['epochs']):
            wandb.log({'epoch': epoch})
            train_loss, val_loss = train(model, train_loader, val_loader, loss_fn, optimizer, epoch)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

            # Save the model if validation loss improves
            if compare_models_val_loss(current_val_loss, val_loss):
                model_path = Path(f"{wb_config.paths['save_dir']}/model.pth")
                save_model(model, model_path)
                current_val_loss = val_loss

    if wb_config.cloud['push_to_cloud']:
        bucket_name = "fashion_mnist_data_bucket"
        model_filename = "model.pth"
        gcs_path = f"models/{model_filename}"  

        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()

        # Reference the GCS bucket
        bucket = storage_client.bucket(bucket_name)

        # Create a blob (file object) in GCS
        blob = bucket.blob(gcs_path)

        # Upload the local model file to GCS
        blob.upload_from_filename(model_path)

        print(f"Model uploaded to gs://{bucket_name}/{gcs_path}")



@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def main(config: DictConfig):
    with open(config.paths.sweep_path, "r") as file:
        sweep_config = yaml.safe_load(file)

    print(OmegaConf.to_yaml(config))
    print(sweep_config)
    download_and_preprocess(config)

    sweep_id = wandb.sweep(sweep_config, project='sweep_project')
    wandb.agent(sweep_id, function=lambda: train_sweep(config), count=6)

if __name__ == "__main__":
    main()
