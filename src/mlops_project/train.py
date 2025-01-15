from data import get_train_loaders, download_and_preprocess
from model import get_model
from pathlib import Path
import typer
import torch
import hydra
from omegaconf import DictConfig 
import os 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config: DictConfig):
    """
    Train the model on Fashion MNIST.
    """
    model = get_model(config)
    model.to(DEVICE)

    train_loader, val_loader = get_train_loaders(config.paths.processed_dir, config.data.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    current_val_loss = float('inf')
    for epoch in range(config.experiment.epochs):
        model.train()
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                y_pred = model(img)
                val_loss += loss_fn(y_pred, label).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch}, val_loss: {val_loss}")

    # Save the model
    if current_val_loss > val_loss:
        model_path = Path(f"{config.experiment.save_dir}/{config.model.save_model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        current_val_loss = val_loss

@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def hydra_train(config: DictConfig):

    download_and_preprocess(config)
    train(config)

if __name__ == "__main__":
    typer.run(hydra_train())