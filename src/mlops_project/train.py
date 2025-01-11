from data import preprocess
from pathlib import Path
import typer
import torch

def train(
    raw_data_path: Path = Path("data/raw"), # TODO: put in config
    output_folder: Path = Path("data/processed"),
    lr: float = 1e-3, 
    batch_size: int = 32, 
    epochs: int = 10
):
    """
    Train the model on Fashion MNIST.
    """
    train_dataset, _ = preprocess(raw_data_path, output_folder)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    typer.run(train)