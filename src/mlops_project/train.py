from data import preprocess
from pathlib import Path
import typer

def train(
    raw_data_path: Path = Path("data/raw"), # TODO: put in config
    output_folder: Path = Path("data/processed")
):
    """
    Train the model on Fashion MNIST.
    """
    train_set, _ = preprocess(raw_data_path, output_folder)

if __name__ == "__main__":
    typer.run(train)