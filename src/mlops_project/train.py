from data import preprocess
from model import Model
from pathlib import Path
import typer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10
):
    """
    Train the model on Fashion MNIST.
    """
    train_dataset, _ = preprocess()
    model = Model().to(DEVICE)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for i, (img, label) in enumerate(train_dataloader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

if __name__ == "__main__":
    typer.run(train)