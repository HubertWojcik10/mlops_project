import os
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import hydra
from omegaconf import DictConfig


def download_and_preprocess(config: DictConfig):
    print("Downloading Fashion MNIST dataset...")

    raw_dir = Path(config.paths.raw_dir)
    processed_dir = Path(config.paths.processed_dir)

    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Define dataset transformation
    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Download the training and test datasets
    train_dataset = datasets.FashionMNIST(
        root=raw_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=raw_dir, train=False, download=True, transform=transform
    )

    # Save raw training and test data
    save_data(test_dataset, "test", processed_dir)
    save_data(train_dataset, "train", processed_dir)

    print(f"Data processed and saved to {processed_dir}")


def save_data(dataset, data_type, processed_dir):
    """Save dataset to Pickle file"""
    data_path = processed_dir / f"{data_type}_data.pkl"

    images = dataset.data.numpy()
    labels = dataset.targets.numpy()

    with open(data_path, "wb") as f:
        pickle.dump({"images": images, "labels": labels}, f)

    print(f"Saved {data_type} data to {data_path}")


class FashionMNISTDataset(Dataset):
    """Custom dataset class for loading Fashion MNIST data from a Pickle file"""

    def __init__(self, data_path: str):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.images = torch.tensor(data["images"], dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_test_loader(processed_dir: str, batch_size: int) -> DataLoader:
    """Get the test DataLoader"""
    processed_dir = Path(processed_dir)
    test_data = FashionMNISTDataset(processed_dir / "test_data.pkl")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


def get_train_loaders(processed_dir: str, batch_size: int) -> DataLoader:
    """Get train and validation DataLoaders"""
    processed_dir = Path(processed_dir)
    train_data = FashionMNISTDataset(processed_dir / "train_data.pkl")
    train_size = int(0.8 * len(train_data))  # 80% for training
    val_size = len(train_data) - train_size  # 20% for validation
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
