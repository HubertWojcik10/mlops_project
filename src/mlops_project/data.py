import os
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


RAW_DIR = Path("../../data/raw")
PROCESSED_DIR = Path("../../data/processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_and_preprocess():
    print("Downloading Fashion MNIST dataset...")

    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

    # Download the training and test datasets
    train_dataset = datasets.FashionMNIST(
        root=RAW_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=RAW_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # Save raw training and test data
    save_data(test_dataset, "test")
    save_data(train_dataset, "train")

    print(f"Data processed and saved to {PROCESSED_DIR}")


def save_data(dataset, data_type):
    data_path = PROCESSED_DIR / f"{data_type}_data.pkl"

    images = dataset.data.numpy()
    labels = dataset.targets.numpy()

    with open(data_path, "wb") as f:
        pickle.dump({
            "images": images,
            "labels": labels
        }, f)

    print(f"Saved {data_type} data to {data_path}")

class FashionMNISTDataset(Dataset):
    def __init__(self, data_path: str):
        """
        Custom dataset class for loading Fashion MNIST data from a Pickle file.
        
        Args:
            data_path (str): Path to the Pickle file containing the data.
        """
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.images = torch.tensor(data["images"], dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return a single sample (image, label) pair."""
        return self.images[idx], self.labels[idx]


def load_processed_data(data_type: str):
    data_path = PROCESSED_DIR / f"{data_type}_data.pkl"
    
    # Load the data
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    return data

def get_test_loader():
    test_data = FashionMNISTDataset(PROCESSED_DIR / "test_data.pkl")
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return test_loader

def get_train_loaders():
    
    train_data = FashionMNISTDataset(PROCESSED_DIR / "train_data.pkl")
    train_size = int(0.8 * len(train_data))  # 80% for training
    val_size = len(train_data) - train_size  # 20% for validation
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

   
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    download_and_preprocess()

     