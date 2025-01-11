"""
    Module to handle the dataset operations.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import struct
import os
from PIL import Image
import numpy as np
import torch
import typer
from torch.utils.data import Dataset

class FashionMNISTDataset(Dataset):
    """
        Fashion MNIST dataset processor.
    """

    def __init__(self, raw_data_path: Path, dataset_type: str, transform: Optional[callable] = None) -> None:
        """
            Initialize the dataset.
        """
        self.data_path = raw_data_path
        self.transform = transform
        self.dataset_type = dataset_type
        self.images, self.labels = [], []
        
        # define file paths based on dataset type
        if dataset_type == 'train':
            images_path = Path(os.path.join(self.data_path, "train-images-idx3-ubyte"))
            labels_path = Path(os.path.join(self.data_path, "train-labels-idx1-ubyte"))
        elif dataset_type == 'test':
            images_path = Path(os.path.join(self.data_path, "t10k-images-idx3-ubyte"))
            labels_path = Path(os.path.join(self.data_path, "t10k-labels-idx1-ubyte"))
        else:
            raise ValueError("dataset_type must be either 'train' or 'test'")
        
        if images_path.exists() and labels_path.exists():
            self.images, self.labels = self._read_mnist_files(images_path, labels_path)
        else:
            raise FileNotFoundError(f"Could not find {dataset_type} dataset files")

    def _read_mnist_files(self, images_path: Path, labels_path: Path) -> Tuple[np.ndarray, List]:
        """
            Read the binary MNIST format files.
        """
        with open(images_path, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(-1, rows, cols)
            
        with open(labels_path, 'rb') as f:
            _, _ = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8).tolist()
            
        return images, labels

    def __len__(self) -> int:
        """
            Return the length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray | torch.Tensor, int]:
        """
            Return a given sample from the dataset.
        """
        image, label = self.images[index], self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """
            Preprocess the raw data and save it to PNG images.
        """
        # create dataset-specific output folder
        dataset_folder = Path(os.path.join(output_folder, self.dataset_type))
        dataset_folder.mkdir(parents=True, exist_ok=True)

        # TODO: PUT THAT IN THE CONFIG FILE
        class_names = ['tshirt_top', 'trouser', 'pullover', 'dress', 'coat', 
                      'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

        # create class-specific folders
        for class_name in class_names:
            Path(os.path.join(dataset_folder, class_name)).mkdir(exist_ok=True)

        # save each image in its class folder
        for idx, (image, label) in enumerate(zip(self.images, self.labels)):
            img = Image.fromarray(image)
            class_name = class_names[label]
            img.save(dataset_folder / class_name / f"{idx}.png")


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """
        Main function to preprocess both training and test datasets.
    """
    print("Preprocessing Fashion MNIST data...")

    print("Processing training dataset...")
    train_dataset = FashionMNISTDataset(raw_data_path, dataset_type='train')
    train_dataset.preprocess(output_folder)

    print("Processing test dataset...")
    test_dataset = FashionMNISTDataset(raw_data_path, dataset_type='test')
    test_dataset.preprocess(output_folder)

    print("Preprocessing complete!")


if __name__ == "__main__":
    typer.run(preprocess)