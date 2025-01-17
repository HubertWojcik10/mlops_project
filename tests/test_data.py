from torch.utils.data import Dataset

from src.mlops_project.data import FashionMNISTDataset


train_dataset = FashionMNISTDataset("data/processed/train_data.pkl")
test_dataset = FashionMNISTDataset("data/processed/test_data.pkl")

def test_my_dataset() -> None:
    """
        Test the FashionMNISTDataset class.
    """
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

def test_dataset_len() -> None:
    """
        Test the length of the dataset.
    """
    assert len(train_dataset) == 60000
    assert len(test_dataset) == 10000

def test_dataset_shape() -> None:
    """
        Test the shape of the dataset's input.
    """
    for dataset in [train_dataset, test_dataset]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)