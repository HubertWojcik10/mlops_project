import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle

def load_and_save_fashion_mnist(save_dir='./data/processed'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False 
    )
    
    # Get full datasets
    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))
    
    # Save datasets
    train_path = os.path.join(save_dir, 'fashion_mnist_train.pkl')
    test_path = os.path.join(save_dir, 'fashion_mnist_test.pkl')
    
    with open(train_path, 'wb') as f:
        pickle.dump({
            'images': train_data[0].numpy(),
            'labels': train_data[1].numpy()
        }, f)
    
    with open(test_path, 'wb') as f:
        pickle.dump({
            'images': test_data[0].numpy(),
            'labels': test_data[1].numpy()
        }, f)
    
    print(f"Dataset saved successfully in {save_dir}")
    print(f"Training set shape: {train_data[0].shape}")
    print(f"Test set shape: {test_data[0].shape}")

# Example usage
if __name__ == "__main__":
    load_and_save_fashion_mnist()

# To load the saved dataset later:
def load_saved_dataset(save_dir='fashion_mnist_data'):
    train_path = os.path.join(save_dir, 'fashion_mnist_train.pkl')
    test_path = os.path.join(save_dir, 'fashion_mnist_test.pkl')
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, test_data