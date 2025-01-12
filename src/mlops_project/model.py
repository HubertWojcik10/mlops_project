import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet18ForFMNIST(nn.Module):
    """
    ResNet18 model adapted for Fashion MNIST.
    """
    def __init__(self, num_classes=10):
        super(ResNet18ForFMNIST, self).__init__()
        
        # Load pre-trained ResNet18 with weights from ImageNet
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first convolution layer to accept 1 channel (resnet is trained on RGB images)
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Modify the fully connected layer to match the number of classes in Fashion MNIST
        self.resnet18.fc = nn.Linear(
            in_features=self.resnet18.fc.in_features, 
            out_features=num_classes
        )

    def forward(self, x):
        return self.resnet18(x)

def get_model():
    model = ResNet18ForFMNIST()
    return model.to(device)
