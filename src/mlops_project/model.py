import torch 
from torch import nn

class Model(nn.Module):
    """ 
        A neural network model for the FMNIST task. 
    """
    def __init__(self) -> None:
        super().__init__()
        classes_num = 10 # TODO: put in the config
        dropout_rate = 0.5  # TODO: put in the config

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, classes_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass.
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return self.fc1(x)