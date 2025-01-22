from src.mlops_project.model import ResNet18ForFMNIST
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

def test_model() -> None:
    """
        Test the model.
    """
    config = {
        "data": {
            "num_classes": 10
        }
    }
    config = OmegaConf.create(config)
    model = ResNet18ForFMNIST(config=config)

    # test 1: assert the right type
    assert isinstance(model, ResNet18ForFMNIST)

    x = torch.randn(1, 1, 28, 28)

    # upsample the input
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    y = model(x)

    # test 2: assert the right shape for a singular input
    assert y.shape == (1, 10)

    batch_size = 32
    x_batch = torch.randn(batch_size, 1, 28, 28)
    x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)
    y_batch = model(x_batch)
    # test 3: assert the right shape given some batch size
    assert y_batch.shape == (batch_size, 10)
