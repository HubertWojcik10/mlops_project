from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
import torch
import io
from PIL import Image
import numpy as np
from http import HTTPStatus
from src.mlops_project.data import get_train_loaders, download_and_preprocess, get_test_loader
from src.mlops_project.model import ResNet18ForFMNIST
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import RandomSampler
from torchvision import transforms
from torch import nn
from typing import Dict
import os

CONFIG_PATH = "./configs/config.yaml"
tags_metadata = [
    {
        "name": "Sample",
        "description": "Sample an image from the dataset (train/test/val).",
    },
    {
        "name": "Predict Sample",
        "description": "Investigate predictions for given ids from the test set. Id ranges from 0 to len(dataset).",
    },
    {
        "name": "Predict File",
        "description": "Check the model peformance by uploading your own files.",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

def load_model(config: DictConfig, path: str) -> nn.Module:
    """
        Load the resnet-18 model.
    """
    model = ResNet18ForFMNIST(config=config)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        print("Running on an empty model.")

    model.eval()
    return model

LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


@app.get("/")
def root() -> Dict[str, str]:
    """ 
        Homepage & Health check.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/sample/image", tags=["Sample"])
async def sample_image(dataset: str = "train", size: int = 280) -> Response:
    """
        Return a sample image from the dataset as PNG.
    """
    config = OmegaConf.load(CONFIG_PATH)
    download_and_preprocess(config)

    # get the appropriate dataset
    if dataset == "test":
        loader = get_test_loader(config.paths.processed_dir, config.data.api_batch_size)
    elif dataset == "train":
        loader, _ = get_train_loaders(config.paths.processed_dir, config.data.api_batch_size)
    elif dataset == "val":
        _, loader = get_train_loaders(config.paths.processed_dir, config.data.api_batch_size)
    else:
        raise ValueError("The dataset parameter should be [train, test, val]")

    # get a random sample
    sampler = RandomSampler(loader.dataset)
    random_idx = next(iter(sampler))

    image, label = loader.dataset[random_idx]

    # convert tensor to PIL Image
    image_np = image.squeeze().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)

    # resize using nearest sampling to maintain the pixel art style
    pil_image = pil_image.resize((size, size), Image.Resampling.NEAREST)

    # create a byte buffer for the image
    # a byte buffer is a temporary storage area in memory that holds bytes of data.
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Fashion-MNIST-Label": LABELS[label],
            "X-Fashion-MNIST-Label-Index": str(label)
        }
    )

@app.get("/predict/sample/{sample_id}", tags=["Predict Sample"])
async def predict_sample(sample_id: int, model_path: str = "./models/model.pth") -> Dict[str, str]:
    """
        Get model prediction for a specific sample from the dataset.
    """
    config = OmegaConf.load(CONFIG_PATH)
    loader = get_test_loader(config.paths.processed_dir, config.data.api_batch_size)

    if sample_id < 0 or sample_id >= len(loader.dataset):
        raise HTTPException(status_code=404, detail="Sample ID out of range")

    image, true_label = loader.dataset[sample_id]
    model = load_model(config=config, path=model_path)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        pred_label = output.argmax(dim=1).item()
        confidence = float(output.softmax(dim=1).max())

    return {
        "prediction": LABELS[pred_label],
        "prediction_index": str(pred_label),
        "true_label": LABELS[true_label],
        "true_label_index": str(true_label),
        "confidence": str(round(float(confidence), 5))
    }

@app.post("/predict/upload", tags=["Predict File"])
async def predict_upload(file: UploadFile = File(...), model_path: str = "./models/model.pth") -> Dict[str, str]:
    """
        Get prediction for an uploaded image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except HTTPException as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except (IOError, OSError) as exc:  
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    try:
        tensor = transform(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Error processing image") from exc

    config = OmegaConf.load(CONFIG_PATH)

    model = load_model(config=config, path=model_path)
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
        pred_label = output.argmax(dim=1).item()
        confidence = float(output.softmax(dim=1).max())

    return {
        "prediction": LABELS[pred_label],
        "prediction_index": str(pred_label),
        "confidence": str(round(float(confidence), 5))
    }