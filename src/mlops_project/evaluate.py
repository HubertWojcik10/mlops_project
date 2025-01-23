from data import get_test_loader
from model import get_model
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("evaluation.log", rotation="10 MB", level="INFO")


def evaluate(config: DictConfig) -> None:
    """
    Evaluate the model on the test set.
    """
    logger.info("Evaluation started")
    test_loader = get_test_loader(config.paths.processed_dir, config.data.batch_size)
    model = get_model(config)
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                logger.debug(f"Batch processed: predicted {predicted}, actual {labels}")

    # Calculate average loss
    average_loss = test_loss / len(test_loader)

    # Calculate accuracy
    accuracy = 100 * correct / total

    # print(f"Test Loss: {average_loss:.4f}")
    # print(f"Test Accuracy: {accuracy:.2f}%")

    logger.info(f"Test Loss: {average_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.2f}%")

    logger.success("Evaluation completed")
