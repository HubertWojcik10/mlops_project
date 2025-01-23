import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

# import the api
from src.mlops_project.api import app

# create a test client
client = TestClient(app)

def test_root_endpoint() -> None:
    """
        Test the root endpoint
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "status-code" in response.json()

def test_sample_image_endpoint() -> None:
    """
        Test the sample image endpoint.
    """
    response = client.get("/sample/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "X-Fashion-MNIST-Label" in response.headers
    assert "X-Fashion-MNIST-Label-Index" in response.headers

    # test with each possible value of the dataset
    datasets = ["train", "test", "val"]
    for dataset in datasets:
        response = client.get(f"/sample/image?dataset={dataset}")
        assert response.status_code == 200

def test_predict_sample_endpoint() -> None:
    """
        Test the predict sample endpoint.
    """

    response = client.get("/predict/sample/0")
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "prediction_index" in response.json()
    assert "true_label" in response.json()
    assert "true_label_index" in response.json()
    assert "confidence" in response.json()

    # Test invalid sample ID
    response = client.get("/predict/sample/-1")
    assert response.status_code == 404

def test_predict_upload_endpoint() -> None:
    """
        Test the predict upload endpoint.
    """

    # create dummy image for testing
    img = Image.new('RGB', (28, 28), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict/upload",
        files={"file": ("test_image.png", img_byte_arr, "image/png")}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "prediction_index" in response.json()
    assert "confidence" in response.json()

    response = client.post(
        "/predict/upload",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )
    assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__])
