import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.project.api import app


@pytest.fixture
def client():
    """Fixture to create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


@patch("src.project.api.model_instance")
def test_predict_endpoint_success(mock_model_instance, client):
    """Test the predict endpoint with a successful prediction."""
    # Mock the model instance
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "predicted_label_idx": 0,
        "confidence": 0.85,
        "probabilities": [0.85, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01],
    }
    mock_model_instance.__bool__.return_value = True  # To pass the None check
    mock_model_instance.predict = mock_model.predict

    # Create a dummy audio file content (fake WAV)
    fake_audio_content = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x01\x00\x08\x00data\x00\x08\x00\x00"

    # Test the endpoint
    response = client.post("/predict", files={"file": ("test.wav", fake_audio_content, "audio/wav")})

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert data["filename"] == "test.wav"
    assert "prediction" in data
    assert data["prediction"] == "belly pain"  # CLASS_NAMES[0]
    assert "confidence" in data
    assert "details" in data


@patch("src.project.api.model_instance", new=None)
def test_predict_endpoint_no_model(client):
    """Test the predict endpoint when model is not loaded."""
    fake_audio_content = b"fake audio data"
    response = client.post("/predict", files={"file": ("test.wav", fake_audio_content, "audio/wav")})

    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "Model not loaded" in data["detail"]


@patch("src.project.api.model_instance")
def test_predict_endpoint_processing_error(mock_model_instance, client):
    """Test the predict endpoint when processing fails."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Processing error")
    mock_model_instance.__bool__.return_value = True
    mock_model_instance.predict = mock_model.predict

    fake_audio_content = b"fake audio data"
    response = client.post("/predict", files={"file": ("test.wav", fake_audio_content, "audio/wav")})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Processing error" in data["detail"]
