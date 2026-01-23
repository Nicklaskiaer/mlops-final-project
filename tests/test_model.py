import torch
from src.project.model import HubertClassifier


def test_model_instantiation():
    """Test that the HubertClassifier can be instantiated with default parameters."""
    model = HubertClassifier()
    assert isinstance(model, HubertClassifier)
    assert model.target_sampling_rate == 16000
    assert model.hf_model is not None
    assert model.feature_extractor is not None


def test_model_instantiation_with_freeze():
    """Test instantiation with freeze_feature_encoder=True."""
    model = HubertClassifier(freeze_feature_encoder=True)
    assert isinstance(model, HubertClassifier)
    # Check that feature extractor parameters are frozen
    for param in model.hf_model.hubert.feature_extractor.parameters():
        assert not param.requires_grad


def test_model_forward_pass():
    """Test the forward pass with dummy input."""
    model = HubertClassifier()
    model.eval()  # Set to eval mode for inference

    # Create dummy input: batch_size=1, sequence_length=16000 (1 second at 16kHz)
    dummy_input = torch.randn(1, 16000)

    with torch.no_grad():
        output = model(dummy_input)

    # Output should be logits tensor with shape (batch_size, num_labels)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 8)  # num_labels=8 by default


def test_model_predict_with_tensor():
    """Test the predict method with a tensor input."""
    model = HubertClassifier()
    model.eval()

    # Dummy audio tensor
    dummy_audio = torch.randn(16000)  # 1 second

    result = model.predict(dummy_audio)

    # Check result structure
    assert isinstance(result, dict)
    assert "predicted_label_idx" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["predicted_label_idx"], int)
    assert 0 <= result["predicted_label_idx"] < 8
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 8


def test_ensure_audio_format_tensor():
    """Test _ensure_audio_format with tensor input."""
    model = HubertClassifier()
    input_tensor = torch.randn(16000)
    result = model._ensure_audio_format(input_tensor)
    assert isinstance(result, torch.Tensor) or isinstance(result, type(input_tensor.numpy()))


def test_ensure_audio_format_numpy():
    """Test _ensure_audio_format with numpy array input."""
    import numpy as np

    model = HubertClassifier()
    input_array = np.random.randn(16000)
    result = model._ensure_audio_format(input_array)
    assert isinstance(result, np.ndarray)


def test_model_forward_pass_with_labels():
    """Test the model forward pass with labels (returns loss and logits)."""
    model = HubertClassifier()

    batch_size, seq_len = 2, 16000
    input_values = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 8, (batch_size,))

    loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert logits.shape == (batch_size, 8)  # num_labels
    assert loss > 0  # loss should be positive
