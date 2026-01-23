import pytest
import torch
import numpy as np
from unittest.mock import patch
from src.project.train import WaveformAugmenter, seed_everything, collate_fn


class TestWaveformAugmenter:
    """Test the WaveformAugmenter class."""

    @pytest.fixture
    def augmenter(self):
        """Fixture to create a WaveformAugmenter instance."""
        return WaveformAugmenter(p=1.0)  # Set p=1.0 to always apply augmentations

    def test_augmenter_initialization(self, augmenter):
        """Test that the augmenter initializes correctly."""
        assert augmenter.sample_rate == 16000
        assert augmenter.p == 1.0
        assert augmenter.noise_level == 0.005

    def test_augmenter_forward_shape_preservation(self, augmenter):
        """Test that the augmenter preserves input shape."""
        batch_size, time = 2, 16000
        input_values = torch.randn(batch_size, time)
        attention_mask = torch.ones(batch_size, time)

        output = augmenter(input_values, attention_mask)

        assert output.shape == input_values.shape
        assert output.device == input_values.device

    def test_augmenter_applies_augmentations(self, augmenter):
        """Test that augmentations change the input."""
        input_values = torch.randn(1, 16000)
        attention_mask = torch.ones(1, 16000)
        original = input_values.clone()

        output = augmenter(input_values, attention_mask)

        # Since p=1.0, augmentations should be applied, so output should differ
        assert not torch.allclose(output, original, atol=1e-6)

    def test_augmenter_respects_mask(self, augmenter):
        """Test that the augmenter respects the attention mask."""
        batch_size, time = 2, 16000
        input_values = torch.randn(batch_size, time)
        attention_mask = torch.zeros(batch_size, time)
        attention_mask[:, :time//2] = 1  # Only first half is real

        output = augmenter(input_values, attention_mask)

        # Padded parts should be zeroed out
        assert torch.allclose(output[:, time//2:], torch.zeros_like(output[:, time//2:]))


def test_seed_everything():
    """Test that seed_everything sets seeds correctly."""
    seed = 42
    seed_everything(seed)

    # Test that random generators are seeded
    a = np.random.rand()
    b = torch.rand(1)
    c = torch.randint(0, 10, (1,))

    seed_everything(seed)  # Reset seed

    assert a == np.random.rand()
    assert torch.allclose(b, torch.rand(1))
    assert torch.equal(c, torch.randint(0, 10, (1,)))


def test_collate_fn():
    """Test the collate_fn function."""
    batch = [
        {"input_values": torch.randn(100), "label": 0},
        {"input_values": torch.randn(150), "label": 1},
        {"input_values": torch.randn(120), "label": 2},
    ]

    collated = collate_fn(batch)

    assert "input_values" in collated
    assert "attention_mask" in collated
    assert "labels" in collated

    # Check shapes
    assert collated["input_values"].shape[0] == 3  # batch size
    assert collated["input_values"].shape[1] == 150  # max length
    assert collated["attention_mask"].shape == collated["input_values"].shape
    assert collated["labels"].shape == (3,)

    # Check attention mask
    assert torch.all(collated["attention_mask"][0, :100] == 1)
    assert torch.all(collated["attention_mask"][0, 100:] == 0)
    assert torch.all(collated["attention_mask"][1, :150] == 1)
    assert torch.all(collated["attention_mask"][1, 150:] == 0)
    assert torch.all(collated["attention_mask"][2, :120] == 1)
    assert torch.all(collated["attention_mask"][2, 120:] == 0)

    # Check labels
    assert torch.equal(collated["labels"], torch.tensor([0, 1, 2]))