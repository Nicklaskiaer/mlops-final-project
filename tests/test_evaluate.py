import torch
from src.project.evaluate import collate_fn


def test_collate_fn():
    """Test the collate_fn function in evaluate.py (same as train.py)."""
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