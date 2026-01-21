from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from project.data import MyDataset, preprocess


def test_always_passes():
    assert True


def test_my_dataset(tmp_path: Path):
    """Test the MyDataset class."""
    data_path = tmp_path / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    processed = tmp_path / "processed"
    (processed / "cat").mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(10), processed / "cat" / "a.pt")

    dataset = MyDataset(data_path)
    assert isinstance(dataset, Dataset)


def test_preprocess_handles_failed_wav_gracefully(tmp_path: Path, capsys):
    """
    preprocess() should not crash if wav decoding fails.
    It should print an error and continue.
    """
    data_path = tmp_path / "raw"
    output_folder = tmp_path / "processed"
    (data_path / "cat").mkdir(parents=True, exist_ok=True)

    (data_path / "cat" / "a.wav").write_bytes(b"fake wav")

    preprocess(data_path, output_folder)

    captured = capsys.readouterr().out
    assert "Failed" in captured
    assert "a.wav" in captured
    assert output_folder.exists()


def test_dataset_len(tmp_path: Path):
    """Test the __len__ method."""
    data_path = tmp_path / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    processed = tmp_path / "processed"
    (processed / "cat").mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(10), processed / "cat" / "a.pt")
    torch.save(torch.randn(10), processed / "cat" / "b.pt")

    dataset = MyDataset(data_path)
    length = dataset.__len__()
    assert isinstance(length, int)
    assert length > 0


def test_dataset_getitem(tmp_path: Path):
    """Test the __getitem__ method."""
    data_path = tmp_path / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    processed = tmp_path / "processed"
    (processed / "cat").mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(10), processed / "cat" / "a.pt")

    dataset = MyDataset(data_path)
    item = dataset.__getitem__(0)

    assert isinstance(item, dict)
    assert "input_values" in item
    assert "label" in item
    assert "label_name" in item
    assert "file_path" in item
    assert isinstance(item["input_values"], torch.Tensor)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_process_indices_splits(tmp_path: Path, split: str):
    """Cover train/val/test branches in process_indices."""
    data_path = tmp_path / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    processed = tmp_path / "processed"
    (processed / "cat").mkdir(parents=True, exist_ok=True)
    (processed / "dog").mkdir(parents=True, exist_ok=True)

    for i in range(10):
        torch.save(torch.randn(10), processed / "cat" / f"cat_{i}.pt")
        torch.save(torch.randn(10), processed / "dog" / f"dog_{i}.pt")

    dataset = MyDataset(data_path, split=split)
    assert len(dataset) > 0


def test_preprocess_handles_failed_3gp_gracefully(tmp_path: Path, capsys):
    """
    preprocess() should handle .3gp decode failures gracefully.
    """
    data_path = tmp_path / "raw"
    output_folder = tmp_path / "processed"
    (data_path / "dog").mkdir(parents=True, exist_ok=True)

    (data_path / "dog" / "b.3gp").write_bytes(b"fake 3gp")

    preprocess(data_path, output_folder)

    captured = capsys.readouterr().out
    assert "Failed" in captured
    assert "b.3gp" in captured
    assert output_folder.exists()
