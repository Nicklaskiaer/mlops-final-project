from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from project.data import MyDataset, preprocess


def test_my_dataset(tmp_path: Path):
    """Test the MyDataset class."""
    data_path = tmp_path / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    # Make a minimal processed structure so __init__/process_indices has something to index
    processed = tmp_path / "processed"
    (processed / "cat").mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(10), processed / "cat" / "a.pt")

    dataset = MyDataset(data_path)
    assert isinstance(dataset, Dataset)


def test_preprocess(tmp_path: Path, monkeypatch):
    """Test the preprocess function."""
    import librosa

    data_path = tmp_path / "raw"
    output_folder = tmp_path / "processed"
    (data_path / "cat").mkdir(parents=True, exist_ok=True)

    # Create a fake wav file (content doesn't matter because we patch librosa.load)
    (data_path / "cat" / "a.wav").write_bytes(b"fake wav")

    def fake_load(_path: str, sr: int = 16000):
        return np.zeros(sr, dtype=np.float32), sr

    monkeypatch.setattr(librosa, "load", fake_load)

    preprocess(data_path, output_folder)

    # Existence check (as before)
    assert output_folder.exists()
    out_file = output_folder / "cat" / "a.pt"
    assert out_file.exists()

    # Added: verify the saved tensor is what we expect (relevant correctness check)
    waveform = torch.load(out_file)
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.numel() == 16000


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
    """
    Parametrized test to cover the train/val/test branches in process_indices
    while keeping the overall test style simple.
    """
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


def test_preprocess_handles_3gp(tmp_path: Path, monkeypatch):
    """Test that preprocess handles .3gp files (without requiring real decoding)."""
    data_path = tmp_path / "raw"
    output_folder = tmp_path / "processed"
    (data_path / "dog").mkdir(parents=True, exist_ok=True)

    # Create a fake .3gp file
    (data_path / "dog" / "b.3gp").write_bytes(b"fake 3gp")

    # Monkeypatch decoding to avoid relying on PyAV in unit tests
    def fake_load_3gp_audio(self, file_path: Path) -> torch.Tensor:
        return torch.ones(16000).float()

    monkeypatch.setattr(MyDataset, "_load_3gp_audio", fake_load_3gp_audio)

    preprocess(data_path, output_folder)

    out_file = output_folder / "dog" / "b.pt"
    assert out_file.exists()

    waveform = torch.load(out_file)
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.numel() == 16000
