from pathlib import Path
from unittest.mock import patch

from torch.utils.data import Dataset

from project.data import MyDataset, preprocess


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(Path("data/raw"))
    assert isinstance(dataset, Dataset)


def test_preprocess():
    """Test the preprocess function."""
    data_path = Path("data/raw")
    output_folder = Path("data/processed")
    with patch('builtins.print') as mock_print:
        preprocess(data_path, output_folder)
        mock_print.assert_called_once_with("Preprocessing data...")


def test_dataset_len():
    """Test the __len__ method."""
    dataset = MyDataset(Path("data/raw"))
    # Since not implemented, it returns None, but for coverage
    length = dataset.__len__()
    assert length is None


def test_dataset_getitem():
    """Test the __getitem__ method."""
    dataset = MyDataset(Path("data/raw"))
    # Since not implemented, it returns None
    item = dataset.__getitem__(0)
    assert item is None
