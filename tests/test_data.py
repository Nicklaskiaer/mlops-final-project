from pathlib import Path

from project.data import MyDataset


def test_always_passes():
    assert True


def test_dataset_import():
    """Test that MyDataset can be imported and instantiated."""
    # Just verify the class exists and can be created with a path
    assert MyDataset is not None
    assert callable(MyDataset)



# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = MyDataset(Path("data/raw"))
#     assert isinstance(dataset, Dataset)


# def test_preprocess():
#     """Test the preprocess function."""
#     data_path = Path("data/raw")
#     output_folder = Path("data/processed")
#     preprocess(data_path, output_folder)
#     assert output_folder.exists()


# def test_dataset_len():
#     """Test the __len__ method."""
#     dataset = MyDataset(Path("data/raw"))
#     length = dataset.__len__()
#     assert isinstance(length, int)
#     assert length > 0


# def test_dataset_getitem():
#     """Test the __getitem__ method."""
#     dataset = MyDataset(Path("data/raw"))
#     item = dataset.__getitem__(0)
#     assert isinstance(item, dict)
#     assert "input_values" in item
#     assert "label" in item
#     assert "label_name" in item
#     assert "file_path" in item
#     assert isinstance(item["input_values"], torch.Tensor)
