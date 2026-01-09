import librosa
import torch
import typer

from pathlib import Path
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.raw_path = data_path
        self.processed_path = data_path.parent / "processed"
        if self.processed_path.exists() and any(self.processed_path.iterdir()):
            # Load from processed data
            self.files = []
            self.labels = []
            for folder in self.processed_path.iterdir():
                if folder.is_dir():
                    label = folder.name
                    for file_path in folder.glob("*.pt"):
                        self.files.append(file_path)
                        self.labels.append(label)
        else:
            # Load from raw data
            self.files = []
            self.labels = []
            for folder in self.raw_path.iterdir():
                if folder.is_dir():
                    label = folder.name
                    for file_path in folder.glob("*"):
                        if file_path.suffix.lower() in [".wav", ".3gp"]:
                            self.files.append(file_path)
                            self.labels.append(label)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        file_path = self.files[index]
        label = self.labels[index]
        label_idx = self.label_to_idx[label]
        if self.processed_path.exists() and any(self.processed_path.iterdir()):
            # Load from processed
            waveform = torch.load(str(file_path))
        else:
            # Load from raw
            waveform, sr = librosa.load(str(file_path), sr=16000)
            waveform = torch.tensor(waveform).float()
        return {"input_values": waveform, "label": label_idx, "label_name": label, "file_path": file_path}

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        for folder in self.raw_path.iterdir():
            if folder.is_dir():
                label = folder.name
                (output_folder / label).mkdir(exist_ok=True)
                for file_path in folder.glob("*"):
                    if file_path.suffix.lower() in [".wav", ".3gp"]:
                        waveform, sr = librosa.load(str(file_path), sr=16000)
                        waveform = torch.tensor(waveform).float()
                        output_path = output_folder / label / f"{file_path.stem}.pt"
                        torch.save(waveform, output_path)


def preprocess(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
