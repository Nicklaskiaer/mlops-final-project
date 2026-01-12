import av
import librosa
import torch
import typer

from pathlib import Path
from torch.utils.data import Dataset
import random
import numpy as np


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, split="test") -> None:
        random.seed(42)
        self.raw_path = data_path
        self.processed_path = data_path.parent / "processed"

        self.files, self.labels, self.classes, self.label_to_idx = self.process_indices(split)

    def process_indices(self, split: str):
        if self.processed_path.exists() and any(self.processed_path.iterdir()):
            # Load from processed data
            files = []
            labels = []
            for folder in self.processed_path.iterdir():
                if folder.is_dir():
                    label = folder.name
                    for file_path in folder.glob("*.pt"):
                        files.append(file_path)
                        labels.append(label)
        else:
            # Load from raw data
            files = []
            labels = []
            for folder in self.raw_path.iterdir():
                if folder.is_dir():
                    label = folder.name
                    for file_path in folder.glob("*"):
                        if file_path.suffix.lower() in [".wav", ".3gp"]:
                            files.append(file_path)
                            labels.append(label)
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        classes = np.array([label_to_idx[label] for label in labels])
        files = np.array(files)

        random_indexes = np.arange(0, len(files))
        np.random.shuffle(random_indexes)
        props_cum = {"train": 0.8, "val": 0.9, "test": 1.0}
        N = len(classes)

        if split == "train":
            train_indexes = random_indexes[: int(props_cum["train"] * N)]
            classes = classes[train_indexes]
            files = files[train_indexes]
            labels = [labels[i] for i in train_indexes]
        elif split == "val":
            val_indexes = random_indexes[int(props_cum["train"] * N) : int(props_cum["val"] * N)]

            classes = classes[val_indexes]
            files = files[val_indexes]
            labels = [labels[i] for i in val_indexes]
        elif split == "test":
            test_indexes = classes[int(props_cum["val"] * N) : int(props_cum["test"] * N)]

            classes = classes[test_indexes]
            files = files[test_indexes]
            labels = [labels[i] for i in test_indexes]

        return files, labels, classes, label_to_idx

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        file_path = self.files[index]
        label = self.labels[index]
        label_idx = self.label_to_idx[label]

        try:
            if self.processed_path.exists() and any(self.processed_path.iterdir()):
                waveform = torch.load(str(file_path))
            else:
                raise FileNotFoundError("Processed data not found")
        except Exception:
            print("Must preprocess raw files")

        return {"input_values": waveform, "label": label_idx, "label_name": label, "file_path": file_path}

    def _load_3gp_audio(self, file_path: Path) -> torch.Tensor:
        """Load audio from .3gp file using PyAV."""
        container = av.open(str(file_path))
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000)  # type: ignore
        waveform = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            for f in resampled_frames:
                waveform.append(torch.tensor(f.to_ndarray()).squeeze().float())
        if waveform:
            return torch.cat(waveform, dim=0)
        else:
            return torch.tensor([]).float()

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        for folder in self.raw_path.iterdir():
            if folder.is_dir():
                label = folder.name
                (output_folder / label).mkdir(exist_ok=True)
                for file_path in folder.glob("*"):
                    if file_path.suffix.lower() in [".wav", ".3gp"]:
                        if file_path.suffix.lower() == ".3gp":
                            waveform = self._load_3gp_audio(file_path)
                        else:
                            waveform, sr = librosa.load(str(file_path), sr=16000)
                            waveform = torch.tensor(waveform).float()
                        output_path = output_folder / label / f"{file_path.stem}.pt"
                        torch.save(waveform, output_path)


def preprocess(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print("Preprocess finished successfully")


if __name__ == "__main__":
    typer.run(preprocess)
