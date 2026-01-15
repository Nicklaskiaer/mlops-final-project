import torch
import typer
import json
import torchaudio
import av  # <--- CRITICAL: PyAV handles the .3gp files
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, split="test") -> None:
        self.raw_path = data_path
        self.processed_path = data_path.parent / "processed"

        # Load indices and save class mapping
        self.files, self.labels, self.classes, self.label_to_idx = self.process_indices(split)

    def process_indices(self, split: str):
        # 1. Gather all files first
        files = []
        labels = []

        # Logic to choose raw or processed source
        # If processed folder has files, use them. Otherwise scan raw.
        if self.processed_path.exists() and any(self.processed_path.iterdir()):
            source_dir = self.processed_path
            is_processed = True
        else:
            source_dir = self.raw_path
            is_processed = False

        for folder in source_dir.iterdir():
            if folder.is_dir():
                label = folder.name
                # If processed, look for .pt; if raw, look for audio ext
                extensions = ["*.pt"] if is_processed else ["*.wav", "*.3gp", "*.mp3"]
                for ext in extensions:
                    for file_path in folder.glob(ext):
                        files.append(file_path)
                        labels.append(label)

        # 2. Create deterministic mappings
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Save labels for API use (Only need to do this once, e.g. during training)
        if split == "train":
            with open(self.raw_path.parent / "labels.json", "w") as f:
                json.dump(unique_labels, f)

        classes = np.array([label_to_idx[label] for label in labels])
        files = np.array(files)

        # 3. Deterministic Shuffle
        np.random.seed(42)
        random_indexes = np.arange(0, len(files))
        np.random.shuffle(random_indexes)

        # Split logic
        props_cum = {"train": 0.8, "val": 0.9, "test": 1.0}
        N = len(classes)

        if split == "train":
            idxs = random_indexes[: int(props_cum["train"] * N)]
        elif split == "val":
            idxs = random_indexes[int(props_cum["train"] * N) : int(props_cum["val"] * N)]
        elif split == "test":
            idxs = random_indexes[int(props_cum["val"] * N) : int(props_cum["test"] * N)]
        else:
            raise ValueError(f"Unknown split {split}")

        return files[idxs], [labels[i] for i in idxs], classes[idxs], label_to_idx

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        file_path = self.files[index]
        label = self.labels[index]
        label_idx = self.label_to_idx[label]

        waveform = None

        try:
            # Load data
            if file_path.suffix == ".pt":
                waveform = torch.load(str(file_path))
            else:
                # Fallback for raw loading (using helper)
                waveform = self._load_audio(file_path)
        except Exception as e:
            # If a specific file fails, return silence to prevent crash
            print(f"Error loading {file_path}: {e}")
            waveform = torch.zeros(16000)

        return {"input_values": waveform, "label": label_idx, "label_name": label, "file_path": str(file_path)}

    def _load_audio(self, file_path: Path) -> torch.Tensor:
        """
        Smart loader: uses PyAV for .3gp and Torchaudio for .wav
        """
        path_str = str(file_path)

        # --- STRATEGY A: Handle .3gp with PyAV ---
        if file_path.suffix.lower() == ".3gp":
            try:
                container = av.open(path_str)
                audio_stream = next(s for s in container.streams if s.type == "audio")

                # Resample immediately using PyAV's internal tools
                resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000)

                parts = []
                for frame in container.decode(audio_stream):
                    resampled_frames = resampler.resample(frame)
                    for f in resampled_frames:
                        # Convert to torch tensor
                        parts.append(torch.from_numpy(f.to_ndarray()).float())

                if parts:
                    waveform = torch.cat(parts, dim=1).squeeze()
                else:
                    return torch.zeros(16000)

                return waveform

            except Exception as e:
                # If AV fails, raise so we see it
                raise RuntimeError(f"PyAV failed on {file_path}: {e}")

        # --- STRATEGY B: Handle everything else with Torchaudio ---
        else:
            wav, sr = torchaudio.load(path_str)

            # Resample
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)

            # Mono
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            return wav.squeeze()


def preprocess(data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")

    # Instantiate dataset just to access the helper method easily,
    # but we will manually iterate ALL files in data/raw to ensure 100% coverage.
    dummy_dataset = MyDataset(data_path, split="train")

    output_folder.mkdir(parents=True, exist_ok=True)

    for folder in data_path.iterdir():
        if folder.is_dir():
            label = folder.name
            print(f"Processing {label}...")
            (output_folder / label).mkdir(exist_ok=True, parents=True)

            for file_path in folder.glob("*"):
                # Filter for audio extensions
                if file_path.suffix.lower() in [".wav", ".3gp", ".mp3"]:
                    try:
                        # Use the robust loader
                        waveform = dummy_dataset._load_audio(file_path)

                        # Save as .pt tensor
                        torch.save(waveform, output_folder / label / f"{file_path.stem}.pt")
                    except Exception as e:
                        print(f"Failed {file_path}: {e}")

    print("Preprocess finished successfully")


if __name__ == "__main__":
    typer.run(preprocess)
