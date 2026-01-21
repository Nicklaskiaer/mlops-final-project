import torch
import hydra
import logging
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

# Import your local modules
from src.project.data import MyDataset
from src.project.model import HubertClassifier
from sklearn.metrics import classification_report
import torchaudio.functional as F_audio

# --- Setup Logging ---
logger = logging.getLogger(__name__)


class WaveformAugmenter(torch.nn.Module):
    """
    Advanced Waveform Augmentation on GPU.
    Includes: Gain, Noise, Polarity, Pitch Shift, and Frequency Masking (Band-Reject).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        p: float = 0.5,
        noise_level: float = 0.005,
        pitch_shift_max: int = 2,  # Max semitones to shift
        mask_freq_width: int = 1000,  # Max width of freq mask in Hz
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.p = p
        self.noise_level = noise_level
        self.pitch_shift_max = pitch_shift_max
        self.mask_freq_width = mask_freq_width

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: (Batch, Time)
            attention_mask: (Batch, Time)
        """
        # Ensure we are on the same device
        device = input_values.device
        batch_size = input_values.size(0)

        # 1. Random Gain (Volume)
        if torch.rand(1) < self.p:
            gains = torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.2)
            input_values = input_values * gains

        # 2. Additive Gaussian Noise
        if torch.rand(1) < self.p:
            noise = torch.randn_like(input_values) * self.noise_level
            input_values = input_values + noise

        # 3. Pitch Shifting
        # Note: This is computationally expensive! We apply it to a subset to save time.
        if torch.rand(1) < self.p:
            # We pick one random shift amount for the whole batch for efficiency,
            # or loop if you need per-sample diversity (slower).
            n_steps = random.randint(-self.pitch_shift_max, self.pitch_shift_max)
            if n_steps != 0:
                # pitch_shift expects (Batch, Channels, Time) or (Channels, Time)
                # We interpret (Batch, Time) as (Batch, 1, Time) for processing
                input_values_unsqueezed = input_values.unsqueeze(1)
                input_values_shifted = F_audio.pitch_shift(input_values_unsqueezed, self.sample_rate, n_steps)
                input_values = input_values_shifted.squeeze(1)

                # Resizing usually happens in pitch shift, forcing us to crop/pad back to original size
                # However, F_audio.pitch_shift usually maintains shape.
                # If shape mismatches occur, we safety-crop:
                if input_values.shape[-1] != attention_mask.shape[-1]:
                    input_values = input_values[..., : attention_mask.shape[-1]]

        # 4. Frequency Masking (via Band-Reject Filter)
        if torch.rand(1) < self.p:
            # Randomly select a frequency band to kill
            # We avoid very low freqs (0-200Hz) usually containing fundamental voice data
            freq_low = 200
            freq_high = self.sample_rate // 2 - self.mask_freq_width

            # Helper to ensure high > low
            if freq_high > freq_low:
                central_freq = random.uniform(freq_low, freq_high)

                # Quality factor (Q) determines the width. Higher Q = narrower cut.
                # We want a cut of ~mask_freq_width Hz.
                # Q = central_freq / bandwidth
                bandwidth = random.uniform(100, self.mask_freq_width)
                Q = central_freq / bandwidth

                # Apply Band-Reject Biquad Filter (GPU compatible)
                input_values = F_audio.bandreject_biquad(input_values, self.sample_rate, central_freq, Q)

        # 5. Polarity Inversion
        if torch.rand(1) < self.p:
            input_values = input_values * -1.0

        # CRITICAL: Re-apply mask to silence padding
        input_values = input_values * attention_mask

        return input_values


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to pad audio waveforms."""
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]

    # Pad sequences [Batch, Time]
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)

    # Create Attention Mask (1 for real data, 0 for padding)
    attention_mask = torch.zeros_like(input_values_padded)
    for i, item in enumerate(input_values):
        attention_mask[i, : item.shape[0]] = 1

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {"input_values": input_values_padded, "attention_mask": attention_mask, "labels": labels_tensor}


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Train the HuBERT Classifier with Augmentation and Gradient Clipping.
    Uses Hydra for configuration management.
    """
    # Log the configuration
    logger.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.seed)

    # --- 1. Setup Device ---
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")

    # --- 2. Prepare Data ---
    data_path = Path(cfg.data.raw_path)
    processed_path = Path(cfg.data.processed_path)

    if not processed_path.exists():
        logger.warning(f"Processed path {processed_path} does not exist. Ensure data is ready.")

    logger.info("Loading datasets...")
    train_dataset = MyDataset(data_path, split="train")
    val_dataset = MyDataset(data_path, split="val")

    num_labels = len(train_dataset.label_to_idx)
    logger.info(f"Detected {num_labels} classes.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- 3. Initialize Model & Components ---
    logger.info(f"Initializing model: {cfg.model.pretrained_name}")
    model = HubertClassifier(
        model_name=cfg.model.pretrained_name,
        num_labels=cfg.model.num_labels,
        freeze_feature_encoder=cfg.model.freeze_feature_encoder,
    )
    model.to(device)

    # Augmenter (only if enabled)
    augmenter = None
    if cfg.augmentation.enabled:
        augmenter = WaveformAugmenter(
            sample_rate=cfg.augmentation.sample_rate,
            p=cfg.augmentation.probability,
            noise_level=cfg.augmentation.noise_level,
            pitch_shift_max=cfg.augmentation.pitch_shift_max,
            mask_freq_width=cfg.augmentation.mask_freq_width,
        )
        augmenter.to(device)
        logger.info(f"Augmentation enabled with probability {cfg.augmentation.probability}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler: Linear warmup is standard for Transformers
    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.scheduler.pct_start,
    )

    # --- 4. Training Loop ---
    best_val_f1 = 0.0
    output_dir = Path(cfg.output.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{cfg.training.epochs}")

        # --- Train Step ---
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Train Ep {epoch + 1}")

        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Apply Augmentation (Only during training, if enabled)
            if augmenter is not None:
                with torch.no_grad():
                    input_values = augmenter(input_values, attention_mask)

            # Forward pass
            loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

            # Backward pass
            loss.backward()

            # GRADIENT CLIPPING (Essential for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradient_clip_norm)

            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()

            # Metrics
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # Memory cleanup for constrained devices
            del input_values, attention_mask, labels, loss, logits, preds
            if step % 20 == 0 and device.type in ["mps", "cuda"]:
                torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()

        avg_train_loss = train_loss_accum / len(train_loader)
        train_acc = train_correct / train_total

        # --- Validation Step ---
        model.eval()
        val_loss_accum = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

                val_loss_accum += loss.item()
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)

        # Calculate detailed metrics
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        val_f1_macro = report["macro avg"]["f1-score"]
        val_acc = report["accuracy"]

        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1_macro:.4f}")

        # --- Save Best Model ---
        # Saving based on F1 score is usually better than Loss for classification
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            save_path = output_dir / "best_hubert_model.pt"
            logger.info(f"New Best Model (F1: {val_f1_macro:.4f})! Saving to {save_path}")
            torch.save(model.state_dict(), save_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    train()
