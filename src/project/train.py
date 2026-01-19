import torch
import typer
import logging
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from tqdm import tqdm
from typing import List, Dict, Optional

# Import your local modules
from src.project.data import MyDataset
from src.project.model import HubertClassifier
from sklearn.metrics import classification_report
import torchaudio.functional as F_audio

app = typer.Typer()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        mask_freq_width: int = 1000  # Max width of freq mask in Hz
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
                input_values_shifted = F_audio.pitch_shift(
                    input_values_unsqueezed, 
                    self.sample_rate, 
                    n_steps
                )
                input_values = input_values_shifted.squeeze(1)
                
                # Resizing usually happens in pitch shift, forcing us to crop/pad back to original size
                # However, F_audio.pitch_shift usually maintains shape. 
                # If shape mismatches occur, we safety-crop:
                if input_values.shape[-1] != attention_mask.shape[-1]:
                     input_values = input_values[..., :attention_mask.shape[-1]]

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
                input_values = F_audio.bandreject_biquad(
                    input_values, 
                    self.sample_rate, 
                    central_freq, 
                    Q
                )

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

@app.command()
def train(
    data_path: Path = Path("data/raw"),
    processed_path: Path = Path("data/processed"),
    output_dir: Path = Path("models/checkpoints"),
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-5,  # Slightly higher start for scheduler
    weight_decay: float = 0.01,
    seed: int = 42,
    device_name: str = "auto",
    augment_prob: float = 0.5,
):
    """
    Train the HuBERT Classifier with Augmentation and Gradient Clipping.
    """
    seed_everything(seed)
    
    # --- 1. Setup Device ---
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info(f"Using device: {device}")

    # --- 2. Prepare Data ---
    if not processed_path.exists():
        logger.warning(f"Processed path {processed_path} does not exist. Ensure data is ready.")

    logger.info("Loading datasets...")
    train_dataset = MyDataset(data_path, split="train")
    val_dataset = MyDataset(data_path, split="val")
    
    num_labels = len(train_dataset.label_to_idx)
    logger.info(f"Detected {num_labels} classes.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # --- 3. Initialize Model & Components ---
    logger.info("Initializing HuBERT model...")
    model = HubertClassifier(num_labels=num_labels)
    model.to(device)

    # Augmenter
    augmenter = WaveformAugmenter(
        sample_rate=16000,  # HuBERT standard
        p=augment_prob,
        pitch_shift_max=2,  # +/- 2 semitones
        mask_freq_width=1500 # Cut up to 1500Hz bands
    )
    augmenter.to(device)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler: Linear warmup is standard for Transformers
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        total_steps=total_steps, 
        pct_start=0.1 # 10% warmup
    )

    # --- 4. Training Loop ---
    best_val_f1 = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

        # --- Train Step ---
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Apply Augmentation (Only during training)
            # We use no_grad because we don't want to learn the parameters of the noise generation
            with torch.no_grad():
                input_values = augmenter(input_values, attention_mask)

            # Forward pass
            loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

            # Backward pass
            loss.backward()

            # GRADIENT CLIPPING (Essential for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        val_f1_macro = report['macro avg']['f1-score']
        val_acc = report['accuracy']

        logger.info(f"Epoch {epoch+1} Summary:")
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
    app()