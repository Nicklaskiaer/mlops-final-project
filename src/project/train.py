import torch
import typer
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import List, Dict

# Import your local modules
# Assuming src is in python path or run as module
from src.project.data import MyDataset
from src.project.model import HubertClassifier

app = typer.Typer()


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad audio waveforms and create attention masks.
    """
    # 1. Extract inputs and labels
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]

    # 2. Pad sequences to max length in this batch
    # batch_first=True -> [Batch, Time]
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)

    # 3. Create Attention Mask (1 for real data, 0 for padding)
    # This is critical for Transformers to ignore the padded silence
    attention_mask = torch.zeros_like(input_values_padded)
    for i, item in enumerate(input_values):
        attention_mask[i, : item.shape[0]] = 1

    # 4. Stack labels
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {"input_values": input_values_padded, "attention_mask": attention_mask, "labels": labels_tensor}


@app.command()
def train(
    data_path: Path = Path("data/raw"),
    processed_path: Path = Path("data/processed"),
    output_dir: Path = Path("models/checkpoints"),
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,  # Low LR is standard for fine-tuning Transformers
    device_name: str = "auto",
):
    """
    Train the HuBERT Classifier.
    """
    # --- 1. Setup Device ---
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    # --- 2. Prepare Data ---
    # Ensure processed data exists (or the dataset class handles it)
    if not processed_path.exists() or not any(processed_path.iterdir()):
        print("Processed data not found. Please run 'python -m src.project.data' first.")
        # Optional: You could call preprocess() here if you imported it.

    print("Loading datasets...")
    train_dataset = MyDataset(data_path, split="train")
    val_dataset = MyDataset(data_path, split="val")

    # Determine number of classes dynamically from the dataset
    # We use 'set' to get unique labels across the full dataset logic if possible,
    # but here we rely on the dataset object having 'label_to_idx' populated.
    num_labels = len(train_dataset.label_to_idx)
    print(f"Detected {num_labels} classes: {train_dataset.label_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to >0 if not on Windows/issues arise
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 3. Initialize Model ---
    print("Initializing HuBERT model...")
    model = HubertClassifier(num_labels=num_labels)
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # --- 4. Training Loop ---
    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train Step
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass (returns loss, logits because labels are provided)
            loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy for display
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, logits = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --- 5. Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / "best_hubert_model.pt"
            print(f"Validation loss improved. Saving model to {save_path}")

            # Save the state dict
            torch.save(model.state_dict(), save_path)

            # Optional: Save HF format if you want to use .from_pretrained() later
            # model.hf_model.save_pretrained(output_dir / "hf_format")

    print("\nTraining complete.")


if __name__ == "__main__":
    app()
