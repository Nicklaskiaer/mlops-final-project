import torch
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from typing import List, Dict

# Import your local modules
from src.project.data import MyDataset
from src.project.model import HubertClassifier

app = typer.Typer()


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Same collate function as train.py to handle variable length audio.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]

    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)

    attention_mask = torch.zeros_like(input_values_padded)
    for i, item in enumerate(input_values):
        attention_mask[i, : item.shape[0]] = 1

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {"input_values": input_values_padded, "attention_mask": attention_mask, "labels": labels_tensor}


@app.command()
def evaluate(
    model_path: Path = Path("models/checkpoints/best_hubert_model.pt"),
    data_path: Path = Path("data/raw"),
    batch_size: int = 16,
    device_name: str = "auto",
    save_matrix: bool = True,
):
    """
    Evaluate the trained HuBERT model on the Test set.
    """
    # --- 1. Setup Device (M1/CUDA Support) ---
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    # --- 2. Load Test Data ---
    print("Loading test dataset...")
    test_dataset = MyDataset(data_path, split="test")

    # Get class names for better reporting
    # Inverting the label_to_idx dictionary: {0: 'belly_pain', 1: 'hungry', ...}
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_labels = len(class_names)

    print(f"Classes: {class_names}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 3. Load Model ---
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        raise typer.Exit(code=1)

    print(f"Loading model from {model_path}...")

    # IMPORTANT: Ensure model_name matches what you trained with (distilhubert)
    model = HubertClassifier(model_name="ntu-spml/distilhubert", num_labels=num_labels)

    # Load weights
    # map_location ensures we can load a GPU model on CPU/MPS if needed
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --- 4. Inference Loop ---
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # We only need logits here
            logits = model(input_values, attention_mask=attention_mask)

            # Handle the tuple return type we fixed earlier
            if isinstance(logits, tuple):
                logits = logits[1]

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Metrics & Reporting ---
    print("\n" + "=" * 30)
    print("CLASSIFICATION REPORT")
    print("=" * 30)

    # FIX: Define the list of all possible label indices [0, 1, ... 7]
    all_label_ids = list(range(num_labels))

    # Detailed text report
    report = classification_report(
        all_labels,
        all_preds,
        labels=all_label_ids,  # <--- ADD THIS: Force it to look for all 8 IDs
        target_names=class_names,  # Now this matches the list above
        zero_division=0,  # Handles classes that have no predictions/samples
    )
    print(report)

    # Confusion Matrix
    if save_matrix:
        # FIX: Also pass labels here so the matrix is always 8x8
        cm = confusion_matrix(all_labels, all_preds, labels=all_label_ids)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - HuBERT Audio Classification")

        save_path = Path("reports/figures/confusion_matrix.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"\nConfusion matrix saved to: {save_path}")


if __name__ == "__main__":
    app()
