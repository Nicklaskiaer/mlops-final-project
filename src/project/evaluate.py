import logging
import torch
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

# Import your local modules
from src.project.data import MyDataset
from src.project.model import HubertClassifier

# --- Setup Logging ---
logger = logging.getLogger(__name__)


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


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate the trained HuBERT model on the Test set.
    Uses Hydra for configuration management.
    """
    # Log the configuration
    logger.info(f"Evaluation configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Setup Device (M1/CUDA Support) ---
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

    # --- 2. Load Test Data ---
    data_path = Path(cfg.data.raw_path)
    logger.info("Loading test dataset...")
    test_dataset = MyDataset(data_path, split="test")

    # Get class names for better reporting
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_labels = len(class_names)

    logger.info(f"Classes: {class_names}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- 3. Load Model ---
    model_path = Path(cfg.checkpoint_path)
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}")
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    logger.info(f"Loading model from {model_path}...")

    # Use model config for initialization
    model = HubertClassifier(
        model_name=cfg.model.pretrained_name,
        num_labels=cfg.model.num_labels,
        freeze_feature_encoder=cfg.model.freeze_feature_encoder,
    )

    # Load weights
    # map_location ensures we can load a GPU model on CPU/MPS if needed
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --- 4. Inference Loop ---
    all_preds = []
    all_labels = []

    logger.info("Running inference...")
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
    logger.info("=" * 30)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 30)

    all_label_ids = list(range(num_labels))

    # Detailed text report
    report = classification_report(
        all_labels,
        all_preds,
        labels=all_label_ids,
        target_names=class_names,
        zero_division=0,
    )
    logger.info(f"\n{report}")
    print(report)

    # Confusion Matrix
    if cfg.evaluation.save_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds, labels=all_label_ids)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - HuBERT Audio Classification")

        save_path = Path(cfg.evaluation.output_dir) / "confusion_matrix.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to: {save_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
