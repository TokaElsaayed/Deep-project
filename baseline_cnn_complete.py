"""
Complete baseline CNN experiment for CUB-200-2011 (Person 1).

Run from the root of the Deep-project repository:
    python baseline_cnn_complete.py

This script includes the dataset split wrapping, CNN architecture, training,
validation, checkpoint saving, test evaluation, and training-curve plotting.
It reuses the repository's existing dataset loader and transforms:
    src.data.eda
    src.data.transforms
and reads:
    configs/eda.yaml
    outputs/eda/splits.csv

The CUB dataset itself is not embedded in this file; it must be available
where the project's existing data loader expects it.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.data.eda import _image_object, load_cub_datasets
from src.data.transforms import (
    build_evaluation_transform,
    build_train_transform,
)


# ----------------------------
# Configuration
# ----------------------------
SEED = 42
NUM_CLASSES = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
CONFIG_PATH = Path("configs/eda.yaml")
SPLITS_PATH = Path("outputs/eda/splits.csv")
OUTPUT_DIR = Path("outputs/baseline")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.pt"
PLOT_PATH = OUTPUT_DIR / "training_curves.png"


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Model
# ----------------------------
class BaselineCNN(nn.Module):
    """Three-block CNN baseline for 200 bird species."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ----------------------------
# Dataset wrapper
# ----------------------------
class BirdClassificationDataset(Dataset):
    """Apply project transforms to rows from the saved split CSV."""

    def __init__(self, source_dataset, rows: pd.DataFrame, transform) -> None:
        self.source_dataset = source_dataset
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image_index = int(row["index"])
        record = self.source_dataset[image_index]
        image = _image_object(record["image"]).convert("RGB")
        image = self.transform(image)
        label = int(record["label"])
        return image, label


# ----------------------------
# Training and evaluation
# ----------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The data loader is empty; check the split CSV and dataset.")

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for required_path in (CONFIG_PATH, SPLITS_PATH):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}. "
                "Run this script from the repository root and check the EDA setup."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_source, test_source = load_cub_datasets(config)
    splits = pd.read_csv(SPLITS_PATH)

    required_columns = {"partition", "index", "label"}
    missing_columns = required_columns - set(splits.columns)
    if missing_columns:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing_columns)}")

    train_rows = splits[splits["partition"] == "train"]
    val_rows = splits[splits["partition"] == "validation"]
    test_rows = splits[splits["partition"] == "test"]

    train_transform = build_train_transform(config, strength="light", seed=SEED)
    eval_transform = build_evaluation_transform(config)

    train_data = BirdClassificationDataset(train_source, train_rows, train_transform)
    val_data = BirdClassificationDataset(train_source, val_rows, eval_transform)
    test_data = BirdClassificationDataset(test_source, test_rows, eval_transform)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(
        f"Dataset sizes — train: {len(train_data)}, "
        f"validation: {len(val_data)}, test: {len(test_data)}"
    )

    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%} | "
            f"val loss: {val_loss:.4f} | "
            f"val accuracy: {val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    torch.save(history, HISTORY_PATH)

    # Evaluate once on the test set using the best validation checkpoint.
    test_loss, test_accuracy = run_epoch(
        model, test_loader, criterion, device
    )
    print("\nFinal results using best validation checkpoint")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Test images evaluated: {len(test_data)}")
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved history: {HISTORY_PATH}")

    # Plot loss and accuracy.
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_title("Baseline CNN — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        epochs, [value * 100 for value in history["train_accuracy"]],
        label="Training accuracy",
    )
    axes[1].plot(
        epochs, [value * 100 for value in history["val_accuracy"]],
        label="Validation accuracy",
    )
    axes[1].set_title("Baseline CNN — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved training curves: {PLOT_PATH}")


if __name__ == "__main__":
    main()
