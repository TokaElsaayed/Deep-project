"""Train or evaluate the scratch-trained ResNet baseline for CUB-200.

Run from the repository root:
    python src/training/train_scratch_resnet.py --help
    python src/training/train_scratch_resnet.py --mode smoke-test
    python src/training/train_scratch_resnet.py --mode train --epochs 100
    python src/training/train_scratch_resnet.py --mode evaluate --checkpoint outputs/baseline/best.pth

Training uses only the train and validation splits. The test split is loaded only
when --mode evaluate is selected, to keep test data out of model selection.
New training runs default to outputs/baseline_reproduced so the recorded
Experiment 9 checkpoint in outputs/baseline is not accidentally overwritten.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Use the one canonical model implementation. This architecture is compatible
# with the committed Experiment 9 checkpoint.
from models.scratch_resnet import ScratchResNet  # noqa: E402

DEFAULT_SPLITS = PROJECT_ROOT / "outputs" / "eda" / "splits.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "baseline" / "best.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_reproduced"

NUM_CLASSES = 200
IMAGE_SIZE = 128
CACHE_SIZE = 144
DEFAULT_MEAN = [0.484575231096779, 0.4990690590550937, 0.4326982401975864]
DEFAULT_STD = [0.2320364540940151, 0.22752646981271832, 0.2660812817755806]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits(split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}\n"
            "Run the project's data preparation/EDA step so outputs/eda/splits.csv exists."
        )
    splits = pd.read_csv(split_csv)
    required = {"partition", "index", "file_name", "label"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")

    train_rows = splits[splits["partition"] == "train"].copy().reset_index(drop=True)
    val_rows = splits[splits["partition"] == "validation"].copy().reset_index(drop=True)
    test_rows = splits[splits["partition"] == "test"].copy().reset_index(drop=True)
    if train_rows.empty or val_rows.empty or test_rows.empty:
        raise ValueError(
            "The split file must contain non-empty train, validation, and test partitions."
        )
    return train_rows, val_rows, test_rows


def to_pil(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


class CachedBirdDataset(Dataset):
    """PIL images cached at 144x144, with train/evaluation transforms applied per item."""

    def __init__(self, images: list[Image.Image], labels: list[int], transform: T.Compose):
        if len(images) != len(labels):
            raise ValueError("Image and label counts do not match.")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[index]), self.labels[index]


def cache_images(rows: pd.DataFrame, source: Any, description: str) -> tuple[list[Image.Image], list[int]]:
    """Cache a selected split from a Hugging Face dataset, matching Experiment 9 preprocessing."""
    cache_transform = T.Compose([T.Resize(CACHE_SIZE), T.CenterCrop(CACHE_SIZE)])
    images: list[Image.Image] = []
    labels: list[int] = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=description):
        item = source[int(row["index"])]
        images.append(cache_transform(to_pil(item["image"])))
        labels.append(int(row["label"]))
    return images, labels


def make_transforms() -> tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    train_transform = T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.25),
    ])
    eval_transform = T.Compose([
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def load_hf_split(dataset_name: str, split_name: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face 'datasets' package is required. Install project dependencies "
            "before running training/evaluation."
        ) from exc
    print(f"Loading dataset {dataset_name!r}, split={split_name!r} ...")
    return load_dataset(dataset_name, split=split_name)


def build_train_and_validation_loaders(
    split_csv: Path, batch_size: int, seed: int, device: torch.device
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_rows, val_rows, _ = load_splits(split_csv)
    source = load_hf_split("Multimodal-Fatima/CUB_train", "train")
    train_images, train_labels = cache_images(train_rows, source, "Caching train images")
    val_images, val_labels = cache_images(val_rows, source, "Caching validation images")
    train_transform, eval_transform = make_transforms()
    train_dataset = CachedBirdDataset(train_images, train_labels, train_transform)
    val_dataset = CachedBirdDataset(val_images, val_labels, eval_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; reduce --batch-size or check the train split.")
    return train_loader, val_loader, {"train": len(train_dataset), "validation": len(val_dataset)}


def build_test_loader(split_csv: Path, batch_size: int, device: torch.device) -> tuple[DataLoader, int]:
    _, _, test_rows = load_splits(split_csv)
    source = load_hf_split("Multimodal-Fatima/CUB_test", "test")
    test_images, test_labels = cache_images(test_rows, source, "Caching test images")
    _, eval_transform = make_transforms()
    test_dataset = CachedBirdDataset(test_images, test_labels, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return test_loader, len(test_dataset)


def cutmix(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    height, width = images.shape[2:]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y, center_x = np.random.randint(height), np.random.randint(width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed_images, labels, labels[permutation], float(lam)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if tta:
                flip_logits = model(torch.flip(images, dims=[3]))
                logits = (logits + flip_logits) / 2
            loss = nn.functional.cross_entropy(logits, labels)
        count = labels.size(0)
        loss_sum += float(loss.item()) * count
        correct += int((logits.argmax(1) == labels).sum().item())
        total += count
    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    return loss_sum / total, correct / total


def plot_history(history: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Scratch ResNet Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_acc"] * 100, label="Training accuracy")
    ax.plot(history["epoch"], history["val_acc"] * 100, label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Scratch ResNet Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curves.png", dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace, device: torch.device) -> None:
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, split_sizes = build_train_and_validation_loaders(
        args.split_csv, args.batch_size, args.seed, device
    )
    model = ScratchResNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    checkpoint_path = args.output_dir / "best.pth"
    history_path = args.output_dir / "history.csv"

    config = {
        "architecture": "ScratchResNet",
        "num_classes": NUM_CLASSES,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "cutmix_probability": args.cutmix_probability,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "train_images": split_sizes["train"],
        "validation_images": split_sizes["validation"],
        "selection_metric": "validation_accuracy",
        "test_split_used_during_training": False,
        "device": str(device),
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Device: {device}")
    print(f"Training images: {split_sizes['train']} | validation images: {split_sizes['validation']}")
    print(f"Saving this run to: {args.output_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if np.random.random() < args.cutmix_probability:
                    mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
                    logits = model(mixed_images)
                    loss = (
                        lam * criterion(logits, labels_a)
                        + (1.0 - lam) * criterion(logits, labels_b)
                    )
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_count = labels.size(0)
            loss_sum += float(loss.item()) * batch_count
            # Preserve the original experiment's reporting convention: for CutMix
            # batches, training accuracy is measured against the original labels.
            correct += int((logits.argmax(1) == labels).sum().item())
            total += batch_count

        val_loss, val_acc = evaluate(model, val_loader, device, tta=False)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "train_acc": correct / max(total, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)

        print(
            f"Epoch {epoch:03d} | train loss {record['train_loss']:.4f} | "
            f"train acc {record['train_acc'] * 100:.2f}% | "
            f"val loss {val_loss:.4f} | val acc {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_accuracy": float(val_acc),
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  Saved new best checkpoint (validation accuracy {val_acc * 100:.2f}%).")

    plot_history(pd.DataFrame(history), args.output_dir)
    print(f"Training finished. Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best checkpoint: {checkpoint_path}")
    print("The test images were not loaded or evaluated during training.")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Provide a valid --checkpoint path or train a model first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model = ScratchResNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with torch.no_grad():
        sample = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        logits = model(sample)
    if tuple(logits.shape) != (2, NUM_CLASSES):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
    print("Checkpoint loaded successfully.")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    val_acc = checkpoint.get("best_val_accuracy")
    if val_acc is not None:
        print(f"Saved best validation accuracy: {float(val_acc) * 100:.2f}%")
    print(f"Smoke-test output shape: {tuple(logits.shape)}")
    print("No training or test-set evaluation was performed.")


def evaluate_test(args: argparse.Namespace, device: torch.device) -> None:
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    test_loader, test_count = build_test_loader(args.split_csv, args.batch_size, device)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    saved_val = checkpoint.get("best_val_accuracy")
    if saved_val is not None:
        print(f"Saved best validation accuracy: {float(saved_val) * 100:.2f}%")

    standard_loss, standard_acc = evaluate(model, test_loader, device, tta=False)
    print(f"Standard test loss: {standard_loss:.4f}")
    print(f"Standard test accuracy: {standard_acc * 100:.2f}%")
    tta_loss, tta_acc = evaluate(model, test_loader, device, tta=True)
    print(f"Horizontal-flip TTA test loss: {tta_loss:.4f}")
    print(f"Horizontal-flip TTA test accuracy: {tta_acc * 100:.2f}%")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_accuracy": float(saved_val) if saved_val is not None else None,
        "test_images": test_count,
        "standard_test_loss": standard_loss,
        "standard_test_accuracy": standard_acc,
        "horizontal_flip_tta_test_loss": tta_loss,
        "horizontal_flip_tta_test_accuracy": tta_acc,
        "note": "Test results are for final evaluation only, not hyperparameter selection.",
    }
    results_path = args.output_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Evaluation summary saved to: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke-test", "train", "evaluate"),
        default="smoke-test",
        help="Default smoke-test safely validates the existing checkpoint without training.",
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder. Training defaults to baseline_reproduced to protect recorded results.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.cutmix_probability <= 1.0:
        parser.error("--cutmix-probability must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    device = get_device()
    if args.mode == "smoke-test":
        smoke_test(args, device)
    elif args.mode == "train":
        train(args, device)
    elif args.mode == "evaluate":
        evaluate_test(args, device)


if __name__ == "__main__":
    main()
