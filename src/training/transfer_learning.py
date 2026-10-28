"""Train, evaluate, and report torchvision transfer-learning experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.data.transfer_dataset import load_transfer_splits
from src.models.transfer_learning import MODEL_SPECS, configure_trainable_layers, create_transfer_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
    return device


def _load_config(path: Path):
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config, path.parent.parent


def _loader(dataset, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    loss_sum = correct = count = 0
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if count == 0:
        raise ValueError("Training loader produced no samples.")
    return loss_sum / count, correct / count


@torch.inference_mode()
def _evaluate_epoch(model, loader, criterion, device, use_amp=False, timed=False):
    model.eval()
    loss_sum = correct = count = 0
    ids, truths, predictions, confidences = [], [], [], []
    start = time.perf_counter()
    for images, labels, sample_ids in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probabilities = logits.float().softmax(1)
        predicted = probabilities.argmax(1)
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (predicted == labels).sum().item()
        truths.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        confidences.extend(probabilities.max(1).values.cpu().tolist())
        ids.extend(sample_ids)
    if device.type == "cuda" and timed:
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - start
    if count == 0:
        raise ValueError("Evaluation loader produced no samples.")
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "ids": ids,
        "truths": truths,
        "predictions": predictions,
        "confidences": confidences,
        "duration_seconds": duration,
        "samples": count,
    }


def _make_plot(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(args) -> None:
    config, root = _load_config(args.config)
    set_seed(int(config["seed"]))
    device = _device(args.device)
    model, weights = create_transfer_model(args.model, int(config["dataset"]["num_classes"]))
    configure_trainable_layers(model, args.model, args.stage, config["training"]["fine_tune_backbone_blocks"])
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = root / init_path
        state = torch.load(init_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    data, _ = load_transfer_splits(config, root, weights)
    section = config["training"][args.stage]
    epochs = int(args.epochs or section["epochs"])
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    workers = int(config["training"]["workers"])
    train_loader = _loader(data["train"], batch_size, workers, True)
    validation_loader = _loader(data["validation"], batch_size, workers, False)
    model = model.to(device)
    groups = []
    head_parameters = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("fc.") or n.startswith("classifier."))]
    backbone_parameters = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("fc.") or n.startswith("classifier."))]
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(section["head_learning_rate"])})
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": float(section.get("backbone_learning_rate", section["head_learning_rate"]))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(section["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    out_dir = root / config["outputs_dir"] / args.model / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_checkpoint.pth"
    history_rows = []
    best_accuracy, best_loss, stale_epochs = -1.0, float("inf"), 0
    best_state = None
    stage_seconds = 0.0
    previous_seconds = 0.0
    if args.init_checkpoint:
        previous_seconds = float(state.get("cumulative_training_seconds", 0.0))

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        epoch_lrs = [group["lr"] for group in optimizer.param_groups]
        try:
            train_loss, train_accuracy = _train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
            validation = _evaluate_epoch(model, validation_loader, criterion, device, use_amp)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"{args.model} ran out of memory at batch_size={batch_size}. "
                    "Retry with --batch-size 16 (or 8); resolution is fixed by the pretrained weights."
                ) from exc
            raise
        epoch_seconds = time.perf_counter() - start
        stage_seconds += epoch_seconds
        scheduler.step(validation["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation["accuracy"],
            "head_learning_rate": epoch_lrs[0],
            "backbone_learning_rate": epoch_lrs[-1],
            "epoch_seconds": epoch_seconds,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)
        print(
            f"{args.model} {args.stage} epoch {epoch}/{epochs}: "
            f"train loss {train_loss:.4f}, val loss {validation['loss']:.4f}, "
            f"train acc {train_accuracy:.4f}, val acc {validation['accuracy']:.4f}, "
            f"lr {[group['lr'] for group in optimizer.param_groups]}, {epoch_seconds:.1f}s",
            flush=True,
        )
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss
        )
        if improved:
            best_accuracy, best_loss, stale_epochs = validation["accuracy"], validation["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": args.model,
                    "stage": args.stage,
                    "model_state_dict": best_state,
                    "best_epoch": epoch,
                    "best_validation_accuracy": best_accuracy,
                    "best_validation_loss": best_loss,
                    "stage_training_seconds": stage_seconds,
                    "cumulative_training_seconds": previous_seconds + stage_seconds,
                    "num_classes": int(config["dataset"]["num_classes"]),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        patience = args.patience if args.patience is not None else section.get("patience")
        if patience and stale_epochs >= int(patience):
            print(f"Early stopping after {epoch} epochs without validation improvement.")
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; check the configured epochs and data.")
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_checkpoint["stage_training_seconds"] = stage_seconds
    saved_checkpoint["cumulative_training_seconds"] = previous_seconds + stage_seconds
    torch.save(saved_checkpoint, checkpoint_path)
    _make_plot(pd.DataFrame(history_rows), out_dir / "training_curves.png")
    print(f"Best validation accuracy: {best_accuracy:.4f}; checkpoint: {checkpoint_path}")


def evaluate(args) -> None:
    config, root = _load_config(args.config)
    device = _device(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("model") != args.model:
        raise ValueError(f"Checkpoint is for {saved.get('model')}, not requested {args.model}.")
    model, weights = create_transfer_model(args.model, int(config["dataset"]["num_classes"]), pretrained=False)
    model.load_state_dict(saved["model_state_dict"])
    data, _ = load_transfer_splits(config, root, weights)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    loader = _loader(data["test"], batch_size, int(config["training"]["workers"]), False)
    model = model.to(device)
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    result = _evaluate_epoch(model, loader, nn.CrossEntropyLoss(), device, use_amp, timed=True)
    labels = list(range(int(config["dataset"]["num_classes"])))
    precision, recall, f1, _ = precision_recall_fscore_support(
        result["truths"], result["predictions"], labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(result["truths"], result["predictions"], average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(result["truths"], result["predictions"], average="weighted", zero_division=0)
    out_dir = root / config["outputs_dir"] / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_mapping.json").write_text(
        json.dumps({str(label): label for label in range(int(config["dataset"]["num_classes"]))}, indent=2),
        encoding="utf-8",
    )
    try:
        checkpoint_for_report = str(checkpoint.relative_to(root))
    except ValueError:
        checkpoint_for_report = str(checkpoint)
    summary = {
        "model": args.model,
        "checkpoint": checkpoint_for_report,
        "best_validation_accuracy": saved.get("best_validation_accuracy"),
        "best_validation_loss": saved.get("best_validation_loss"),
        "best_epoch": saved.get("best_epoch"),
        "test_accuracy": accuracy_score(result["truths"], result["predictions"]),
        "test_loss": result["loss"],
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
        "test_samples": result["samples"],
        "total_training_seconds": saved.get("cumulative_training_seconds", saved.get("stage_training_seconds")),
        "test_inference_seconds_end_to_end": result["duration_seconds"],
        "test_inference_seconds_per_image": result["duration_seconds"] / result["samples"],
        "test_timing_method": "wall-clock around the full DataLoader iteration, including preprocessing and transfers; CUDA synchronized at end",
        "input_crop_size": int(weights.transforms().crop_size[0]),
        "averaging_methods": "macro and weighted; per-class scores are one-vs-rest",
    }
    (out_dir / "evaluation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"label": labels, "precision": precision, "recall": recall, "f1": f1}
    ).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(result["truths"], result["predictions"], labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(out_dir / "confusion_matrix.csv", index_label="true_label")
    pd.DataFrame(
        {
            "sample_id": result["ids"],
            "true_label": result["truths"],
            "predicted_label": result["predictions"],
            "confidence": result["confidences"],
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)
    # The scratch baseline was selected by validation accuracy on this same split
    # (see baseline_cnn_complete.py); only its recorded accuracy/loss are available.
    comparison_rows = []
    baseline_readme = root / "outputs" / "baseline" / "README.md"
    if baseline_readme.is_file() and result["samples"] == 5794:
        baseline_text = baseline_readme.read_text(encoding="utf-8")
        if "Test images evaluated: 5,794" in baseline_text:
            comparison_rows.append(
                {
                    "model": "baseline_cnn_scratch",
                    "best_validation_accuracy": 0.1001,
                    "test_accuracy": 0.1001,
                    "test_loss": 4.9442,
                    "macro_f1": np.nan,
                    "results_source": "outputs/baseline/README.md; test split from outputs/eda/splits.csv",
                }
            )
    for metrics_path in (root / config["outputs_dir"]).glob("*/evaluation_metrics.json"):
        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        comparison_rows.append(
            {
                "model": model_metrics["model"],
                "best_validation_accuracy": model_metrics["best_validation_accuracy"],
                "test_accuracy": model_metrics["test_accuracy"],
                "test_loss": model_metrics["test_loss"],
                "macro_f1": model_metrics["macro_f1"],
                "results_source": str(metrics_path.relative_to(root)),
            }
        )
    pd.DataFrame(comparison_rows).drop_duplicates("model", keep="last").to_csv(
        root / config["outputs_dir"] / "comparison.csv", index=False
    )
    summary_path = root / config["outputs_dir"] / "results_summary.md"
    summary_path.write_text(
        "# Transfer-learning results\n\n"
        "See `comparison.csv` and each model's `evaluation_metrics.json`. "
        "Checkpoints are selected by validation accuracy; the held-out test split is evaluated only by the explicit evaluate command. "
        "The scratch baseline row contains only metrics recorded in its README; precision, recall, F1, predictions, and timing are unavailable for that run. "
        "Inference timing is end-to-end across DataLoader iteration, including preprocessing and device transfer.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=Path("configs/transfer_learning.yaml"))
        sub.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
        sub.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit torch device")
        sub.add_argument("--batch-size", type=int)
        if command == "train":
            sub.add_argument("--stage", choices=("feature_extraction", "fine_tuning"), required=True)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--patience", type=int)
            sub.add_argument("--init-checkpoint", help="Optional checkpoint to continue from, e.g. feature-extraction best")
            sub.set_defaults(func=train)
        else:
            sub.add_argument("--checkpoint", required=True)
            sub.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
