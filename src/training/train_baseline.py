import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
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
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_baseline(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    learning_rate=0.001,
    checkpoint_path="outputs/baseline/best_model.pth",
):
    set_seed(42)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_accuracy = -1.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.2%}",
            flush=True,
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_accuracy,
                },
                checkpoint,
            )

    torch.save(history, checkpoint.parent / "training_history.pt")
    model.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Saved checkpoint: {checkpoint}")

    return model, history
