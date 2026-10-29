"""Load and evaluate the saved scratch ResNet checkpoint.

This script does not train automatically. It verifies that the saved
checkpoint can be loaded and that the model produces valid logits.

Run from the repository root:
    python src/training/train_scratch_resnet.py
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "baseline" / "best.pth"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from models.scratch_resnet import ScratchResNet


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Place the saved best.pth file in outputs/baseline/."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = ScratchResNet(num_classes=200)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # A small forward-pass smoke test; no training occurs.
    with torch.no_grad():
        sample = torch.randn(2, 3, 128, 128)
        logits = model(sample)

    assert logits.shape == (2, 200), (
        f"Unexpected output shape: {tuple(logits.shape)}"
    )

    print("Checkpoint loaded successfully.")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.2%}"
    )
    print(f"Smoke-test output shape: {tuple(logits.shape)}")
    print("No training was performed.")


if __name__ == "__main__":
    main()
