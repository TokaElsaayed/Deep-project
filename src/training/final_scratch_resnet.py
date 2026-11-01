"""Retrain the tuned scratch ResNet settings for the full schedule and compare them.

Run from the repository root after tune_scratch_resnet.py has produced
<tuning-dir>/<method>_trials.csv files:
    python src/training/final_scratch_resnet.py --dry-run
    python src/training/final_scratch_resnet.py

Candidates are the best setting found by each tuning method plus the original
Experiment 9 setting ("baseline"). Each is trained for the same number of epochs
with the same seed and recipe (train_scratch_resnet.py), so only max_lr and
weight_decay differ. The winner is chosen on validation accuracy; every
candidate's best checkpoint is then evaluated on the test split for reporting.
Finished candidates are skipped when the command is run again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.train_scratch_resnet import (  # noqa: E402
    DEFAULT_SPLITS,
    evaluate_test,
    get_device,
    train,
)
from training.tune_scratch_resnet import METHOD_LABELS, SEARCH_METHODS  # noqa: E402

DEFAULT_TUNING_DIR = PROJECT_ROOT / "outputs" / "hyperparameter_tuning"
# Experiment 9's settings (outputs/baseline/README.md).
BASELINE = {"max_lr": 0.002, "weight_decay": 0.03}
CANDIDATE_LABELS = {"baseline": "Baseline (Experiment 9 settings)", **METHOD_LABELS}


def select_candidates(tuning_dir: Path, include_baseline: bool = True) -> list[dict[str, Any]]:
    """The best trial of each tuning method, plus the baseline, without duplicate settings."""
    candidates: list[dict[str, Any]] = []
    if include_baseline:
        candidates.append({"name": "baseline", **BASELINE, "short_run_val_acc": None})
    for method in SEARCH_METHODS:
        path = tuning_dir / f"{method}_trials.csv"
        if not path.exists():
            continue
        trials = pd.read_csv(path)
        best = trials.loc[trials["best_val_acc"].idxmax()]
        candidates.append({
            "name": method,
            "max_lr": float(best["max_lr"]),
            "weight_decay": float(best["weight_decay"]),
            "short_run_val_acc": float(best["best_val_acc"]),
        })

    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        duplicate = next(
            (kept for kept in unique
             if abs(kept["max_lr"] - candidate["max_lr"]) < 1e-12
             and abs(kept["weight_decay"] - candidate["weight_decay"]) < 1e-12),
            None,
        )
        if duplicate is None:
            unique.append(candidate)
        else:
            print(f"{candidate['name']} has the same settings as {duplicate['name']}; training it once.")
    if not any(candidate["name"] != "baseline" for candidate in unique):
        raise FileNotFoundError(f"No *_trials.csv files in {tuning_dir}; run tune_scratch_resnet.py first.")
    return unique


def train_and_evaluate(candidate: dict[str, Any], args: argparse.Namespace, device) -> None:
    run_dir = args.output_dir / candidate["name"]
    if (run_dir / "evaluation_results.json").exists():
        print(f"[{candidate['name']}] already finished; skipping.")
        return
    print(
        f"\n[{candidate['name']}] training {args.epochs} epochs with "
        f"max_lr={candidate['max_lr']:.4g}, weight_decay={candidate['weight_decay']:.4g}"
    )
    start = time.perf_counter()
    train(
        argparse.Namespace(
            split_csv=args.split_csv,
            output_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_lr=candidate["max_lr"],
            weight_decay=candidate["weight_decay"],
            label_smoothing=args.label_smoothing,
            cutmix_probability=args.cutmix_probability,
            seed=args.seed,
        ),
        device,
    )
    (run_dir / "training_minutes.txt").write_text(f"{(time.perf_counter() - start) / 60:.2f}\n")
    evaluate_test(
        argparse.Namespace(
            split_csv=args.split_csv,
            checkpoint=run_dir / "best.pth",
            output_dir=run_dir,
            batch_size=args.batch_size,
        ),
        device,
    )


def summarize(candidates: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        run_dir = output_dir / candidate["name"]
        results_path = run_dir / "evaluation_results.json"
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        history = pd.read_csv(run_dir / "history.csv")
        minutes_path = run_dir / "training_minutes.txt"
        rows.append({
            "candidate": candidate["name"],
            "max_lr": candidate["max_lr"],
            "weight_decay": candidate["weight_decay"],
            "short_run_val_acc": candidate["short_run_val_acc"],
            "epochs": len(history),
            "best_val_acc": float(history["val_acc"].max()),
            "best_epoch": int(history.loc[history["val_acc"].idxmax(), "epoch"]),
            "test_acc": results["standard_test_accuracy"],
            "test_acc_tta": results["horizontal_flip_tta_test_accuracy"],
            "training_minutes": float(minutes_path.read_text()) if minutes_path.exists() else None,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["selected"] = summary["best_val_acc"] == summary["best_val_acc"].max()
    summary.to_csv(output_dir / "final_comparison.csv", index=False)
    write_markdown(summary, output_dir)
    plot_curves(summary, output_dir)

    winner = summary.loc[summary["selected"]].iloc[0]
    print(summary.to_string(index=False))
    print(
        f"\nSelected on validation accuracy: {winner['candidate']} "
        f"(val {winner['best_val_acc'] * 100:.2f}%, test {winner['test_acc'] * 100:.2f}%)"
    )
    return summary


def write_markdown(summary: pd.DataFrame, output_dir: Path) -> None:
    def percent(value) -> str:
        return "—" if value is None or pd.isna(value) else f"{value * 100:.2f}%"

    lines = [
        "| Settings from | max_lr | weight_decay | 10-epoch val acc | Best val acc | Best epoch | Test acc | Test acc (flip TTA) | Training time (min) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary.to_dict("records"):
        name = CANDIDATE_LABELS[row["candidate"]] + (" **(selected)**" if row["selected"] else "")
        minutes = "—" if row["training_minutes"] is None or pd.isna(row["training_minutes"]) else f"{row['training_minutes']:.0f}"
        lines.append(
            f"| {name} | {row['max_lr']:.4g} | {row['weight_decay']:.4g} | {percent(row['short_run_val_acc'])} | "
            f"{percent(row['best_val_acc'])} | {row['best_epoch']} | {percent(row['test_acc'])} | "
            f"{percent(row['test_acc_tta'])} | {minutes} |"
        )
    epochs = int(summary["epochs"].max())
    text = (
        "# Final training: tuned settings vs. baseline (scratch ResNet)\n\n"
        f"Every model was trained for {epochs} epochs with the same seed, data and recipe; only max_lr and "
        "weight_decay differ. The selected model has the highest validation accuracy. The test split was used "
        "only to report results, not to choose between models.\n\n"
        + "\n".join(lines)
        + "\n\nThe recorded Experiment 9 run reached 53.39% validation and 56.04% test accuracy "
        "(`outputs/baseline/README.md`). The baseline row above retrains those settings on the same hardware as "
        "the tuned runs.\n\nPlot: `final_validation_curves.png`.\n"
    )
    (output_dir / "final_comparison.md").write_text(text, encoding="utf-8")


def plot_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in summary["candidate"]:
        history = pd.read_csv(output_dir / name / "history.csv")
        ax.plot(history["epoch"], history["val_acc"] * 100, label=CANDIDATE_LABELS[name])
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Full training: tuned settings vs. baseline")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_validation_curves.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <tuning-dir>/final.")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-baseline", action="store_true", help="Skip retraining the Experiment 9 settings.")
    parser.add_argument("--dry-run", action="store_true", help="List the candidates without training.")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    args.output_dir = args.output_dir or args.tuning_dir / "final"
    return args


def main() -> None:
    args = parse_args()
    candidates = select_candidates(args.tuning_dir, include_baseline=not args.no_baseline)
    for candidate in candidates:
        short = candidate["short_run_val_acc"]
        print(
            f"{candidate['name']:>8}: max_lr={candidate['max_lr']:.4g}, weight_decay={candidate['weight_decay']:.4g}"
            + ("" if short is None else f" (10-epoch val acc {short * 100:.2f}%)")
        )
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    for candidate in candidates:
        train_and_evaluate(candidate, args, device)
    summarize(candidates, args.output_dir)


if __name__ == "__main__":
    main()
