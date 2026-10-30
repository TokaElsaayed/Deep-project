"""Grid search, random search and Optuna tuning for the scratch ResNet.

Run from the repository root:
    python src/training/tune_scratch_resnet.py --method grid --dry-run
    python src/training/tune_scratch_resnet.py --method grid
    python src/training/tune_scratch_resnet.py --method random
    python src/training/tune_scratch_resnet.py --method optuna
    python src/training/tune_scratch_resnet.py --method compare

--method all runs grid, random and optuna in turn and then compares them.

Every trial trains a fresh ScratchResNet for a short, fixed number of epochs and
is scored by its best validation accuracy. The test split is never loaded.
Each finished trial is appended to outputs/hyperparameter_tuning/<method>_trials.csv
straight away, so an interrupted run (e.g. a Colab disconnect) resumes where it
stopped when the same command is run again.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.scratch_resnet import ScratchResNet  # noqa: E402
from training.train_scratch_resnet import (  # noqa: E402
    DEFAULT_SPLITS,
    NUM_CLASSES,
    build_train_and_validation_loaders,
    cutmix,
    evaluate,
    get_device,
    seed_everything,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hyperparameter_tuning.yaml"
SEARCH_METHODS = ("grid", "random", "optuna")
METHOD_LABELS = {"grid": "Grid search", "random": "Random search", "optuna": "Optuna (TPE + pruning)"}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for name, spec in config["search_space"].items():
        if not spec.get("grid"):
            raise ValueError(f"search_space.{name} needs a non-empty grid list")
        if float(spec["low"]) >= float(spec["high"]):
            raise ValueError(f"search_space.{name}: low must be smaller than high")
        if spec.get("log") and float(spec["low"]) <= 0:
            raise ValueError(f"search_space.{name}: log scale needs a positive low bound")
    return config


def grid_configurations(search_space: dict[str, dict[str, Any]]) -> list[dict[str, float]]:
    """Every combination of the grid values, in a fixed order."""
    names = list(search_space)
    value_lists = [[float(value) for value in search_space[name]["grid"]] for name in names]
    return [dict(zip(names, values)) for values in itertools.product(*value_lists)]


def sample_value(spec: dict[str, Any], rng: np.random.Generator) -> float:
    low, high = float(spec["low"]), float(spec["high"])
    if spec.get("log"):
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    return float(rng.uniform(low, high))


def random_configurations(
    search_space: dict[str, dict[str, Any]], n_trials: int, seed: int
) -> list[dict[str, float]]:
    """Independent samples from each range; the same seed gives the same list,
    which is what lets an interrupted random search resume."""
    rng = np.random.default_rng(seed)
    return [
        {name: sample_value(spec, rng) for name, spec in search_space.items()}
        for _ in range(n_trials)
    ]


def trial_budget(config: dict[str, Any], method: str, override: int | None) -> int:
    if override is not None:
        return override
    grid_size = len(grid_configurations(config["search_space"]))
    if method == "grid":
        return grid_size
    configured = config["budget"].get(f"{method}_trials")
    return grid_size if configured is None else int(configured)


def run_trial(
    params: dict[str, float],
    fixed: dict[str, Any],
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    seed: int,
    should_prune: Callable[[int, float], bool] | None = None,
) -> dict[str, Any]:
    """Train one fresh model with Experiment 9's recipe and return its validation scores."""
    seed_everything(seed)
    train_loader.generator.manual_seed(seed)
    model = ScratchResNet(num_classes=NUM_CLASSES, dropout=float(fixed["dropout"])).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(fixed["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["max_lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params["max_lr"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cutmix_probability = float(fixed["cutmix_probability"])

    start = time.perf_counter()
    val_history: list[float] = []
    pruned = False
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if np.random.random() < cutmix_probability:
                    mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
                    logits = model(mixed_images)
                    loss = (
                        lam * criterion(logits, labels_a)
                        + (1.0 - lam) * criterion(logits, labels_b)
                    )
                else:
                    loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        _, val_acc = evaluate(model, val_loader, device, tta=False)
        val_history.append(val_acc)
        print(f"  epoch {epoch:02d}/{epochs} | val acc {val_acc * 100:.2f}%")
        if not math.isfinite(loss.item()):
            print("  Loss is not finite; stopping this trial early.")
            break
        if should_prune is not None and epoch < epochs and should_prune(epoch, val_acc):
            pruned = True
            print("  Pruned by Optuna (below the median of earlier trials).")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    best_epoch = int(np.argmax(val_history)) + 1
    return {
        "best_val_acc": float(max(val_history)),
        "best_epoch": best_epoch,
        "final_val_acc": float(val_history[-1]),
        "epochs_completed": len(val_history),
        "pruned": pruned,
        "seconds": time.perf_counter() - start,
    }


class TrialLog:
    """Append-only CSV of finished trials for one search method."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = (
            pd.read_csv(path).to_dict("records") if path.exists() else []
        )

    def done(self) -> set[int]:
        return {int(row["trial"]) for row in self.rows}

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def print_trial_header(method: str, trial: int, total: int, params: dict[str, float]) -> None:
    settings = ", ".join(f"{name}={value:.4g}" for name, value in params.items())
    print(f"[{method}] trial {trial + 1}/{total}: {settings}")


def out_of_time(context: dict[str, Any]) -> bool:
    """True when another trial (estimated from the longest so far) would overrun --max-hours."""
    deadline = context.get("deadline")
    if deadline is None:
        return False
    if time.monotonic() + context.get("longest_trial_seconds", 0.0) <= deadline:
        return False
    print("Stopping: another trial would exceed --max-hours. Re-run the same command to continue.")
    return True


def record_duration(context: dict[str, Any], seconds: float) -> None:
    context["longest_trial_seconds"] = max(context.get("longest_trial_seconds", 0.0), seconds)


def run_listed_search(
    method: str,
    configurations: list[dict[str, float]],
    context: dict[str, Any],
) -> None:
    """Grid and random search: train each listed configuration that is not yet logged."""
    log = TrialLog(context["output_dir"] / f"{method}_trials.csv")
    for trial, params in enumerate(configurations):
        if trial in log.done():
            continue
        if out_of_time(context):
            return
        print_trial_header(method, trial, len(configurations), params)
        result = run_trial(
            params, context["fixed"], context["train_loader"], context["val_loader"],
            context["device"], context["epochs"], context["seed"],
        )
        record_duration(context, result["seconds"])
        log.append({"method": method, "trial": trial, **params, "epoch_budget": context["epochs"], **result})
        print(f"  best val acc {result['best_val_acc'] * 100:.2f}% in {result['seconds'] / 60:.1f} min")


def run_optuna_search(n_trials: int, context: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for --method optuna: pip install optuna") from exc

    output_dir: Path = context["output_dir"]
    storage = f"sqlite:///{(output_dir / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        study_name="scratch_resnet",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=context["seed"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=int(config["budget"]["optuna_pruner_warmup_epochs"]),
        ),
    )
    log = TrialLog(output_dir / "optuna_trials.csv")
    search_space = config["search_space"]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            name: trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
            for name, spec in search_space.items()
        }
        print_trial_header("optuna", trial.number, n_trials, params)

        def should_prune(epoch: int, val_acc: float) -> bool:
            trial.report(val_acc, epoch)
            return trial.should_prune()

        result = run_trial(
            params, context["fixed"], context["train_loader"], context["val_loader"],
            context["device"], context["epochs"], context["seed"], should_prune,
        )
        record_duration(context, result["seconds"])
        log.append({"method": "optuna", "trial": trial.number, **params, "epoch_budget": context["epochs"], **result})
        print(f"  best val acc {result['best_val_acc'] * 100:.2f}% in {result['seconds'] / 60:.1f} min")
        if result["pruned"]:
            raise optuna.TrialPruned()
        return result["best_val_acc"]

    finished = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    remaining = n_trials - len(finished)

    def stop_if_out_of_time(study: "optuna.Study", _trial: "optuna.trial.FrozenTrial") -> None:
        if out_of_time(context):
            study.stop()

    if remaining > 0 and not out_of_time(context):
        study.optimize(objective, n_trials=remaining, callbacks=[stop_if_out_of_time])


def summarize_method(trials: pd.DataFrame, param_names: list[str]) -> dict[str, Any]:
    trials = trials.sort_values("trial").reset_index(drop=True)
    best_index = int(trials["best_val_acc"].idxmax())
    best = trials.loc[best_index]
    cumulative = trials["seconds"].cumsum()
    return {
        "trials": len(trials),
        "epoch_budget": int(trials["epoch_budget"].max()),
        "pruned_trials": int(trials["pruned"].astype(bool).sum()),
        "epochs_trained": int(trials["epochs_completed"].sum()),
        "best_val_acc": float(best["best_val_acc"]),
        "best_trial": int(best["trial"]),
        "trials_to_best": best_index + 1,
        "minutes_to_best": float(cumulative.iloc[best_index] / 60),
        "total_minutes": float(trials["seconds"].sum() / 60),
        "mean_minutes_per_trial": float(trials["seconds"].mean() / 60),
        **{f"best_{name}": float(best[name]) for name in param_names},
    }


def compare_methods(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    param_names = list(config["search_space"])
    results: dict[str, pd.DataFrame] = {}
    for method in SEARCH_METHODS:
        path = output_dir / f"{method}_trials.csv"
        if path.exists():
            results[method] = pd.read_csv(path)
    if not results:
        raise FileNotFoundError(f"No *_trials.csv files in {output_dir}; run a search first.")

    summary = pd.DataFrame(
        [{"method": method, **summarize_method(trials, param_names)} for method, trials in results.items()]
    )
    summary.to_csv(output_dir / "comparison.csv", index=False)
    plot_comparison(results, param_names, config["search_space"], output_dir)
    write_comparison_markdown(summary, param_names, config, output_dir)

    winner = summary.loc[summary["best_val_acc"].idxmax()]
    best = {
        "selected_by": "highest best_val_acc across all tuning methods (validation split only)",
        "method": winner["method"],
        "best_val_acc": float(winner["best_val_acc"]),
        **{name: float(winner[f"best_{name}"]) for name in param_names},
        "fixed": config["fixed"],
    }
    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print(summary.to_string(index=False))
    print(f"\nSaved comparison to {output_dir}")
    return summary


def plot_comparison(
    results: dict[str, pd.DataFrame],
    param_names: list[str],
    search_space: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, (by_trial, by_time) = plt.subplots(1, 2, figsize=(13, 5))
    for method, trials in results.items():
        trials = trials.sort_values("trial")
        running_best = trials["best_val_acc"].cummax() * 100
        by_trial.step(range(1, len(trials) + 1), running_best, where="post", marker="o", label=METHOD_LABELS[method])
        by_time.step(trials["seconds"].cumsum() / 60, running_best, where="post", marker="o", label=METHOD_LABELS[method])
    by_trial.set(xlabel="Trial", ylabel="Best validation accuracy so far (%)", title="Best score vs. number of trials")
    by_time.set(xlabel="Cumulative tuning time (min)", ylabel="Best validation accuracy so far (%)", title="Best score vs. time")
    for axis in (by_trial, by_time):
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "best_score_comparison.png", dpi=160)
    plt.close(fig)

    if len(param_names) < 2:
        return
    x_name, y_name = param_names[:2]
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), squeeze=False)
    all_scores = pd.concat(results.values())["best_val_acc"] * 100
    for axis, (method, trials) in zip(axes[0], results.items()):
        points = axis.scatter(
            trials[x_name], trials[y_name], c=trials["best_val_acc"] * 100,
            vmin=all_scores.min(), vmax=all_scores.max(), cmap="viridis", s=70, edgecolors="black",
        )
        best = trials.loc[trials["best_val_acc"].idxmax()]
        axis.scatter([best[x_name]], [best[y_name]], marker="*", s=300, c="red", label="best")
        # Same axes in every panel: the whole search box, ticked at the grid values.
        for name, set_scale, set_limits, set_ticks, ticks_axis in (
            (x_name, axis.set_xscale, axis.set_xlim, axis.set_xticks, axis.xaxis),
            (y_name, axis.set_yscale, axis.set_ylim, axis.set_yticks, axis.yaxis),
        ):
            spec = search_space[name]
            if spec.get("log"):
                set_scale("log")
            ticks_axis.set_minor_locator(matplotlib.ticker.NullLocator())
            margin = 1.15 if spec.get("log") else 1.0
            set_limits(float(spec["low"]) / margin, float(spec["high"]) * margin)
            grid_values = [float(value) for value in spec["grid"]]
            set_ticks(grid_values, [f"{value:g}" for value in grid_values])
        axis.set(xlabel=x_name, ylabel=y_name, title=METHOD_LABELS[method])
        axis.legend(loc="lower right")
    fig.subplots_adjust(wspace=0.35)
    fig.colorbar(points, ax=axes[0].tolist(), label="Best validation accuracy (%)")
    fig.savefig(output_dir / "search_space_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_comparison_markdown(
    summary: pd.DataFrame, param_names: list[str], config: dict[str, Any], output_dir: Path
) -> None:
    rows = [
        "| Method | Trials | Pruned | Epochs trained | Best val acc | Best " + " | Best ".join(param_names)
        + " | Trials to best | Time to best (min) | Total time (min) |",
        "|" + "---|" * (8 + len(param_names)),
    ]
    for record in summary.to_dict("records"):
        params = " | ".join(f"{record[f'best_{name}']:.4g}" for name in param_names)
        rows.append(
            f"| {METHOD_LABELS[record['method']]} | {record['trials']} | {record['pruned_trials']} | "
            f"{record['epochs_trained']} | {record['best_val_acc'] * 100:.2f}% | {params} | "
            f"{record['trials_to_best']} | {record['minutes_to_best']:.1f} | {record['total_minutes']:.1f} |"
        )
    text = (
        "# Hyperparameter tuning comparison (scratch ResNet)\n\n"
        f"Each trial trains for up to {int(summary['epoch_budget'].max())} epochs with seed {config['seed']} "
        "and is scored by its best validation accuracy. The test split is not used.\n\n"
        + "\n".join(rows)
        + "\n\nPlots: `best_score_comparison.png`, `search_space_coverage.png`. "
        "Selected settings: `best_hyperparameters.json`.\n"
    )
    (output_dir / "comparison.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=(*SEARCH_METHODS, "all", "compare"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=None, help="Overrides output_dir in the config.")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides budget.epochs_per_trial.")
    parser.add_argument("--trials", type=int, default=None, help="Trial count for random/optuna.")
    parser.add_argument(
        "--max-hours", type=float, default=None,
        help="Stop starting new trials so the run ends within this many hours (e.g. Kaggle's 12-hour limit).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List the trial settings without training.")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.trials is not None and args.trials <= 0:
        parser.error("--trials must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "compare":
        compare_methods(config, output_dir)
        return

    seed = int(config["seed"])
    methods = SEARCH_METHODS if args.method == "all" else (args.method,)
    search_space = config["search_space"]
    listed = {
        "grid": lambda: grid_configurations(search_space),
        "random": lambda: random_configurations(search_space, trial_budget(config, "random", args.trials), seed),
    }

    if args.dry_run:
        for method in methods:
            if method == "optuna":
                print(f"[optuna] {trial_budget(config, 'optuna', args.trials)} trials chosen adaptively by TPE")
                continue
            configurations = listed[method]()
            for trial, params in enumerate(configurations):
                print_trial_header(method, trial, len(configurations), params)
        return

    start = time.monotonic()
    device = get_device()
    fixed = config["fixed"]
    train_loader, val_loader, sizes = build_train_and_validation_loaders(
        args.split_csv, int(fixed["batch_size"]), seed, device
    )
    epochs = args.epochs or int(config["budget"]["epochs_per_trial"])
    print(f"Device: {device} | train {sizes['train']} | validation {sizes['validation']} | {epochs} epochs per trial")
    context = {
        "fixed": fixed,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device,
        "epochs": epochs,
        "seed": seed,
        "output_dir": output_dir,
        "deadline": None if args.max_hours is None else start + args.max_hours * 3600,
    }
    for method in methods:
        if method == "optuna":
            run_optuna_search(trial_budget(config, "optuna", args.trials), context, config)
        else:
            run_listed_search(method, listed[method](), context)
    compare_methods(config, output_dir)


if __name__ == "__main__":
    main()
