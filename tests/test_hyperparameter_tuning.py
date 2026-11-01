import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.tune_scratch_resnet import (
    DEFAULT_CONFIG,
    compare_methods,
    grid_configurations,
    load_config,
    out_of_time,
    random_configurations,
    run_trial,
    select_final_candidates,
    summarize_final,
    trial_budget,
)


def write_trials(folder: Path, method: str, rows: list[tuple[float, float, float]]) -> None:
    pd.DataFrame(
        [{"trial": i, "max_lr": lr, "weight_decay": wd, "best_val_acc": acc} for i, (lr, wd, acc) in enumerate(rows)]
    ).to_csv(folder / f"{method}_trials.csv", index=False)


class HyperparameterTuningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)
        self.space = self.config["search_space"]

    def test_grid_covers_every_combination(self):
        grid = grid_configurations(self.space)
        expected = 1
        for spec in self.space.values():
            expected *= len(spec["grid"])
        self.assertEqual(len(grid), expected)
        self.assertEqual(len({tuple(sorted(params.items())) for params in grid}), expected)

    def test_random_search_is_reproducible_and_in_range(self):
        first = random_configurations(self.space, 20, seed=42)
        self.assertEqual(first, random_configurations(self.space, 20, seed=42))
        self.assertNotEqual(first, random_configurations(self.space, 20, seed=7))
        for params in first:
            for name, spec in self.space.items():
                self.assertGreaterEqual(params[name], spec["low"])
                self.assertLessEqual(params[name], spec["high"])

    def test_default_budgets_match_grid_size(self):
        grid_size = len(grid_configurations(self.space))
        self.assertEqual(trial_budget(self.config, "random", None), grid_size)
        self.assertEqual(trial_budget(self.config, "optuna", None), grid_size)
        self.assertEqual(trial_budget(self.config, "random", 4), 4)

    def test_time_limit_counts_the_next_trial(self):
        now = time.monotonic()
        self.assertFalse(out_of_time({"deadline": None}))
        self.assertFalse(out_of_time({"deadline": now + 100, "longest_trial_seconds": 50}))
        self.assertTrue(out_of_time({"deadline": now + 100, "longest_trial_seconds": 500}))

    def test_run_trial_on_tiny_data(self):
        generator = torch.Generator()
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 200, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=True, generator=generator)
        result = run_trial(
            {"max_lr": 1e-3, "weight_decay": 0.01},
            self.config["fixed"],
            loader,
            loader,
            torch.device("cpu"),
            epochs=2,
            seed=0,
            should_prune=lambda epoch, val_acc: True,
        )
        self.assertTrue(result["pruned"])
        self.assertEqual(result["epochs_completed"], 1)
        self.assertTrue(0.0 <= result["best_val_acc"] <= 1.0)

    def test_compare_methods_writes_reports(self):
        rows = []
        for method, scores in {"grid": [0.20, 0.30, 0.25], "random": [0.28, 0.22, 0.31]}.items():
            for trial, score in enumerate(scores):
                rows.append({
                    "method": method, "trial": trial, "max_lr": 1e-3 * (trial + 1),
                    "weight_decay": 0.01 * (trial + 1), "epoch_budget": 10, "best_val_acc": score,
                    "best_epoch": 9, "final_val_acc": score, "epochs_completed": 10,
                    "pruned": False, "seconds": 60.0,
                })
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            frame = pd.DataFrame(rows)
            for method in ("grid", "random"):
                frame[frame["method"] == method].to_csv(output_dir / f"{method}_trials.csv", index=False)
            summary = compare_methods(self.config, output_dir).set_index("method")
            self.assertAlmostEqual(summary.loc["grid", "best_val_acc"], 0.30)
            self.assertEqual(summary.loc["grid", "trials_to_best"], 2)
            self.assertEqual(summary.loc["random", "trials_to_best"], 3)
            for name in ("comparison.csv", "comparison.md", "best_hyperparameters.json",
                         "best_score_comparison.png", "search_space_coverage.png"):
                self.assertTrue((output_dir / name).exists(), name)

    def test_final_candidates_are_each_methods_best_plus_baseline(self):
        baseline = self.config["final"]["baseline"]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_trials(folder, "grid", [(0.001, 0.03, 0.0857), (0.004, 0.1, 0.05)])
            write_trials(folder, "random", [(0.0012, 0.028, 0.0823)])
            candidates = {c["name"]: c for c in select_final_candidates(self.config, folder)}
            self.assertEqual(list(candidates), ["baseline", "grid", "random"])
            self.assertEqual(candidates["grid"]["max_lr"], 0.001)
            self.assertAlmostEqual(candidates["grid"]["short_run_val_acc"], 0.0857)
            self.assertEqual(candidates["baseline"]["max_lr"], baseline["max_lr"])
            without_baseline = select_final_candidates(self.config, folder, include_baseline=False)
            self.assertEqual([c["name"] for c in without_baseline], ["grid", "random"])

    def test_final_trains_identical_settings_once(self):
        baseline = self.config["final"]["baseline"]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_trials(folder, "grid", [(baseline["max_lr"], baseline["weight_decay"], 0.09)])
            write_trials(folder, "random", [(0.0012, 0.028, 0.08)])
            names = [c["name"] for c in select_final_candidates(self.config, folder)]
            self.assertEqual(names, ["baseline", "random"])

    def test_final_needs_tuning_results(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                select_final_candidates(self.config, Path(folder))

    def test_final_summary_selects_on_validation_not_test(self):
        candidates = [
            {"name": "baseline", "max_lr": 0.002, "weight_decay": 0.03, "short_run_val_acc": None},
            {"name": "optuna", "max_lr": 0.0017, "weight_decay": 0.089, "short_run_val_acc": 0.089},
        ]
        scores = {"baseline": (0.53, 0.58), "optuna": (0.55, 0.56)}
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            for name, (val, test) in scores.items():
                run_dir = output_dir / name
                run_dir.mkdir()
                pd.DataFrame({"epoch": [1, 2], "val_acc": [0.1, val]}).to_csv(run_dir / "history.csv", index=False)
                (run_dir / "evaluation_results.json").write_text(json.dumps({
                    "standard_test_accuracy": test, "horizontal_flip_tta_test_accuracy": test + 0.005,
                }))
            summary = summarize_final(candidates, output_dir).set_index("candidate")
            self.assertTrue(summary.loc["optuna", "selected"])
            self.assertFalse(summary.loc["baseline", "selected"])
            self.assertEqual(summary.loc["optuna", "best_epoch"], 2)
            for name in ("final_comparison.csv", "final_comparison.md", "final_validation_curves.png"):
                self.assertTrue((output_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
