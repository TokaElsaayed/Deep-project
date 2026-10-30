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
    trial_budget,
)


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


if __name__ == "__main__":
    unittest.main()
