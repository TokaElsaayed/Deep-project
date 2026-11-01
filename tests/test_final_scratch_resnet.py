import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.final_scratch_resnet import BASELINE, select_candidates, summarize


def write_trials(folder: Path, method: str, rows: list[tuple[float, float, float]]) -> None:
    pd.DataFrame(
        [{"trial": i, "max_lr": lr, "weight_decay": wd, "best_val_acc": acc} for i, (lr, wd, acc) in enumerate(rows)]
    ).to_csv(folder / f"{method}_trials.csv", index=False)


class FinalTrainingTests(unittest.TestCase):
    def test_candidates_are_each_methods_best_plus_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_trials(folder, "grid", [(0.001, 0.03, 0.0857), (0.004, 0.1, 0.05)])
            write_trials(folder, "random", [(0.0012, 0.028, 0.0823)])
            candidates = {c["name"]: c for c in select_candidates(folder)}
            self.assertEqual(list(candidates), ["baseline", "grid", "random"])
            self.assertEqual(candidates["grid"]["max_lr"], 0.001)
            self.assertAlmostEqual(candidates["grid"]["short_run_val_acc"], 0.0857)
            self.assertEqual(candidates["baseline"]["max_lr"], BASELINE["max_lr"])
            self.assertEqual([c["name"] for c in select_candidates(folder, include_baseline=False)], ["grid", "random"])

    def test_identical_settings_are_trained_once(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_trials(folder, "grid", [(BASELINE["max_lr"], BASELINE["weight_decay"], 0.09)])
            write_trials(folder, "random", [(0.0012, 0.028, 0.08)])
            self.assertEqual([c["name"] for c in select_candidates(folder)], ["baseline", "random"])

    def test_missing_tuning_results_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                select_candidates(Path(folder))

    def test_summary_selects_on_validation_not_test(self):
        candidates = [
            {"name": "baseline", **BASELINE, "short_run_val_acc": None},
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
            summary = summarize(candidates, output_dir).set_index("candidate")
            self.assertTrue(summary.loc["optuna", "selected"])
            self.assertFalse(summary.loc["baseline", "selected"])
            self.assertEqual(summary.loc["optuna", "best_epoch"], 2)
            for name in ("final_comparison.csv", "final_comparison.md", "final_validation_curves.png"):
                self.assertTrue((output_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
