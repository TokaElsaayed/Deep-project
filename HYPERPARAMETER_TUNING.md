# Hyperparameter tuning (scratch ResNet)

This step tunes the scratch-trained ResNet from Experiment 9 (`src/models/scratch_resnet.py`) with Grid Search, Random Search and, as the optional extra, Optuna. All three methods are compared on best validation accuracy, number of trials and time taken. The script reuses the data loading, augmentation, CutMix and evaluation code in `src/training/train_scratch_resnet.py`, so each trial trains with the same recipe as the baseline; only the tuned values change.

## What is tuned

| Hyperparameter | Grid values | Random / Optuna range | Experiment 9 value |
|---|---|---|---|
| `max_lr` (OneCycleLR peak) | 0.001, 0.002, 0.004 | 0.001–0.004, log-uniform | 0.002 |
| `weight_decay` (AdamW) | 0.01, 0.03, 0.1 | 0.01–0.1, log-uniform | 0.03 |

Learning rate and weight decay are the two settings that most affect how this model trains and how much it overfits. Dropout (0.3), label smoothing (0.1), CutMix probability (0.5) and batch size (64) stay at their Experiment 9 values. The search space and the fixed values are set in `configs/hyperparameter_tuning.yaml`. To tune another hyperparameter, add an entry under `search_space`, for example `dropout`, which is read from `fixed` today. Any tuned parameter also has to be wired into `run_trial`.

## How the comparison is kept fair

- **Same search box.** Random search and Optuna sample from the range between the lowest and highest grid values.
- **Same trial budget.** Random search and Optuna each get as many trials as the grid has (3 × 3 = 9). You can override this with `budget.random_trials` / `budget.optuna_trials` or `--trials`.
- **Same training per trial.** Every trial trains a new model from scratch for 10 epochs (`budget.epochs_per_trial`, with the OneCycle schedule compressed to fit), using the same seed (42) and the same data order.
- **Same score.** A trial's score is its best validation accuracy across its epochs, the same rule used to pick the baseline checkpoint. The test split is never loaded during tuning.
- **Optuna differences.** Optuna uses a TPE sampler (each new trial is chosen from the results so far) and a median pruner. The pruner stops a trial early, after at least 3 epochs, when its validation accuracy is below the median of earlier trials at the same epoch. Pruning is how Optuna saves time, so the comparison reports pruned trials and total epochs trained.

Ten epochs is a short proxy for the full 100-epoch run. It ranks settings by how quickly they learn, which usually, but not always, matches how they rank after full training. The final model must be retrained with the selected settings for the full schedule (see below).

## Running on Kaggle

Upload `notebooks/kaggle_hyperparameter_tuning.ipynb` to Kaggle (**Create → New Notebook → File → Import Notebook**). Set **Accelerator: GPU T4 x2** and **Internet: On**, then click **Save Version → Save & Run All (Commit)** so it runs in the background. The notebook clones this branch, runs all three searches with `--max-hours 11` (Kaggle stops sessions at 12 hours), and saves the results to the version's Output tab. If the time limit stops it early, attach that output as input and run again; finished trials are skipped.

## Running elsewhere (Colab or a local GPU)

```bash
pip install -r requirements.txt
# Optional: print every grid/random configuration without training
python src/training/tune_scratch_resnet.py --method all --dry-run

python src/training/tune_scratch_resnet.py --method grid
python src/training/tune_scratch_resnet.py --method random
python src/training/tune_scratch_resnet.py --method optuna      # optional
python src/training/tune_scratch_resnet.py --method compare
```

`--method all` runs all three searches one after another and then the comparison. Each search writes a row to `outputs/hyperparameter_tuning/<method>_trials.csv` as soon as a trial finishes. If Colab disconnects, run the same command again and it continues from the next unfinished trial; Optuna resumes from `optuna_study.db`. On Colab, keep `outputs/hyperparameter_tuning/` on Google Drive (or copy it there between sessions), so it survives a runtime reset. Use `--epochs` to change the per-trial budget and `--output-dir` to keep a separate set of runs.

## Outputs (`outputs/hyperparameter_tuning/`)

| File | Contents |
|---|---|
| `grid_trials.csv`, `random_trials.csv`, `optuna_trials.csv` | One row per trial: settings, best/final validation accuracy, best epoch, epochs completed, whether it was pruned, seconds |
| `comparison.csv`, `comparison.md` | Per method: trials, pruned trials, epochs trained, best validation accuracy and settings, trials and minutes needed to reach the best, total time |
| `best_score_comparison.png` | Best-so-far validation accuracy against trial number and against cumulative time |
| `search_space_coverage.png` | The settings each method tried and their scores (grid points vs. random/TPE samples) |
| `best_hyperparameters.json` | The best settings across all methods, chosen on validation accuracy only |

## Final model

Retrain with the selected values for the full 100-epoch schedule, then evaluate on the test set once:

```bash
python src/training/train_scratch_resnet.py --mode train --epochs 100 \
    --max-lr <best max_lr> --weight-decay <best weight_decay> \
    --output-dir outputs/hyperparameter_tuning/final_model
python src/training/train_scratch_resnet.py --mode evaluate \
    --checkpoint outputs/hyperparameter_tuning/final_model/best.pth \
    --output-dir outputs/hyperparameter_tuning/final_model
```

Compare the result with the Experiment 9 baseline (53.39% validation, 56.04% test; `outputs/baseline/README.md`).

## Tests

```bash
python -m unittest tests.test_hyperparameter_tuning -v
```

The tests check the grid's size, that random sampling is reproducible and stays within the bounds, the default trial budgets, one pruned trial on tiny random tensors, and the comparison reports. They don't need the dataset or a GPU.
