# Transfer-learning results

See `comparison.csv` and each model's `evaluation_metrics.json`. Checkpoints are selected by validation accuracy; the held-out test split is evaluated only by the explicit evaluate command. The scratch baseline row contains only metrics recorded in its README; precision, recall, F1, predictions, and timing are unavailable for that run. Inference timing is end-to-end across DataLoader iteration, including preprocessing and device transfer.
