# Transfer-Learning Results

## 1. Objective

This experiment compares ResNet-50 and EfficientNet-B3 for the
200-class CUB-200-2011 bird species classification task.

Two transfer-learning strategies were evaluated for each model:
- Feature extraction: freeze the pretrained backbone and train the classifier.
- Fine-tuning: train the classifier and selected backbone layers.

## 2. Experimental Results

All four checkpoints were selected using validation accuracy.
The held-out test set contains 5,794 images.

| Model | Stage | Best Epoch | Best Validation Accuracy | Test Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Feature extraction | 18 | 60.85% | 60.75% | 60.80% |
| ResNet-50 | Fine-tuning | 2 | 64.29% | 66.33% | 66.24% |
| EfficientNet-B3 | Feature extraction | 23 | 69.30% | 70.37% | 70.36% |
| EfficientNet-B3 | Fine-tuning | 1 | 68.85% | 70.25% | 70.30% |

## 3. Discussion

EfficientNet-B3 with feature extraction achieved the highest test
accuracy (70.37%) and macro F1-score (70.36%) among these four runs.

Fine-tuning improved ResNet-50's test accuracy from 60.75% to 66.33%.
For EfficientNet-B3, fine-tuning produced a slightly lower test accuracy
than feature extraction (70.25% versus 70.37%).

These findings describe the experiments conducted; they do not establish
that one architecture or training strategy will always perform best.

## 4. Timing

| Model | Stage | Training Time (seconds) | End-to-End Inference Time per Image (seconds) |
|---|---|---:|---:|
| ResNet-50 | Feature extraction | 1116.32 | 0.00818 |
| ResNet-50 | Fine-tuning | 1549.53 | 0.00828 |
| EfficientNet-B3 | Feature extraction | 2112.90 | 0.01309 |
| EfficientNet-B3 | Fine-tuning | 2684.42 | 0.01304 |

Inference timing includes DataLoader iteration, preprocessing, and device
transfer. These timings are specific to the recorded environment.

## 5. Output Files

Each model folder contains its class mapping, evaluation metrics,
confusion matrix, per-class metrics, and test predictions.
Each experiment folder contains its best checkpoint, training history,
and training-curve plot.

- `comparison.csv`
- `comparison_all_experiments.csv`
- `results_summary.md`

The scratch-baseline row, if included in the comparison, has only the
metrics recorded in its README. Precision, recall, F1, predictions, and
timing are unavailable for that run.
