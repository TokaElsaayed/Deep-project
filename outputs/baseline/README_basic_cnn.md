# Baseline CNN — Person 1

## Objective
Build a CNN from scratch to classify 200 bird species.

## Dataset
- Training images: 5,092
- Validation images: 899
- Test images: 5,794
- Classes: 200
- Input: RGB images, 224 x 224 pixels
- Training: light data augmentation
- Validation/test: deterministic preprocessing
- Random seed: 42

## Architecture
1. Conv2D 3 -> 32, 3x3 kernel, ReLU, max pooling
2. Conv2D 32 -> 64, 3x3 kernel, ReLU, max pooling
3. Conv2D 64 -> 128, 3x3 kernel, ReLU, max pooling
4. Adaptive average pooling to 1x1
5. Linear classifier 128 -> 200

Trainable parameters: 119,048.
Weights initialized from scratch; no pretrained model used.

## Training configuration
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Cross-entropy
- Batch size: 32
- Total epochs recorded: 30
- Model selection: highest validation accuracy
- Seed: 42

## Recorded results
- Best validation accuracy: 10.01%
- Test accuracy: 10.01%
- Test loss: 4.9442
- Test images evaluated: 5,794

## Interpretation
The model learned training patterns, but performance remains limited
for fine-grained classification across 200 bird species. This baseline
provides a reference point for comparison with improved models.

## Artifacts
- Model: src/models/baseline_cnn.py
- Training script: src/training/train_baseline.py
- Training curves: outputs/baseline/training_curves.png

Note: The graph was recreated from metrics printed in the original
notebook. Trained checkpoint weights are not included unless recovered
separately from the original runtime or another saved location.
