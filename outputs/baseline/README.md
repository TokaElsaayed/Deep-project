# Person 1 — Scratch-Trained CNN Baseline

## Objective
Train a convolutional neural network from random initialization to classify 200 bird species. This experiment is the improved scratch-trained baseline; it does not use pretrained weights.

## Dataset
- Training images: 5,092
- Validation images: 899
- Test images: 5,794
- Classes: 200
- Input resolution: 128 × 128 RGB
- Random seed: 42

## Architecture
- Convolutional stem with 64 output channels
- Four residual stages with widths 64, 128, 256, and 512
- Two residual blocks per stage
- Adaptive average pooling
- Dropout: 0.3
- Linear classifier: 512 → 200 classes
- Weights initialized from scratch

## Training Configuration
- Epochs: 100
- Optimizer: AdamW
- Maximum learning rate: 0.002
- Weight decay: 0.03
- Learning-rate scheduler: OneCycleLR
- Loss: Cross-entropy with label smoothing (0.1)
- Augmentation: RandomResizedCrop, horizontal flip, ColorJitter, RandomErasing, and CutMix
- Mixed precision when CUDA is available
- Best checkpoint selected using validation accuracy

## Results
- Best validation accuracy: 53.39% (epoch 91)
- Standard test accuracy: 56.04%
- Test accuracy with horizontal-flip test-time augmentation (TTA): 56.56%

TTA averages predictions from the original and horizontally flipped versions of each test image. It is reported separately from standard test accuracy.

## Artifacts
- `best.pth`: best model checkpoint
- `history.csv`: training and validation metrics by epoch
- `loss_curves.png`: loss plot
- `accuracy_curves.png`: accuracy plot
- `training_curves.png`: previously recorded training curves

## Notes
The original basic CNN baseline and its documentation are preserved separately. The results in this README describe Experiment 9.

The test set should be used for final evaluation, not for selecting hyperparameters.


## Longer-training experiment

To test whether additional training improves the scratch CNN, run up to
100 epochs and save the experiment separately from this recorded baseline:

```powershell
python baseline_cnn_complete.py --epochs 100 --output-dir outputs/baseline_extended
```

The script saves the checkpoint with the highest validation accuracy and
reports test accuracy once at the end. Compare the new validation and test
results with the baseline above; do not replace the baseline record. More
epochs may not improve accuracy, so use the validation-selected checkpoint
and report the measured results.

