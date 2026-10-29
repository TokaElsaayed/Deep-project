# Model comparison

This compares the two completed transfer-learning models with the scratch-trained ResNet. All three use the same held-out test split of 5,794 images.

| Model | Training approach | Best validation accuracy | Standard test accuracy | Test accuracy with TTA | Test loss | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Scratch ResNet | Trained from random initialization | 53.39% | 56.04% | 56.56% | N/A | N/A |
| ResNet-50 | ImageNet-pretrained, fine-tuned | 64.29% | 66.29% | N/A | 1.2535 | 66.23% |
| EfficientNet-B3 | ImageNet-pretrained, feature extraction | 69.30% | **70.31%** | N/A | 1.6243 | **70.33%** |

EfficientNet-B3 has the highest standard test accuracy, 4.02 percentage points above transfer ResNet-50 and 14.27 points above the scratch ResNet. Transfer ResNet-50 is 10.25 points above the scratch ResNet. The scratch ResNet's horizontal-flip test-time augmentation (TTA) result is reported separately because the transfer-model results do not include TTA.

The models were evaluated on the same test examples, but their input resolutions and preprocessing differ. The scratch ResNet uses 128 x 128 inputs and training-set normalization; the transfer models use their pretrained weights' evaluation transforms at 224 x 224 or 300 x 300 and ImageNet normalization. This compares the completed model pipelines rather than isolating architecture alone.

The scratch ResNet's reported results come from the local `main` branch's `outputs/baseline/README.md`; its validation accuracy is corroborated by `outputs/baseline/history.csv` (best epoch 91). Its README does not provide test loss or macro F1. The transfer-model values come from their saved `evaluation_metrics.json` files. Training and inference times are omitted because the scratch ResNet report does not provide comparable timing measurements.
