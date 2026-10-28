# Transfer learning

The transfer-learning pipeline reuses `outputs/eda/splits.csv` and the Hugging Face loader in `src/data/eda.py`. It does not create or modify partitions. Labels remain the existing integer IDs 0–199. ResNet-50 uses the official ImageNet weights' 224-pixel evaluation crop; EfficientNet-B3 uses its official 300-pixel crop. Training uses random resized crops and horizontal flips. Validation and test use the selected weights' deterministic resize, center crop, and ImageNet normalization. The project's scratch-model mean/std transform is not used here.

Install dependencies in the project's virtual environment. Install a matching PyTorch/torchvision pair for your operating system and CUDA version using the official PyTorch install selector, then install the remaining requirements:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first training run downloads the selected ImageNet weights and loads the configured CUB datasets. From the repository root, train the classifier head, then fine-tune from that stage's best validation checkpoint:

```powershell
# ResNet-50: feature extraction and fine-tuning
.venv\Scripts\python.exe -m src.training.transfer_learning train --model resnet50 --stage feature_extraction
.venv\Scripts\python.exe -m src.training.transfer_learning train --model resnet50 --stage fine_tuning --init-checkpoint outputs/transfer_learning/resnet50/feature_extraction/best_checkpoint.pth

# EfficientNet-B3: feature extraction and fine-tuning
.venv\Scripts\python.exe -m src.training.transfer_learning train --model efficientnet_b3 --stage feature_extraction
.venv\Scripts\python.exe -m src.training.transfer_learning train --model efficientnet_b3 --stage fine_tuning --init-checkpoint outputs/transfer_learning/efficientnet_b3/feature_extraction/best_checkpoint.pth
```

Once a model/stage has been selected using validation performance, evaluate its best checkpoint on test exactly once:

```powershell
.venv\Scripts\python.exe -m src.training.transfer_learning evaluate --model resnet50 --checkpoint outputs/transfer_learning/resnet50/fine_tuning/best_checkpoint.pth
.venv\Scripts\python.exe -m src.training.transfer_learning evaluate --model efficientnet_b3 --checkpoint outputs/transfer_learning/efficientnet_b3/fine_tuning/best_checkpoint.pth
```

Choose another stage checkpoint if feature extraction is the selected model. For limited GPU memory, reduce batch size explicitly, for example by appending `--batch-size 16` (or `8`) to a train or evaluate command. Epochs, batch size, worker count, learning rates, weight decay, patience, and backbone blocks are configurable in `configs/transfer_learning.yaml`; `--epochs` and `--patience` override stage values. `--device cpu` forces CPU execution, and `--device cuda` requires an available CUDA device.

Training writes a validation-selected checkpoint, epoch history CSV, and loss/accuracy plot under `outputs/transfer_learning/<model>/<stage>/`. Evaluation writes JSON metrics, per-class metrics, a 200×200 confusion matrix CSV, predictions with CUB sample IDs, a class-index mapping, and an updated comparison table under `outputs/transfer_learning/<model>/`. Reported macro and weighted precision/recall/F1 use one-vs-rest class scores with zero-division set to zero. Inference time is wall-clock time for the full test DataLoader iteration, including preprocessing and transfers; CUDA is synchronized at the end. Test data is not used for checkpoint selection.

The scratch baseline row in the comparison table is copied from `outputs/baseline/README.md`; it contains recorded accuracy and loss only, without per-class metrics, predictions, or timing. No pretrained model results are included until the commands above are actually run.

Run checks from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
