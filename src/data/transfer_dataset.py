"""CUB dataset view backed by the repository's established split CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import Dataset

from src.data.eda import _image_object, load_cub_datasets


class CUBTransferDataset(Dataset):
    """Read split membership from CSV and image records from the existing HF loader."""

    def __init__(self, rows: pd.DataFrame, source_datasets: dict[str, Any], transform=None):
        self.rows = rows.reset_index(drop=True)
        self.source_datasets = source_datasets
        self.transform = transform
        if self.rows.empty:
            raise ValueError("The requested dataset partition is empty.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        row = self.rows.iloc[item]
        label = int(row["label"])
        source_split = str(row["split"])
        source_index = int(row["index"])
        try:
            record = self.source_datasets[source_split][source_index]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Missing {source_split} dataset record at index {source_index}.") from exc
        if int(record["label"]) != label:
            raise ValueError(
                f"Split label mismatch for {source_split}[{source_index}]: "
                f"CSV={label}, dataset={record['label']}."
            )
        with _image_object(record["image"]) as opened:
            image = opened.convert("RGB")
            image.load()
            image = image.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(row["id"])


def load_transfer_splits(config: dict, root: Path, model_weights):
    """Load exact existing splits and weight-specific transforms; never resplit."""
    split_path = root / config["dataset"]["splits"]
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Established split file not found: {split_path}. Run the existing EDA pipeline first."
        )
    splits = pd.read_csv(split_path)
    required = {"partition", "split", "index", "id", "label"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Split file is missing columns: {sorted(missing)}")
    splits["index"] = pd.to_numeric(splits["index"], errors="raise").astype(int)
    splits["label"] = pd.to_numeric(splits["label"], errors="raise").astype(int)
    num_classes = int(config["dataset"]["num_classes"])
    if (splits["label"] < 0).any() or (splits["label"] >= num_classes).any():
        raise ValueError(f"Split labels must be in [0, {num_classes - 1}].")
    if splits["id"].isna().any() or splits.duplicated(["split", "index"]).any():
        raise ValueError("Split file contains missing IDs or duplicate source records.")
    if set(splits["partition"]) != {"train", "validation", "test"}:
        raise ValueError("Split file must contain train, validation, and test partitions.")

    dataset_config = yaml.safe_load(
        (root / config["dataset"]["config"]).read_text(encoding="utf-8")
    )
    train_source, test_source = load_cub_datasets(dataset_config)
    sources = {"train": train_source, "test": test_source}
    for _, row in splits.iterrows():
        if row["split"] not in sources or int(row["index"]) >= len(sources[row["split"]]):
            raise ValueError(f"Invalid source reference in split file: {row['split']}[{row['index']}].")

    preprocessing = model_weights.transforms()
    from torchvision import transforms

    crop_size = preprocessing.crop_size
    size = int(crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, interpolation=preprocessing.interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocessing.mean, std=preprocessing.std),
        ]
    )
    evaluation_transform = preprocessing
    result = {}
    for partition in ("train", "validation", "test"):
        rows = splits[splits["partition"] == partition]
        result[partition] = CUBTransferDataset(
            rows,
            sources,
            train_transform if partition == "train" else evaluation_transform,
        )
    return result, splits
