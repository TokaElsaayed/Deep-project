"""Config-driven image preprocessing and train-only augmentation utilities."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance

from src.data.eda import _image_object, load_cub_datasets


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    if width <= height:
        new_width = short_side
        new_height = int(short_side * height / width)
    else:
        new_height = short_side
        new_width = int(short_side * width / height)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    left = int(round((image.width - size) / 2.0))
    top = int(round((image.height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def _random_resized_crop(image: Image.Image, size: int, scale: list[float], rng: random.Random) -> Image.Image:
    area = image.width * image.height
    log_ratio = (math.log(0.75), math.log(4.0 / 3.0))
    for _ in range(10):
        target_area = area * rng.uniform(float(scale[0]), float(scale[1]))
        aspect_ratio = math.exp(rng.uniform(*log_ratio))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= image.width and 0 < crop_height <= image.height:
            left = rng.randint(0, image.width - crop_width)
            top = rng.randint(0, image.height - crop_height)
            return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                (size, size), Image.Resampling.BILINEAR
            )
    width, height = image.size
    input_ratio = width / height
    if input_ratio < 0.75:
        crop_width, crop_height = width, int(round(width / 0.75))
    elif input_ratio > 4.0 / 3.0:
        crop_width, crop_height = int(round(height * (4.0 / 3.0))), height
    else:
        crop_width, crop_height = width, height
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


class BirdImageTransform:
    """PIL preprocessing that returns a normalized PyTorch tensor when called."""

    def __init__(self, config: dict[str, Any], mode: str, strength: str = "light", seed: int | None = None):
        if mode not in {"train", "evaluation"}:
            raise ValueError("mode must be 'train' or 'evaluation'")
        self.config = config
        self.mode = mode
        self.strength = strength
        self.rng = random.Random(int(config["seed"] if seed is None else seed))
        preprocessing = config["preprocessing"]
        self.size = int(preprocessing["input_size"])
        self.resize_short_side = int(preprocessing["resize_short_side"])
        if mode == "train" and strength not in preprocessing["augmentation"]:
            raise ValueError(f"Unknown augmentation strength: {strength}")

    def apply_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.mode == "evaluation":
            return _center_crop(_resize_short_side(image, self.resize_short_side), self.size)

        settings = self.config["preprocessing"]["augmentation"][self.strength]
        image = _random_resized_crop(image, self.size, settings["crop_scale"], self.rng)
        if self.rng.random() < float(settings["horizontal_flip_probability"]):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        degrees = float(settings["rotation_degrees"])
        angle = self.rng.uniform(-degrees, degrees)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        radians = math.radians(abs(angle))
        safe_size = max(1, int(self.size / (math.cos(radians) + math.sin(radians))))
        image = _center_crop(image, safe_size).resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        brightness = float(settings["brightness"])
        contrast = float(settings["contrast"])
        image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(1 - brightness, 1 + brightness))
        image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(1 - contrast, 1 + contrast))
        return image

    def __call__(self, image: Image.Image):
        transformed = self.apply_image(image)
        pixels = np.asarray(transformed, dtype=np.float32).copy() / 255.0
        tensor_data = np.transpose(pixels, (2, 0, 1)).copy()
        import torch

        tensor = torch.from_numpy(tensor_data)
        normalization = self.config["normalization"]
        mean = torch.tensor(normalization["rgb_mean"], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(normalization["rgb_std"], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


def build_train_transform(config: dict[str, Any], strength: str = "light", seed: int | None = None) -> BirdImageTransform:
    """Build a train-only light or medium augmentation pipeline."""
    return BirdImageTransform(config, mode="train", strength=strength, seed=seed)


def build_evaluation_transform(config: dict[str, Any]) -> BirdImageTransform:
    """Build deterministic resize/center-crop preprocessing for validation/test."""
    return BirdImageTransform(config, mode="evaluation")


def _rgb_moments(images: Iterable[Image.Image]) -> tuple[list[float], list[float]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for image in images:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        channel_sum += pixels.sum(axis=(0, 1))
        channel_square_sum += np.square(pixels).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]
    if pixel_count == 0:
        raise ValueError("Cannot compute RGB statistics from an empty image collection.")
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean), 0.0)
    return mean.tolist(), np.sqrt(variance).tolist()


def compute_train_rgb_statistics(dataset, train_indices: Iterable[int]) -> tuple[list[float], list[float]]:
    """Compute channel-wise statistics using only listed training image indices."""
    return _rgb_moments(_image_object(dataset[int(index)]["image"]) for index in train_indices)


def _save_config(config_file: Path, config: dict[str, Any]) -> None:
    config_file.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_config(config_file: Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config_file = config_file.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = config_file.parent.parent
    return config, root, {key: root / value for key, value in config["paths"].items()}


def _make_augmentation_figure(config: dict[str, Any], paths: dict[str, Path], dataset, splits: pd.DataFrame) -> None:
    train_indices = splits.loc[splits["partition"] == "train", "index"].astype(int).tolist()
    if not train_indices:
        raise ValueError("The train split is empty; cannot create augmentation examples.")
    rng = random.Random(int(config["seed"]))
    selected = sorted(rng.sample(train_indices, min(3, len(train_indices))))
    evaluation = build_evaluation_transform(config)
    light = build_train_transform(config, "light", seed=int(config["seed"]))
    medium = build_train_transform(config, "medium", seed=int(config["seed"]) + 1)
    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3.2 * len(selected)), squeeze=False)
    titles = ("Original", "Validation/test", "Train: light", "Train: medium")
    for row_number, image_index in enumerate(selected):
        source = _image_object(dataset[image_index]["image"]).convert("RGB")
        for column, (transform, title) in enumerate(zip((None, evaluation, light, medium), titles)):
            shown = source if transform is None else transform.apply_image(source)
            axes[row_number, column].imshow(shown)
            axes[row_number, column].set_title(title)
            axes[row_number, column].axis("off")
    fig.tight_layout()
    fig.savefig(paths["augmentation_examples"], dpi=160)
    plt.close(fig)


def _make_test_class_plot(config: dict[str, Any], paths: dict[str, Path], splits: pd.DataFrame) -> None:
    test_rows = splits[splits["partition"] == "test"]
    class_count = int(config["dataset"]["num_classes"])
    counts = test_rows["label"].value_counts().reindex(range(class_count), fill_value=0).sort_index()
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.bar(counts.index, counts.values, width=1.0)
    axis.set(title="Test images per species", xlabel="Label ID", ylabel="Images")
    axis.set_xticks(range(0, class_count, 10))
    axis.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(paths["test_class_distribution_plot"], dpi=160)
    plt.close(fig)
    print(f"Test images: {len(test_rows)}; per-class min={counts.min()}, max={counts.max()}; test rows left unchanged.")


def run_phase_c(config_file: Path) -> None:
    config, _, paths = _load_config(config_file)
    splits = pd.read_csv(paths["splits"])
    train_indices = splits.loc[splits["partition"] == "train", "index"].astype(int).tolist()
    train_dataset, _ = load_cub_datasets(config)
    mean, std = compute_train_rgb_statistics(train_dataset, train_indices)
    config["normalization"] = {"rgb_mean": mean, "rgb_std": std, "computed_from": "final train split only"}
    _save_config(config_file.resolve(), config)
    _make_augmentation_figure(config, paths, train_dataset, splits)
    _make_test_class_plot(config, paths, splits)
    print(f"Final train images used for normalization: {len(train_indices)}")
    print("RGB mean:", [round(value, 6) for value in mean])
    print("RGB std:", [round(value, 6) for value in std])
    print(f"Saved transform config and plots under {paths['outputs_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eda.yaml"))
    args = parser.parse_args()
    run_phase_c(args.config)


if __name__ == "__main__":
    main()
