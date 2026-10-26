"""Load CUB data, inspect it, clean captions, and create reproducible splits."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageStat, UnidentifiedImageError


def load_cub_datasets(config: dict):
    """Load configured Hugging Face partitions."""
    from datasets import Dataset, config as datasets_config, load_dataset

    datasets = config["dataset"]

    def load_partition(dataset_id: str, split: str):
        offline = os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
        if offline:
            dataset_name = dataset_id.rsplit("/", 1)[-1].lower()
            cache_root = Path(datasets_config.HF_DATASETS_CACHE)
            cached_files = sorted(
                cache_root.glob(f"*___{dataset_name}/**/{dataset_name}-{split}.arrow"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if cached_files:
                return Dataset.from_file(str(cached_files[0]))
        return load_dataset(dataset_id, split=split)

    return (
        load_partition(datasets["train_id"], datasets["train_split"]),
        load_partition(datasets["test_id"], datasets["test_split"]),
    )


def split_description(description: Any) -> list[str]:
    """Split descriptions without changing caption content."""
    if not isinstance(description, str):
        return []
    return description.splitlines()


def _image_object(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field
    if isinstance(image_field, dict):
        if image_field.get("path"):
            return Image.open(image_field["path"])
        if image_field.get("bytes"):
            import io

            return Image.open(io.BytesIO(image_field["bytes"]))
    if isinstance(image_field, (str, Path)):
        return Image.open(image_field)
    raise ValueError("Unsupported or missing image value")


def _inspect_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    description = record.get("description")
    if not isinstance(description, str):
        description = ""
    row = {
        "split": split,
        "index": index,
        "id": str(record.get("id", "")),
        "file_name": str(record.get("file_name", "")),
        "label": int(record["label"]),
        "captions": split_description(description),
        "valid_image": False,
        "width": None,
        "height": None,
        "mode": None,
        "mean_brightness": None,
        "contrast": None,
        "image_hash": None,
    }
    try:
        image = _image_object(record.get("image"))
        image.load()
        rgb = image.convert("RGB")
        row.update(
            valid_image=True,
            width=image.width,
            height=image.height,
            mode=image.mode,
            mean_brightness=sum(ImageStat.Stat(rgb.convert("L")).mean) / 255,
            contrast=ImageStat.Stat(rgb.convert("L")).stddev[0] / 255,
            image_hash=hashlib.sha256(
                f"{rgb.width}x{rgb.height}:".encode("ascii") + rgb.tobytes()
            ).hexdigest(),
        )
        image.close()
    except (OSError, ValueError, UnidentifiedImageError, TypeError):
        pass
    return row


def inspect_datasets(train_dataset, test_dataset) -> pd.DataFrame:
    rows = []
    for split, dataset in (("train", train_dataset), ("test", test_dataset)):
        rows.extend(_inspect_record(row, split, index) for index, row in enumerate(dataset))
    frame = pd.DataFrame(rows)
    frame["caption_count"] = frame["captions"].map(len)
    frame["caption_lengths"] = frame["captions"].map(
        lambda captions: [len(caption.split()) for caption in captions]
    )
    return frame


def _caption_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sample in frame.iterrows():
        for caption_number, caption in enumerate(sample["captions"], start=1):
            words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", caption.lower())
            rows.append(
                {
                    "split": sample["split"],
                    "id": sample["id"],
                    "label": sample["label"],
                    "caption_number": caption_number,
                    "original_caption": caption,
                    "word_count": len(words),
                    "words": words,
                }
            )
    return pd.DataFrame(rows)


def _plot_eda(frame: pd.DataFrame, captions: pd.DataFrame, paths: dict[str, Path]) -> None:
    output_dir = paths["outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = frame[frame["split"] == "train"]["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(counts.index.astype(str), counts.values, width=1)
    ax.set(title="Training images per species", xlabel="Label ID", ylabel="Images")
    ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(paths["class_distribution_plot"], dpi=160)
    plt.close(fig)

    valid = frame[frame["valid_image"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(valid["width"], bins=30)
    axes[0].set(title="Image widths", xlabel="Pixels", ylabel="Images")
    axes[1].hist(valid["height"], bins=30)
    axes[1].set(title="Image heights", xlabel="Pixels", ylabel="Images")
    fig.tight_layout()
    fig.savefig(paths["image_dimensions_plot"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(valid["mean_brightness"], bins=30, range=(0, 1))
    axes[0].set(title="Image brightness", xlabel="Mean grayscale intensity", ylabel="Images")
    axes[1].hist(valid["contrast"], bins=30, range=(0, 1))
    axes[1].set(title="Image contrast", xlabel="Grayscale standard deviation", ylabel="Images")
    fig.tight_layout()
    fig.savefig(paths["image_quality_plot"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(captions["word_count"], bins=30)
    axes[0].set(title="Caption length", xlabel="Words", ylabel="Captions")
    axes[1].hist(frame["caption_count"], bins=range(0, 13))
    axes[1].set(title="Captions per image", xlabel="Captions", ylabel="Images")
    fig.tight_layout()
    fig.savefig(paths["caption_statistics_plot"], dpi=160)
    plt.close(fig)

    duplicate_hashes = valid[valid["image_hash"].duplicated(keep=False)]
    corrupt = frame[~frame["valid_image"]]
    frame.drop(columns=["captions", "caption_lengths"]).to_csv(paths["image_report"], index=False)
    captions.drop(columns=["words"]).to_csv(paths["caption_source"], index=False)
    duplicate_hashes.to_csv(paths["duplicate_images"], index=False)
    corrupt.drop(columns=["captions", "caption_lengths"]).to_csv(paths["corrupt_images"], index=False)
    frame[frame["split"] == "train"]["label"].value_counts().sort_index().rename_axis(
        "label"
    ).reset_index(name="image_count").to_csv(paths["class_counts"], index=False)
    frame[["split", "id", "label", "caption_count"]].assign(
        exactly_ten_captions=lambda rows: rows["caption_count"] == 10
    ).to_csv(paths["caption_quality"], index=False)
    vocabulary = Counter(word for words in captions["words"] for word in words)
    pd.DataFrame(vocabulary.most_common(), columns=["word", "count"]).to_csv(
        paths["vocabulary"], index=False
    )


def _plot_sample_images(dataset, output_file: Path) -> None:
    """Save a small montage of examples from different species."""
    selected = {}
    for record in dataset:
        selected.setdefault(int(record["label"]), record)
        if len(selected) == 12:
            break
    if not selected:
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for axis, (label, record) in zip(axes.flat, selected.items()):
        try:
            image = _image_object(record["image"])
            image.load()
            axis.imshow(image.convert("RGB"))
            axis.set_title(f"Label {label}")
            axis.axis("off")
            image.close()
        except (OSError, ValueError, UnidentifiedImageError, TypeError):
            axis.set_title(f"Label {label}: unreadable")
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def _write_typo_candidates(captions: pd.DataFrame, output_file: Path) -> int:
    """Write likely misspellings for human review; never alter caption text automatically."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        print("Typo review skipped; install pyspellchecker to generate typo candidates.")
        return 0
    checker = SpellChecker()
    counts = Counter(word for words in captions["words"] for word in words)
    candidates = [
        {"word": word, "count": count, "suggestion": checker.correction(word) or ""}
        for word, count in counts.items()
        if len(word) > 2 and word.isalpha() and word not in checker
    ]
    pd.DataFrame(candidates, columns=["word", "count", "suggestion"]).sort_values("count", ascending=False).to_csv(
        output_file, index=False
    )
    return len(candidates)


def _print_summary(frame: pd.DataFrame, captions: pd.DataFrame, output_dir: Path, typo_count: int) -> None:
    vocab = Counter(word for words in captions["words"] for word in words)
    train_counts = frame.loc[frame["split"] == "train", "label"].value_counts()
    print(f"Images: {len(frame)} ({(frame['split'] == 'train').sum()} train, {(frame['split'] == 'test').sum()} test)")
    print(f"Valid images: {int(frame['valid_image'].sum())}; corrupt/unreadable: {int((~frame['valid_image']).sum())}")
    print(f"Train species: {train_counts.size}; images/species min={train_counts.min()}, max={train_counts.max()}, mean={train_counts.mean():.1f}")
    print(f"Exact duplicate image pairs are listed in {output_dir / 'duplicate_images.csv'}")
    print(f"Caption rows: {len(captions)}; images with exactly 10 captions: {(frame['caption_count'] == 10).sum()}/{len(frame)}")
    if not captions.empty:
        print(f"Caption words: mean={captions['word_count'].mean():.1f}, median={captions['word_count'].median():.0f}, vocabulary={len(vocab)} unique words")
    print(f"Possible misspelled word types for review: {typo_count} (spell checks may flag bird terminology).")
    print("Raw captions are preserved; conservative cleaned captions are saved separately.")
    print(f"Reports and plots: {output_dir}")


def run_eda(config_file: Path) -> None:
    config_file = config_file.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    project_root = config_file.parent.parent
    paths = {key: project_root / value for key, value in config["paths"].items()}
    output_dir = paths["outputs_dir"]
    train_dataset, test_dataset = load_cub_datasets(config)
    frame = inspect_datasets(train_dataset, test_dataset)
    captions = _caption_table(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_sample_images(train_dataset, paths["sample_images_plot"])
    _plot_eda(frame, captions, paths)
    typo_count = _write_typo_candidates(captions, paths["typo_candidates"])
    _print_summary(frame, captions, output_dir, typo_count)
    from src.data.clean_phase2 import run as run_cleaning

    run_cleaning(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eda.yaml"))
    arguments = parser.parse_args()
    run_eda(arguments.config)


if __name__ == "__main__":
    main()
