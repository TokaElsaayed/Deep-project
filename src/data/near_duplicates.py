"""Find near-duplicate images and safely rebuild the train/validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split

from src.data.eda import _image_object, load_cub_datasets


def _phash(image: Image.Image, hash_size: int, image_size: int) -> int:
    pixels = np.asarray(
        image.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    coefficients = dct(dct(pixels, axis=0), axis=1)
    low_frequency = coefficients[:hash_size, :hash_size]
    median = np.median(low_frequency[1:])
    packed = np.packbits((low_frequency > median).reshape(-1))
    return int.from_bytes(packed.tobytes(), byteorder="big")


def _pair_distances(left: list[int], right: list[int], max_distance: int, same: bool) -> list[tuple[int, int, int]]:
    """Compare 64-bit hashes in bounded NumPy blocks."""
    left_array = np.asarray(left, dtype=np.uint64)
    right_array = np.asarray(right, dtype=np.uint64)
    popcount = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)
    results: list[tuple[int, int, int]] = []
    for start in range(0, len(left), 128):
        end = min(start + 128, len(left))
        xor_values = left_array[start:end, None] ^ right_array[None, :]
        byte_values = xor_values.view(np.uint8).reshape(end - start, len(right), 8)
        distances = popcount[byte_values].sum(axis=2)
        rows, cols = np.where(distances <= max_distance)
        for row, col in zip(rows.tolist(), cols.tolist()):
            left_index = start + row
            if same and left_index >= col:
                continue
            results.append((left_index, col, int(distances[row, col])))
    return results


def _get_image(dataset, index: int) -> Image.Image:
    image = _image_object(dataset[index]["image"])
    image.load()
    return image


def scan(config_file: Path, report_key: str = "perceptual_duplicates") -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    config_file = config_file.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    project_root = config_file.parent.parent
    config["project_root"] = str(project_root)
    paths = {key: project_root / value for key, value in config["paths"].items()}
    train_data, test_data = load_cub_datasets(config)
    splits = pd.read_csv(paths["splits"])
    indexed_datasets = {"train": train_data, "test": test_data}
    hash_size = int(config["perceptual_hash"]["hash_size"])
    image_size = int(config["perceptual_hash"]["image_size"])
    records: list[dict[str, Any]] = []
    for partition in ("train", "validation", "test"):
        subset = splits[splits["partition"] == partition]
        source = "test" if partition == "test" else "train"
        dataset = indexed_datasets[source]
        for _, row in subset.iterrows():
            image = _get_image(dataset, int(row["index"]))
            records.append({
                "partition": partition,
                "source_split": source,
                "index": int(row["index"]),
                "id": str(row["id"]),
                "file_name": str(row["file_name"]),
                "label": int(row["label"]),
                "hash": _phash(image, hash_size, image_size),
            })
    groups = {partition: [row for row in records if row["partition"] == partition]
              for partition in ("train", "validation", "test")}
    max_distance = int(config["perceptual_hash"]["max_hamming_distance"])
    pair_groups = (
        ("within_train", groups["train"], groups["train"], True),
        ("train_validation", groups["train"], groups["validation"], False),
        ("train_test", groups["train"], groups["test"], False),
    )
    candidates: list[dict[str, Any]] = []
    for pair_type, left_rows, right_rows, same in pair_groups:
        for left_index, right_index, distance in _pair_distances(
            [row["hash"] for row in left_rows],
            [row["hash"] for row in right_rows],
            max_distance,
            same,
        ):
            left = left_rows[left_index]
            right = right_rows[right_index]
            candidates.append({
                "pair_type": pair_type,
                "hamming_distance": distance,
                "left_partition": left["partition"],
                "left_index": left["index"],
                "left_id": left["id"],
                "left_file_name": left["file_name"],
                "left_label": left["label"],
                "left_phash": f"{left['hash']:016x}",
                "right_partition": right["partition"],
                "right_index": right["index"],
                "right_id": right["id"],
                "right_file_name": right["file_name"],
                "right_label": right["label"],
                "right_phash": f"{right['hash']:016x}",
                "visual_decision": "pending review",
                "train_image_removed": "",
            })
    result = pd.DataFrame(candidates)
    output_path = paths[report_key]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return config, result, {"datasets": indexed_datasets, "records": records, "splits": splits, "paths": paths}


def save_review_figure(candidates: pd.DataFrame, data: dict[str, Any], output_path: Path) -> None:
    if candidates.empty:
        fig, axis = plt.subplots(figsize=(7, 2))
        axis.text(0.5, 0.5, "No pairs within configured pHash threshold", ha="center", va="center")
        axis.axis("off")
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    records = {(row["source_split"], row["index"]): row for row in data["records"]}
    limit = len(candidates)
    fig, axes = plt.subplots(limit, 2, figsize=(8, max(2.5, limit * 2.5)), squeeze=False)
    for axis_row, row in zip(axes, candidates.itertuples(index=False)):
        for axis, source_split, index, file_name, side in (
            (axis_row[0], "test" if row.left_partition == "test" else "train", row.left_index, row.left_file_name, "left"),
            (axis_row[1], "test" if row.right_partition == "test" else "train", row.right_index, row.right_file_name, "right"),
        ):
            image = _get_image(data["datasets"][source_split], int(index)).convert("RGB")
            image.thumbnail((320, 220))
            axis.imshow(image)
            axis.set_title(f"{row.pair_type} | d={row.hamming_distance}\n{file_name}", fontsize=8)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def apply_review(config: dict[str, Any], candidates: pd.DataFrame, data: dict[str, Any]) -> pd.DataFrame:
    """Remove manually confirmed train copies, then recreate a seeded stratified split."""
    if candidates.empty or (candidates["visual_decision"] == "pending review").any():
        raise ValueError("Review all pHash candidates and set visual_decision before applying removals.")
    paths = data["paths"]
    removals: set[tuple[str, int]] = set()
    for row in candidates.itertuples(index=False):
        if row.visual_decision != "confirmed duplicate":
            continue
        if row.pair_type == "within_train":
            removals.add(("train", max(int(row.left_index), int(row.right_index))))
        elif row.pair_type == "train_validation":
            removals.add(("train", int(row.left_index)))
        elif row.pair_type == "train_test":
            removals.add(("train", int(row.left_index)))
    annotated = candidates.copy()
    annotated["train_image_removed"] = annotated["train_image_removed"].astype(object)
    for row_index, row in annotated.iterrows():
        if row["visual_decision"] == "confirmed duplicate":
            annotated.loc[row_index, "train_image_removed"] = max(
                [str(row["left_id"]), str(row["right_id"])],
                key=lambda image_id: int(image_id) if image_id.isdigit() else -1,
            ) if row["pair_type"] == "within_train" else row["left_id"]
    annotated.to_csv(paths["perceptual_duplicates"], index=False)
    removal_rows = []
    for split, index in sorted(removals):
        source_partition = "test" if split == "test" else "train"
        source_row = data["splits"]
        source_row = source_row[(source_row["split"] == source_partition) & (source_row["index"] == index)]
        if source_row.empty:
            raise ValueError(f"Cannot map removed training index {index} to a dataset record.")
        selected = source_row.iloc[0]
        removal_rows.append({
            "source_split": "train",
            "index": int(index),
            "id": str(selected["id"]),
            "file_name": str(selected["file_name"]),
            "label": int(selected["label"]),
            "reason": "confirmed perceptual duplicate; train copy removed",
        })
    removal_columns = ["source_split", "index", "id", "file_name", "label", "reason"]
    newly_confirmed = pd.DataFrame(removal_rows, columns=removal_columns)
    existing_path = paths["perceptual_removals"]
    if existing_path.exists():
        previous = pd.read_csv(existing_path)
        confirmed_removals = pd.concat([previous, newly_confirmed], ignore_index=True)
        confirmed_removals = confirmed_removals.drop_duplicates(
            subset=["source_split", "index"], keep="first"
        )
    else:
        confirmed_removals = newly_confirmed
    confirmed_removals.to_csv(existing_path, index=False)

    old_splits = data["splits"]
    pool = old_splits[old_splits["partition"].isin(["train", "validation"])].copy()
    removal_indices = {index for split, index in removals if split == "train"}
    pool = pool[~pool["index"].isin(removal_indices)]
    train_rows, validation_rows = train_test_split(
        pool,
        test_size=float(config["validation_fraction"]),
        random_state=int(config["seed"]),
        stratify=pool["label"],
    )
    test_rows = old_splits[old_splits["partition"] == "test"].copy()
    output_parts = []
    for partition, subset in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        subset = subset.copy()
        subset["partition"] = partition
        output_parts.append(subset)
    pd.concat(output_parts, ignore_index=True).to_csv(paths["splits"], index=False)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eda.yaml"))
    parser.add_argument("--apply-reviewed", action="store_true")
    parser.add_argument("--final-audit", action="store_true")
    args = parser.parse_args()
    if args.apply_reviewed:
        config_file = args.config.resolve()
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        project_root = config_file.parent.parent
        paths = {key: project_root / value for key, value in config["paths"].items()}
        candidates = pd.read_csv(paths["perceptual_duplicates"])
        data = {"splits": pd.read_csv(paths["splits"]), "paths": paths}
        apply_review(config, candidates, data)
        from src.data.clean_phase2 import run as run_cleaning

        run_cleaning(config_file)
        print("Applied reviewed removals and regenerated caption/split outputs.")
        return
    report_key = "perceptual_final_duplicates" if args.final_audit else "perceptual_duplicates"
    review_key = "perceptual_final_review" if args.final_audit else "perceptual_review"
    config, candidates, data = scan(args.config, report_key)
    save_review_figure(candidates, data, data["paths"][review_key])
    print(f"pHash candidates (Hamming distance <= {config['perceptual_hash']['max_hamming_distance']}): {len(candidates)}")
    if not candidates.empty:
        print(candidates.groupby("pair_type").size().to_string())
    print(f"Candidate report: {data['paths'][report_key]}")
    print(f"Visual review: {data['paths'][review_key]}")


if __name__ == "__main__":
    main()
