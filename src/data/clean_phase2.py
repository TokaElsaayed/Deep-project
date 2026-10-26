"""Conservative caption cleanup and leakage-safe CUB split generation."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


TYPO_MAP = {
    "whtie": "white",
    "beek": "beak",
    "bil": "bill",
    "supercillary": "superciliary",
    "larsus": "tarsus",
    "rectices": "rectrices",
}
TOKEN_PATTERN = re.compile(r"\b([A-Za-z]+)\b")
REPEATED_WORD_PATTERN = re.compile(r"\b([\w']+)(\s+\1\b)+", re.IGNORECASE)


def _config_path(config: dict, key: str) -> Path:
    return Path(config["project_root"]) / config["paths"][key]


def _clean_caption(original: str) -> tuple[str, Counter]:
    counts: Counter = Counter()
    if "\ufffd" in original:
        counts["replacement_character_preserved"] += original.count("\ufffd")
    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = TYPO_MAP.get(word.lower())
        if replacement:
            counts[f"typo:{word.lower()}->{replacement}"] += 1
            return replacement
        return word

    text = TOKEN_PATTERN.sub(replace_word, original)
    text, repeated = REPEATED_WORD_PATTERN.subn(lambda match: match.group(1), text)
    counts["adjacent_repeated_words_removed"] += repeated
    text, gray_count = re.subn(r"\bgrey\b", "gray", text, flags=re.IGNORECASE)
    counts["grey_to_gray"] += gray_count
    return text, counts


def run(config_file: Path) -> None:
    config_file = config_file.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["project_root"] = str(config_file.parent.parent)
    image_report = pd.read_csv(_config_path(config, "image_report"))
    captions = pd.read_csv(_config_path(config, "captions"))
    official_test = image_report[image_report["split"] == "test"].copy()
    valid_train = image_report[
        (image_report["split"] == "train") & image_report["valid_image"].astype(bool)
    ].copy()

    valid_test_hashes = set(
        official_test.loc[official_test["valid_image"].astype(bool), "image_hash"].dropna()
    )
    cross_test = valid_train[valid_train["image_hash"].isin(valid_test_hashes)]
    source_train = valid_train[~valid_train["image_hash"].isin(valid_test_hashes)].copy()
    within_duplicates = source_train[source_train.duplicated("image_hash", keep="first")]
    source_train = source_train.drop_duplicates("image_hash", keep="first")
    perceptual_path = config["paths"].get("perceptual_removals")
    perceptual_removals = pd.DataFrame()
    if perceptual_path and (Path(config["project_root"]) / perceptual_path).exists():
        perceptual_removals = pd.read_csv(Path(config["project_root"]) / perceptual_path)
        removed_indices = set(perceptual_removals["index"].astype(int))
        source_train = source_train[~source_train["index"].astype(int).isin(removed_indices)].copy()
    duplicate_rows = pd.concat(
        [
            cross_test.assign(duplicate_type="train_test_exact"),
            within_duplicates.assign(duplicate_type="within_train_exact"),
            perceptual_removals.assign(duplicate_type="confirmed_perceptual_train_copy"),
        ],
        ignore_index=True,
    )
    duplicate_rows.to_csv(_config_path(config, "duplicate_report"), index=False)

    train_rows, validation_rows = train_test_split(
        source_train,
        test_size=float(config["validation_fraction"]),
        random_state=int(config["seed"]),
        stratify=source_train["label"],
    )
    split_parts = []
    for partition, part in (("train", train_rows), ("validation", validation_rows), ("test", official_test)):
        result = part[["split", "index", "id", "file_name", "label"]].copy()
        result.insert(0, "partition", partition)
        split_parts.append(result)
    splits = pd.concat(split_parts, ignore_index=True)
    splits.to_csv(_config_path(config, "splits"), index=False)
    train_counts = train_rows["label"].value_counts().sort_index()
    train_counts.rename_axis("label").reset_index(name="image_count").to_csv(
        _config_path(config, "class_counts"), index=False
    )
    if "class_distribution_plot" in config["paths"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(14, 4))
        axis.bar(train_counts.index.astype(str), train_counts.values, width=1)
        axis.set(title="Final training images per species", xlabel="Label ID", ylabel="Images")
        axis.tick_params(axis="x", labelsize=6)
        figure.tight_layout()
        figure.savefig(_config_path(config, "class_distribution_plot"), dpi=160)
        plt.close(figure)

    original_column = "caption" if "caption" in captions else "original_caption"
    captions = captions.rename(columns={original_column: "original_caption"})
    captions = captions.drop(columns=["partition", "cleaned_caption", "encoding_flag"], errors="ignore")
    caption_partitions = splits[["split", "id", "partition"]].drop_duplicates()
    captions = captions.merge(caption_partitions, on=["split", "id"], how="inner", validate="many_to_one")
    event_counts: Counter = Counter()
    cleaned_values = []
    for _, row in captions.iterrows():
        original = str(row["original_caption"])
        if row["partition"] == "test":
            cleaned = original
            if "\ufffd" in original:
                event_counts["replacement_character_preserved"] += original.count("\ufffd")
        else:
            cleaned, counts = _clean_caption(original)
            event_counts.update(counts)
        cleaned_values.append(cleaned)
    captions["cleaned_caption"] = cleaned_values
    captions["encoding_flag"] = captions["original_caption"].str.contains("\ufffd", regex=False)
    captions = captions.drop(columns=["caption"], errors="ignore")
    captions.to_csv(_config_path(config, "cleaned_captions"), index=False)
    repeated_caption_rows = captions["original_caption"].map(
        lambda caption: bool(REPEATED_WORD_PATTERN.search(str(caption)))
    )

    summary_rows = [
        {"measure": "images_before_train_cleaning", "count": int((image_report["split"] == "train").sum())},
        {"measure": "exact_train_test_duplicates_removed", "count": int(len(cross_test))},
        {"measure": "exact_within_train_duplicates_removed", "count": int(len(within_duplicates))},
        {"measure": "confirmed_perceptual_train_duplicates_removed", "count": int(len(perceptual_removals))},
        {"measure": "training_images_after_dedup", "count": int(len(source_train))},
        {"measure": "validation_images", "count": int(len(validation_rows))},
        {"measure": "test_images_retained", "count": int(len(official_test))},
        {"measure": "captions_changed_outside_test", "count": int(((captions["partition"] != "test") & (captions["original_caption"] != captions["cleaned_caption"])).sum())},
        {"measure": "test_captions_changed", "count": int(((captions["partition"] == "test") & (captions["original_caption"] != captions["cleaned_caption"])).sum())},
        {"measure": "captions_with_adjacent_repeated_words_before_cleaning", "count": int(repeated_caption_rows.sum())},
        {"measure": "captions_with_replacement_character", "count": int(captions["encoding_flag"].sum())},
    ]
    summary_rows.extend({"measure": key, "count": value} for key, value in sorted(event_counts.items()))
    pd.DataFrame(summary_rows).to_csv(_config_path(config, "cleaning_summary"), index=False)
    print(
        f"Removed {len(cross_test)} exact train-test, {len(within_duplicates)} exact within-train, "
        f"and {len(perceptual_removals)} confirmed perceptual train duplicates."
    )
    print(f"Split: {len(train_rows)} train, {len(validation_rows)} validation, {len(official_test)} untouched test images.")
    print(f"Saved original/clean caption pairs and summary under {_config_path(config, 'outputs_dir')}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eda.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
