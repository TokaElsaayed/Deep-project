# From Pixels to Language — Task 1.1 (EDA)

This repository's first milestone analyzes the CUB bird images/captions, creates a leakage-aware train/validation/test split, and prepares image transforms for classification. Paths, dataset IDs, split seed, pHash threshold, preprocessing sizes, and augmentation settings are in `configs/eda.yaml`.

## Rebuild the EDA outputs

Run these commands from the repository root. The first command downloads/loads the configured Hugging Face datasets; the first run needs network access. The EDA command rebuilds the raw image/caption reports, plots, cleaned captions, and split. If `outputs/eda/confirmed_perceptual_removals.csv` is present, it also excludes those previously reviewed training copies (never test records).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.data.eda --config configs/eda.yaml
.venv\Scripts\python.exe -m src.data.transforms --config configs/eda.yaml
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To refresh the perceptual-hash review, run:

```powershell
.venv\Scripts\python.exe -m src.data.near_duplicates --config configs/eda.yaml
```

Review `outputs/eda/perceptual_duplicate_review.png` and `outputs/eda/perceptual_duplicate_pairs.csv`. Mark each row's `visual_decision` as `confirmed duplicate` only when images are the same photo; otherwise record why it is not a duplicate. Then apply the review and regenerate captions/splits:

```powershell
.venv\Scripts\python.exe -m src.data.near_duplicates --config configs/eda.yaml --apply-reviewed
.venv\Scripts\python.exe -m src.data.transforms --config configs/eda.yaml
```

The optional `--final-audit` flag writes a separate pHash report for the resulting split. It is read-only with respect to the split, but its candidates still need visual review. Keep the small reviewed removal CSV and decision reports in version control for reproducibility; `caption_source.csv`, `clean_captions.csv`, and `image_report.csv` are larger generated files and are ignored by `.gitignore`.

## Findings → Decision → Reason

| Finding (evidence) | Decision | Reason |
|---|---|---|
| The final train split has 200 species with 25–26 images per species (`outputs/eda/class_counts.csv`). | Use stratified sampling for train/validation. | It keeps each species represented and nearly balanced in both partitions. |
| The supplied test split has 5,794 images; class counts range from 11 to 30 (`outputs/eda/test_class_distribution.png`). | Do not rebalance or edit test. Later report macro-F1 and per-class accuracy as well as accuracy. | The class support differs substantially, so overall accuracy alone can hide weak classes. |
| There are 11,788 images total; 11,780 are RGB and 8 are stored as grayscale `L`. Widths are 121–500 px, heights 120–500 px, and aspect ratios are 0.375–3.597 (`outputs/eda/image_report.csv`). | Convert every image to three-channel RGB and use 224×224 inputs. | A common input shape is required by the CNN; grayscale inputs must match RGB model inputs. |
| All 11,788 images decoded successfully. Mean normalized brightness is 0.488 and mean contrast proxy is 0.178 (`outputs/eda/image_report.csv`). | Record brightness/contrast as basic quality indicators, not a definitive quality score. | Blur, focus, and subject visibility were not measured automatically. |
| Exact-hash review found one duplicate within the original training data and zero exact train/test duplicates. pHash (8×8, Hamming distance ≤5) initially flagged 6 pairs; visual review confirmed 2 training copies to remove. The final scan's 4 remaining pairs were false positives; no train/validation candidates remained (`outputs/eda/perceptual_duplicate_pairs.csv`, `outputs/eda/perceptual_duplicate_pairs_final.csv`). | Remove only confirmed training copies; leave test unchanged. Treat pHash hits as candidates, not automatic duplicates. | Different photos of the same species and unrelated birds can have similar perceptual hashes. |
| The cleaned split contains 5,092 train, 899 validation, and all 5,794 test images (`outputs/eda/splits.csv`); seed is 42. | Stratify the train/validation split by label; keep the supplied test partition fixed. | This makes train/validation assignment reproducible and prevents test leakage from split generation. |
| There are 117,880 captions. Mean length is 15.27 words (median 14; range 1–66); the tokenized vocabulary has 5,466 word types (`outputs/eda/caption_source.csv`, `outputs/eda/vocabulary.csv`). | Split each description into separate caption rows and retain raw text beside cleaned text. | Caption-level analysis and later captioning need individual captions, and preserving raw text makes cleanup auditable. |
| A spell-check heuristic flagged 1,301 word types, including valid bird terms such as `wingbars`, `rectrices`, and `superciliary` (`outputs/eda/typo_candidates.csv`). Six approved mappings corrected 240 occurrences; `grey`→`gray` was applied 10,562 times (`outputs/eda/caption_cleaning_summary.csv`). | Apply only the reviewed mapping; do not run automatic spell correction. | General spell-check suggestions often damage correct ornithology terms. |
| 479 caption rows contained adjacent repeated-word runs; cleanup made 250 regex substitutions. | Remove adjacent duplicate runs, but keep the original caption column. | The 479 count is captions containing a run; 250 counts matched runs, not affected captions or individual repeated tokens. One substitution may collapse multiple adjacent repeats such as “with with with.” |
| The raw text contains 1,700 replacement characters (`�`) in 56 captions (`outputs/eda/caption_cleaning_summary.csv`). | Preserve and flag these characters. | The original characters cannot be reliably reconstructed without source evidence. |
| Sample review shows text/watermark content in at least one image; there is no dataset-wide detector/count. | Do not remove image regions; inspect later with Grad-CAM and report this as an unquantified risk. | A watermark or background text may become a shortcut feature, but cropping it could remove useful bird evidence. |
| Possible label noise in test has not been manually audited or counted. | Keep test labels unchanged; record this as a limitation and review suspicious errors later. | Changing test labels would compromise an independent evaluation. |
| RGB mean/std were computed using only the final 5,092 training images: mean `[0.484575, 0.499069, 0.432698]`, std `[0.232036, 0.227526, 0.266081]` (`configs/eda.yaml`). | Use these statistics for a scratch model; use ImageNet statistics for an ImageNet-pretrained model. | Normalization should match the model's training distribution; test data must not contribute to estimates. |
| Training uses random-resized crop, 50% horizontal flip, rotation up to 5°/10°, and brightness/contrast jitter of 0.08/0.15 for light/medium. Validation/test use short-side resize 256 and center crop 224 (`src/data/transforms.py`). | Compare light and medium augmentation; avoid vertical flips and hue changes. | Small viewpoint/lighting variation helps generalization while preserving bird orientation and species-defining color. |

## Task 1.1 checklist

| Requirement | Status | Evidence / remaining work |
|---|---|---|
| Class counts and imbalance analysis | DONE | `class_counts.csv`; test distribution plot; test support 11–30 per class. |
| Image dimensions, grayscale count, and basic quality | PARTIAL | Dimensions, grayscale modes, brightness, and contrast are reported. No blur/focus/subject-quality score. |
| Corrupt-image and duplicate checks | DONE | Zero unreadable images; exact and pHash reports; only confirmed training copies removed. |
| Caption length, vocabulary, typo candidates, and noisy-caption checks | PARTIAL | Length/vocabulary and repeated words are measured; spell-check output is a candidate list, not a full human typo audit. Replacement characters are preserved and flagged. |
| Split and caption cleaning | DONE | Ten captions per image are split; raw and cleaned columns remain; seed-42 stratified train/validation split; test rows/captions unchanged. |
| Resize, RGB conversion, and train-only normalization | DONE | Transform code and train-only RGB statistics in `configs/eda.yaml`. |
| Train-only augmentation and examples | DONE | Light/medium pipelines and `outputs/eda/augmentation_examples.png`; evaluation transforms are deterministic. |
| Test label noise audit | MISSING | No manual label audit or noise count exists; preserve test labels and assess errors during evaluation. |
| Dataset-wide text/watermark estimate | MISSING | A watermark is visible in sample review, but no OCR/watermark scan or prevalence estimate has been produced. Grad-CAM review remains for the classification milestone. |
