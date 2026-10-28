#!/usr/bin/env python3
"""Build an isolated, one-class YOLO dataset from the six fruit detection sources.

This script does not read or write the classification split manifest or its
artifacts. Source images are copied; labels are normalized to one `fruit` class.
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUSIONS = []
ANNOTATION_ISSUES = []
CLIPPED_BOXES = 0


def image_files(folder):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def paired_records(root, split, fruit, mode=None):
    base = root / mode if mode else root / split
    image_dir, label_dir = base / "images", base / "labels"
    images = image_files(image_dir)
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    records = []
    for image in images:
        label = labels.pop(image.stem, None)
        if label is None:
            EXCLUSIONS.append({"source_path_or_label": image.relative_to(PROJECT_ROOT).as_posix(), "fruit_source": fruit,
                               "source_split_or_view": mode or split,
                               "reason": "image has no matching annotation file"})
            continue
        records.append({"fruit": fruit, "source_path": image, "source_split": mode or split,
                        "source_label": label, "split": split})
    if labels:
        for label in labels.values():
            EXCLUSIONS.append({"source_path_or_label": label.relative_to(PROJECT_ROOT).as_posix(), "fruit_source": fruit,
                               "source_split_or_view": mode or split,
                               "reason": "annotation has no matching image file"})
    return records


def parse_as_fruit_boxes(label_path):
    """Convert YOLO boxes or segmentation polygons to class-0 axis-aligned boxes."""
    global CLIPPED_BOXES
    converted = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            int(parts[0])  # Validate the source class ID, then intentionally discard it.
            coords = [float(x) for x in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid annotation {label_path}:{line_no}") from exc

        if len(coords) == 4:
            cx, cy, width, height = coords
            x1, y1 = cx - width / 2, cy - height / 2
            x2, y2 = cx + width / 2, cy + height / 2
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            raise ValueError(f"Expected a YOLO box or polygon at {label_path}:{line_no}")

        if any(v < -0.002 or v > 1 + 0.002 for v in (x1, y1, x2, y2)):
            ANNOTATION_ISSUES.append({"label_path": label_path.relative_to(PROJECT_ROOT).as_posix(), "line": line_no,
                                      "reason": "box extends beyond image boundaries", "annotation": line})
            continue
        bounded = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
        CLIPPED_BOXES += bounded != [x1, y1, x2, y2]
        x1, y1, x2, y2 = bounded
        if x2 <= x1 or y2 <= y1:
            ANNOTATION_ISSUES.append({"label_path": label_path.relative_to(PROJECT_ROOT).as_posix(), "line": line_no,
                                      "reason": "degenerate box removed", "annotation": line})
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        converted.append(f"0 {cx:.8f} {cy:.8f} {x2 - x1:.8f} {y2 - y1:.8f}")
    return converted


def split_unassigned_train(records, valid_count):
    """Carve a deterministic validation subset from train, preserving the source test split."""
    target = min(len(records), valid_count)
    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)
    selected = {r["source_path"] for r in shuffled[:target]}
    for record in records:
        if record["source_path"] in selected:
            record["split"] = "val"


def make_records():
    root = PROJECT_ROOT
    records = []

    specs = [
        (root / "DragonFruit/Dragon Fruit Detection", "dragon_fruit", {"train": "train", "valid": "val", "test": "test"}),
        (root / "avocado_hass_by_class/Avocado", "avocado", {"train": "train", "valid": "val", "test": "test"}),
        (root / "papaya", "papaya", {"train": "train", "test": "test"}),
        (root / "pineapple", "pineapple", {"train": "train", "valid": "val", "test": "test"}),
        (root / "sawo", "sawo", {"train": "train", "test": "test"}),
    ]
    for dataset_root, fruit, split_map in specs:
        for source_split, target_split in split_map.items():
            records.extend(paired_records(dataset_root, target_split, fruit, mode=source_split))

    mango_root = root / "Mango Shelf-Life Dataset/Mango_Dataset"
    rng = random.Random(SEED)
    for mode in ("Far_Field", "Proximal", "Single"):
        group = paired_records(mango_root, "train", "mango", mode=mode)
        rng.shuffle(group)
        n_train = round(len(group) * 0.70)
        n_val = round(len(group) * 0.15)
        for i, record in enumerate(group):
            record["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        records.extend(group)

    for fruit in ("papaya", "sawo"):
        source_train = [r for r in records if r["fruit"] == fruit and r["source_split"] == "train"]
        complete_dataset_count = sum(r["fruit"] == fruit for r in records)
        split_unassigned_train(source_train, round(complete_dataset_count * 0.15))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="detection_preprocessed", help="Output directory (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the generated output directory")
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {out} (use --overwrite to rebuild this output only)")
        shutil.rmtree(out)

    records = make_records()
    prepared_records = []
    for record in records:
        source_text = record["source_label"].read_text()
        boxes = parse_as_fruit_boxes(record["source_label"])
        if source_text.strip() and not boxes:
            EXCLUSIONS.append({"source_path_or_label": record["source_path"].relative_to(PROJECT_ROOT).as_posix(),
                               "fruit_source": record["fruit"],
                               "source_split_or_view": record["source_split"],
                               "reason": "all source annotations were invalid"})
            continue
        record["boxes"] = boxes
        prepared_records.append(record)
    records = prepared_records
    out.mkdir(parents=True)
    counts = Counter()
    manifest_rows = []
    for index, record in enumerate(sorted(records, key=lambda r: (r["split"], r["fruit"], str(r["source_path"])) )):
        split = record["split"]
        dest_name = f"{record['fruit']}__{index:06d}{record['source_path'].suffix.lower()}"
        image_dest = out / "images" / split / dest_name
        label_dest = out / "labels" / split / f"{Path(dest_name).stem}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source_path"], image_dest)
        boxes = record["boxes"]
        label_dest.write_text("\n".join(boxes) + ("\n" if boxes else ""))
        counts[(record["fruit"], split)] += 1
        manifest_rows.append({
            "image": image_dest.relative_to(out).as_posix(),
            "label": label_dest.relative_to(out).as_posix(),
            "fruit_source": record["fruit"],
            "split": split,
            "source_split_or_view": record["source_split"],
            "source_path": record["source_path"].relative_to(PROJECT_ROOT).as_posix(),
            "object_count": len(boxes),
        })

    (out / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: ['fruit']\n"
    )
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = [
        "# Prepared fruit detection dataset",
        "",
        "This is a separate YOLO detection dataset. It does not change the classification datasets, scripts, or `artifacts/splits.csv`.",
        "All source object classes are mapped to one class, `fruit`; the downstream classification models remain responsible for fruit identity and maturity.",
        "Where available, source train/validation/test assignments were retained. Mango had no split folders, so its three source views were deterministically split 70/15/15 with seed 42. Papaya and Sawo had no validation folder; 15% of each complete source dataset was deterministically selected from its train split for validation, while source test images were retained.",
        "YOLO segmentation polygons in the Avocado train set were converted to their enclosing axis-aligned bounding boxes. Empty source label files remain empty negative images. Images were copied without resizing; YOLO can resize/letterbox during training.",
        f"Bounding boxes clipped to image edges for rounding overflow: {CLIPPED_BOXES}; invalid annotation rows removed: {len(ANNOTATION_ISSUES)}.",
        "",
        "| Fruit source | train | val | test | total |",
        "|---|---:|---:|---:|---:|",
    ]
    fruits = ("avocado", "dragon_fruit", "mango", "papaya", "pineapple", "sawo")
    for fruit in fruits:
        vals = [counts[(fruit, split)] for split in ("train", "val", "test")]
        summary.append(f"| `{fruit}` | {vals[0]} | {vals[1]} | {vals[2]} | {sum(vals)} |")
    summary.extend(["", "`manifest.csv` records each output image's source fruit and path, source split/view, and converted object count.",
                    f"Unpaired or unusable source files excluded: {len(EXCLUSIONS)}; details are in `exclusions.csv`."])
    (out / "README.md").write_text("\n".join(summary) + "\n")
    with (out / "exclusions.csv").open("w", newline="") as f:
        fields = ("source_path_or_label", "fruit_source", "source_split_or_view", "reason")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(EXCLUSIONS)
    with (out / "annotation_issues.csv").open("w", newline="") as f:
        fields = ("label_path", "line", "reason", "annotation")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ANNOTATION_ISSUES)
    print(f"Prepared {len(records)} images in {out}")
    for fruit in fruits:
        print(f"{fruit}: " + ", ".join(f"{split}={counts[(fruit, split)]}" for split in ("train", "val", "test")))


if __name__ == "__main__":
    main()
