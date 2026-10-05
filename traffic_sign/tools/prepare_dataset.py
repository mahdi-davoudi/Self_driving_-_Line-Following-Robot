"""
One-command dataset preparation for YOLO training.

Pipeline:
  1) Read all raw images + YOLO label files.
  2) Stratified split into train / val (by the first class in each label file;
     images without objects are treated as "background" and also split).
  3) Copy into the layout Ultralytics expects:
         <output>/images/{train,val}   and   <output>/labels/{train,val}
  4) Augment ONLY the train images (photometric augmentations from
     augment_dataset.py) and copy the original label for each augmented copy.
     The val set is never augmented, so there is no train/val leakage.

Usage (from the project root, in PowerShell):

  python tools/prepare_dataset.py `
      --images raw/images --labels raw/labels `
      --output dataset

Put this file next to augment_dataset.py (it imports from it).
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from augment_dataset import FUNCTIONS, IMAGE_EXTENSIONS

DEFAULT_CLASS_NAMES = ["stop", "right", "left", "straight", "park"]
BACKGROUND = -1  # group key for images that contain no objects


def read_classes(label_path: Path | None) -> list[int]:
    """Return all class ids found in a YOLO label file (empty list if none)."""
    if label_path is None or not label_path.is_file():
        return []
    classes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            classes.append(int(line.split()[0]))
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Split, copy and augment a YOLO dataset.")
    parser.add_argument("--images", type=Path, required=True, help="folder with raw images")
    parser.add_argument("--labels", type=Path, required=True, help="folder with raw YOLO .txt labels")
    parser.add_argument("--output", type=Path, required=True, help="output dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=len(DEFAULT_CLASS_NAMES))
    parser.add_argument("--no-augment", action="store_true", help="skip augmenting the train set")
    parser.add_argument("--drop-augmented", action="store_true",
                        help="ignore raw images whose name ends with an augmentation suffix "
                             "(e.g. _blur, _noise) to avoid leakage from a previous augment run")
    parser.add_argument("--force", action="store_true",
                        help="delete existing <output>/images and <output>/labels first")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    # ---------- prepare output folders ----------
    out_images = args.output / "images"
    out_labels = args.output / "labels"
    for folder in (out_images, out_labels):
        if folder.exists() and any(folder.iterdir()):
            if not args.force:
                raise SystemExit(f"{folder} is not empty. Use --force to overwrite it.")
            shutil.rmtree(folder)
    for split in ("train", "val"):
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    # ---------- collect raw samples ----------
    aug_suffixes = tuple(f"_{name}" for name in FUNCTIONS)
    image_paths = sorted(
        p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    seen_stems: set[str] = set()
    augmented_like = 0
    samples = []  # (image_path, label_path_or_None, classes)
    for image_path in image_paths:
        stem = image_path.stem
        if stem.endswith(aug_suffixes):
            augmented_like += 1
            if args.drop_augmented:
                continue
        if stem in seen_stems:
            raise SystemExit(f"Duplicate image name found: {stem}. Names must be unique.")
        seen_stems.add(stem)

        label_path = args.labels / f"{stem}.txt"
        if not label_path.is_file():
            label_path = None
        classes = read_classes(label_path)

        bad = [c for c in classes if not 0 <= c < args.num_classes]
        if bad:
            raise SystemExit(f"{stem}: class id(s) {sorted(set(bad))} outside 0..{args.num_classes - 1}")

        samples.append((image_path, label_path, classes))

    if augmented_like and not args.drop_augmented:
        print(f"WARNING: {augmented_like} raw images look already augmented "
              f"(names end with {aug_suffixes}). They may leak between train and val. "
              f"Re-run with --drop-augmented if they are copies of other images.")

    # ---------- stratified split ----------
    groups: dict[int, list] = defaultdict(list)
    for sample in samples:
        classes = sample[2]
        groups[classes[0] if classes else BACKGROUND].append(sample)

    train_samples, val_samples = [], []
    for key, items in sorted(groups.items()):
        rng.shuffle(items)
        if len(items) > 1:
            n_val = min(max(1, round(len(items) * args.val_ratio)), len(items) - 1)
        else:
            n_val = 0
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])
        label = "background" if key == BACKGROUND else f"class {key}"
        print(f"{label}: train={len(items) - n_val} val={n_val}")

    # ---------- copy (+ augment train) ----------
    def write_label(src: Path | None, dst: Path) -> None:
        if src is not None:
            shutil.copy(src, dst)
        else:
            dst.write_text("", encoding="utf-8")  # background image -> empty label

    counts = {"train": Counter(), "val": Counter()}
    n_aug = 0

    for split, items in (("train", train_samples), ("val", val_samples)):
        for image_path, label_path, classes in items:
            stem, suffix = image_path.stem, image_path.suffix
            shutil.copy(image_path, out_images / split / image_path.name)
            write_label(label_path, out_labels / split / f"{stem}.txt")
            counts[split].update(classes)

            if split == "train" and not args.no_augment:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Warning: unable to read {image_path}, skipping augmentation")
                    continue
                for name, function in FUNCTIONS.items():
                    aug_name = f"{stem}_{name}"
                    cv2.imwrite(str(out_images / split / f"{aug_name}{suffix}"), function(image))
                    write_label(label_path, out_labels / split / f"{aug_name}.txt")
                    counts[split].update(classes)
                    n_aug += 1

    # ---------- summary ----------
    names = DEFAULT_CLASS_NAMES
    print("\n===== SUMMARY =====")
    print(f"train originals: {len(train_samples)} | val: {len(val_samples)} | augmented train copies: {n_aug}")
    for split in ("train", "val"):
        print(f"[{split}] objects per class:")
        for class_id in range(args.num_classes):
            name = names[class_id] if class_id < len(names) else f"class_{class_id}"
            print(f"  {class_id} {name:>9}: {counts[split][class_id]}")
    print(f"\nDone. Dataset written to: {args.output}")
    print("Reminder: set train: images/train and val: images/val in configs/data.yaml")


if __name__ == "__main__":
    main()