from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def read_class_id(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return int(line.split()[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation manifests.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=Path("train.txt"))
    parser.add_argument("--val-output", type=Path, default=Path("val.txt"))
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    by_class: dict[int, list[str]] = defaultdict(list)
    missing_labels = 0
    empty_labels = 0

    image_paths = sorted(
        p for p in args.images.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for image_path in image_paths:
        class_id = read_class_id(args.labels / f"{image_path.stem}.txt")
        if class_id is None:
            label_path = args.labels / f"{image_path.stem}.txt"
            if label_path.exists():
                empty_labels += 1
            else:
                missing_labels += 1
            continue
        by_class[class_id].append(str(image_path.as_posix()))

    train_items: list[str] = []
    val_items: list[str] = []

    for class_id, items in sorted(by_class.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio))
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])
        print(f"class={class_id}: train={len(items)-n_val}, val={n_val}")

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text("\n".join(train_items) + "\n", encoding="utf-8")
    args.val_output.write_text("\n".join(val_items) + "\n", encoding="utf-8")

    print(f"train={len(train_items)} val={len(val_items)}")
    print(f"missing_labels={missing_labels} empty_labels={empty_labels}")


if __name__ == "__main__":
    main()
