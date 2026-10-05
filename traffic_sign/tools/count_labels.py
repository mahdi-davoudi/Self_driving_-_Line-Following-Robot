from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

CLASS_NAMES = {
    0: "stop",
    1: "right",
    2: "left",
    3: "straight",
    4: "park",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Count YOLO annotation objects by class.")
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[int] = Counter()
    for path in args.labels.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counts[int(line.split()[0])] += 1

    total = sum(counts.values())
    for class_id in sorted(counts):
        print(f"{class_id:>2} {CLASS_NAMES.get(class_id, f'class_{class_id}'):>10}: {counts[class_id]}")
    print(f"total objects: {total}")


if __name__ == "__main__":
    main()
