from __future__ import annotations
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import shutil

AUGMENTATIONS = (
    "blur",
    "motion",
    "bright",
    "contrast",
    "gamma",
    "noise",
    "hsv",
    "sharp",
    "jpeg",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    kernel = random.choice((3, 5))
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def motion_blur(image: np.ndarray) -> np.ndarray:
    size = random.randint(5, 11)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def brightness(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=random.randint(-40, 40))


def contrast(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=random.uniform(0.7, 1.4), beta=0)


def gamma_correction(image: np.ndarray) -> np.ndarray:
    gamma = random.uniform(0.7, 1.5)
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def gaussian_noise(image: np.ndarray) -> np.ndarray:
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def hsv_shift(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-5, 5)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-20, 20), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def jpeg_compression(image: np.ndarray) -> np.ndarray:
    quality = random.randint(35, 80)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


FUNCTIONS = {
    "blur": gaussian_blur,
    "motion": motion_blur,
    "bright": brightness,
    "contrast": contrast,
    "gamma": gamma_correction,
    "noise": gaussian_noise,
    "hsv": hsv_shift,
    "sharp": sharpen,
    "jpeg": jpeg_compression,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image augmentations.")
    parser.add_argument("--labels-in", type=Path, default=None)
    parser.add_argument("--labels-out", type=Path, default=None)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    processed = 0
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"Warning: unable to read {path}")
            continue

        cv2.imwrite(str(args.output / path.name), image)
        for name, function in FUNCTIONS.items():
            augmented = function(image)
            cv2.imwrite(str(args.output / f"{path.stem}_{name}{path.suffix}"), augmented)

        processed += 1
        if args.labels_in is not None and args.labels_out is not None:
            args.labels_out.mkdir(parents=True, exist_ok=True)
            src_label = args.labels_in / f"{path.stem}.txt"
            if src_label.is_file():
                shutil.copy(src_label, args.labels_out / src_label.name)
                for name in FUNCTIONS:
                    shutil.copy(src_label, args.labels_out / f"{path.stem}_{name}.txt")

    print(f"Processed {processed} source images.")
    print(f"Generated {processed * (1 + len(FUNCTIONS))} images including originals.")


if __name__ == "__main__":
    main()
