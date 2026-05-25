from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synthetic shelf demo image from SKU reference images."
    )
    parser.add_argument(
        "--sku-dir",
        default="database/sku_images",
        help="Directory containing per-SKU image folders.",
    )
    parser.add_argument(
        "--output",
        default="synthetic_demo_shelf.jpg",
        help="Output image path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of SKUs to place. Default: random between 20 and 30.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260423,
        help="Random seed for reproducible layouts.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3840,
        help="Canvas width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2160,
        help="Canvas height in pixels.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of grid rows.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of grid columns.",
    )
    parser.add_argument(
        "--product-height",
        type=int,
        default=400,
        help="Target product height inside each cell.",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        default=True,
        help="Draw SKU ids below each item.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_false",
        dest="with_labels",
        help="Disable SKU id labels.",
    )
    return parser.parse_args()


def iter_image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(folder.glob(pattern)))
    return files


def load_candidates(sku_dir: Path) -> list[tuple[str, list[Path]]]:
    candidates: list[tuple[str, list[Path]]] = []
    for folder in sorted(sku_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = iter_image_files(folder)
        if images:
            candidates.append((folder.name, images))
    return candidates


def trim_white_border(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    if image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return image
    x, y, w, h = cv2.boundingRect(points)
    return image[y : y + h, x : x + w]


def apply_brightness_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    factor = rng.uniform(0.90, 1.10)
    out = image.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    target_height: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    scale = target_height / float(h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))

    if resized_w > max_width:
        scale = max_width / float(w)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))

    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


def draw_label(
    canvas: np.ndarray,
    text: str,
    x_center: int,
    y_top: int,
    max_width: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    margin = 6

    while scale > 0.35:
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        if text_w <= max_width - margin * 2:
            x = x_center - text_w // 2
            y = y_top + text_h + baseline
            cv2.rectangle(
                canvas,
                (x - margin, y_top),
                (x + text_w + margin, y + baseline + margin),
                (255, 255, 255),
                -1,
            )
            cv2.putText(canvas, text, (x, y), font, scale, (30, 30, 30), thickness, cv2.LINE_AA)
            return
        scale -= 0.05


def build_demo_image(
    selected: list[tuple[str, Path]],
    width: int,
    height: int,
    rows: int,
    cols: int,
    product_height: int,
    with_labels: bool,
    rng: random.Random,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    pad_x = 80
    pad_y = 80
    gap_x = 30
    gap_y = 40
    cell_w = (width - pad_x * 2 - gap_x * (cols - 1)) // cols
    cell_h = (height - pad_y * 2 - gap_y * (rows - 1)) // rows
    label_h = 52 if with_labels else 0
    usable_h = max(60, min(product_height, cell_h - label_h - 16))
    usable_w = max(60, cell_w - 24)

    for idx, (sku_id, image_path) in enumerate(selected):
        row = idx // cols
        col = idx % cols
        if row >= rows:
            break

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = trim_white_border(image)
        image = apply_brightness_jitter(image, rng)
        image = resize_to_fit(image, max_width=usable_w, target_height=usable_h)
        img_h, img_w = image.shape[:2]

        cell_x = pad_x + col * (cell_w + gap_x)
        cell_y = pad_y + row * (cell_h + gap_y)
        x = cell_x + (cell_w - img_w) // 2
        y = cell_y + 8

        canvas[y : y + img_h, x : x + img_w] = image

        if with_labels:
            draw_label(
                canvas=canvas,
                text=sku_id,
                x_center=cell_x + cell_w // 2,
                y_top=cell_y + img_h + 16,
                max_width=cell_w,
            )

    return canvas


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    sku_dir = Path(args.sku_dir)
    if not sku_dir.exists():
        raise FileNotFoundError(f"SKU directory not found: {sku_dir}")

    candidates = load_candidates(sku_dir)
    if not candidates:
        raise RuntimeError(f"No SKU images found under: {sku_dir}")

    capacity = args.rows * args.cols
    if args.count > 0:
        count = args.count
    else:
        count = rng.randint(20, 30)
    count = max(1, min(count, capacity, len(candidates)))

    selected_skus = rng.sample(candidates, count)
    selected_items: list[tuple[str, Path]] = []
    for sku_id, image_paths in selected_skus:
        selected_items.append((sku_id, rng.choice(image_paths)))

    canvas = build_demo_image(
        selected=selected_items,
        width=args.width,
        height=args.height,
        rows=args.rows,
        cols=args.cols,
        product_height=args.product_height,
        with_labels=args.with_labels,
        rng=rng,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved synthetic shelf: {output_path}")
    print(f"Canvas: {args.width}x{args.height}")
    print(f"Selected SKUs: {count}")
    for sku_id, image_path in selected_items:
        print(f"  {sku_id} <- {image_path.name}")


if __name__ == "__main__":
    main()
