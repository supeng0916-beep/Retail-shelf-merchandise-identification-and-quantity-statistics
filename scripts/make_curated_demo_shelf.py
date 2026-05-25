from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CANVAS_W = 3840
CANVAS_H = 2160
ROWS = 4
ROW_GAP = 42
LEFT_PAD = 90
RIGHT_PAD = 90
TOP_PAD = 120
BOTTOM_PAD = 120
SHELF_DEPTH = 18
LABEL_HEIGHT = 0
OUTPUT_PATH = Path("synthetic_demo_shelf_curated.jpg")

CURATED_IMAGES = [
    "database/sku_images/BEV_100_ORI_ELE/0002.jpg",
    "database/sku_images/BEV_BOH_GRE_ORI/0003.jpg",
    "database/sku_images/BEV_CED_PEA_ORI/0001.jpg",
    "database/sku_images/BEV_CED_PEA_ORI_02/0001.jpg",
    "database/sku_images/BEV_CHE_3_IPO/0002.jpg",
    "database/sku_images/BEV_COC_SPR_ORI/0001.jpg",
    "database/sku_images/BEV_DEL_SWE_STY/0001.jpg",
    "database/sku_images/BEV_EMC_GRA_ORI/0003.jpg",
    "database/sku_images/BEV_GAR_BRE_PEN/0001.jpg",
    "database/sku_images/BEV_GAR_ROT_CLA/0001.jpg",
    "database/sku_images/NOD_A1_EMP_HER/0006.jpg",
    "database/sku_images/NOD_BAG_KAR_5P/0003.jpg",
    "database/sku_images/NOD_MAG_AYM_BER/0002.jpg",
]


@dataclass
class ProductAsset:
    sku_id: str
    path: Path
    image: np.ndarray
    mask: np.ndarray


def crop_subject(image: np.ndarray, white_threshold: int = 242) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fg = gray < white_threshold
    if not np.any(fg):
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        return image, mask

    ys, xs = np.where(fg)
    y1, y2 = max(0, ys.min() - 8), min(image.shape[0], ys.max() + 9)
    x1, x2 = max(0, xs.min() - 8), min(image.shape[1], xs.max() + 9)
    cropped = image[y1:y2, x1:x2]

    crop_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    mask = np.where(crop_gray < white_threshold, 255, 0).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cropped, mask


def load_assets() -> list[ProductAsset]:
    assets: list[ProductAsset] = []
    for raw_path in CURATED_IMAGES:
        path = Path(raw_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        cropped, mask = crop_subject(image)
        assets.append(
            ProductAsset(
                sku_id=path.parent.name,
                path=path,
                image=cropped,
                mask=mask,
            )
        )
    return assets


def build_layout(assets: list[ProductAsset], rng: random.Random) -> list[list[ProductAsset]]:
    groups = assets[:]
    rng.shuffle(groups)

    row_slots = [7, 7, 7, 7]
    rows: list[list[ProductAsset]] = [[] for _ in range(ROWS)]
    row_idx = 0
    remaining_slots = sum(row_slots)

    for asset in groups:
        if remaining_slots <= 0:
            break
        copies = rng.randint(2, 3)
        copies = min(copies, remaining_slots)
        while copies > 0:
            if remaining_slots <= 0:
                break
            if row_slots[row_idx] == 0:
                row_idx = (row_idx + 1) % ROWS
                continue
            rows[row_idx].append(asset)
            row_slots[row_idx] -= 1
            remaining_slots -= 1
            copies -= 1
            if row_slots[row_idx] == 0:
                row_idx = (row_idx + 1) % ROWS

    return rows


def make_background() -> np.ndarray:
    bg = np.full((CANVAS_H, CANVAS_W, 3), 248, dtype=np.uint8)

    gradient = np.linspace(252, 236, CANVAS_H, dtype=np.uint8).reshape(CANVAS_H, 1)
    for c in range(3):
        bg[:, :, c] = np.minimum(bg[:, :, c], gradient)

    usable_h = CANVAS_H - TOP_PAD - BOTTOM_PAD
    row_h = (usable_h - ROW_GAP * (ROWS - 1)) // ROWS

    for row in range(ROWS):
        y = TOP_PAD + row * (row_h + ROW_GAP)
        shelf_y = y + row_h - SHELF_DEPTH
        cv2.rectangle(bg, (LEFT_PAD - 30, shelf_y), (CANVAS_W - RIGHT_PAD + 30, shelf_y + SHELF_DEPTH), (192, 196, 202), -1)
        cv2.line(bg, (LEFT_PAD - 30, shelf_y), (CANVAS_W - RIGHT_PAD + 30, shelf_y), (150, 154, 160), 2)
        cv2.line(bg, (LEFT_PAD - 30, shelf_y + SHELF_DEPTH), (CANVAS_W - RIGHT_PAD + 30, shelf_y + SHELF_DEPTH), (228, 230, 233), 2)

    return bg


def resize_asset(asset: ProductAsset, target_h: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    jitter = rng.uniform(0.94, 1.06)
    h, w = asset.image.shape[:2]
    scale = (target_h * jitter) / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = cv2.resize(asset.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(asset.mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    brightness = rng.uniform(0.94, 1.06)
    image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return image, mask


def alpha_paste(canvas: np.ndarray, image: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return

    y1 = max(0, y)
    x1 = max(0, x)
    y2 = min(canvas.shape[0], y + h)
    x2 = min(canvas.shape[1], x + w)
    if y1 >= y2 or x1 >= x2:
        return

    img_y1 = y1 - y
    img_x1 = x1 - x
    img_y2 = img_y1 + (y2 - y1)
    img_x2 = img_x1 + (x2 - x1)

    roi = canvas[y1:y2, x1:x2]
    fg = image[img_y1:img_y2, img_x1:img_x2].astype(np.float32)
    alpha = (mask[img_y1:img_y2, img_x1:img_x2].astype(np.float32) / 255.0)[..., None]
    blended = fg * alpha + roi.astype(np.float32) * (1.0 - alpha)
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)


def add_shadow(canvas: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = mask.shape[:2]
    shadow = np.zeros((h + 20, w + 20), dtype=np.uint8)
    shadow[10 : 10 + h, 10 : 10 + w] = mask
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=10, sigmaY=10)

    shadow_img = np.zeros((shadow.shape[0], shadow.shape[1], 3), dtype=np.uint8)
    alpha_paste(canvas, shadow_img, (shadow * 0.22).astype(np.uint8), x - 10, y - 2)


def render(rows: list[list[ProductAsset]], rng: random.Random) -> np.ndarray:
    canvas = make_background()
    usable_h = CANVAS_H - TOP_PAD - BOTTOM_PAD
    row_h = (usable_h - ROW_GAP * (ROWS - 1)) // ROWS
    product_h = row_h - SHELF_DEPTH - 24 - LABEL_HEIGHT
    cell_w = (CANVAS_W - LEFT_PAD - RIGHT_PAD) // 7

    for row_idx, row_assets in enumerate(rows):
        y_top = TOP_PAD + row_idx * (row_h + ROW_GAP)
        shelf_y = y_top + row_h - SHELF_DEPTH
        x_cursor = LEFT_PAD + rng.randint(0, 20)

        for asset in row_assets:
            image, mask = resize_asset(asset, product_h, rng)
            h, w = image.shape[:2]
            x_jitter = rng.randint(-10, 12)
            x = x_cursor + x_jitter
            y = shelf_y - h - rng.randint(0, 8)

            add_shadow(canvas, mask, x, y)
            alpha_paste(canvas, image, mask, x, y)

            x_cursor += max(int(cell_w * rng.uniform(0.82, 0.98)), w + rng.randint(18, 34))

    return canvas


def main() -> None:
    rng = random.Random(20260423)
    assets = load_assets()
    rows = build_layout(assets, rng)
    canvas = render(rows, rng)

    ok = cv2.imwrite(str(OUTPUT_PATH), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError(f"Failed to save image: {OUTPUT_PATH}")

    print(f"Saved curated synthetic shelf: {OUTPUT_PATH}")
    for i, row in enumerate(rows, start=1):
        print(f"Row {i}: {' | '.join(asset.sku_id for asset in row)}")


if __name__ == "__main__":
    main()
