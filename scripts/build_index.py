"""
Rebuild FAISS index with synthetic shelf-style augmentation.

Usage:
  python scripts/build_index.py
  python scripts/build_index.py --aug-per-image 20
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def compose_display_name(row: dict) -> str:
    parts = [
        (row.get("brand") or "").strip(),
        (row.get("product_name") or "").strip(),
        (row.get("variant") or "").strip(),
        (row.get("size") or "").strip(),
    ]
    text = " ".join(p for p in parts if p)
    return " ".join(text.split())


def load_sku_meta(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        sku_id = (row.get("sku_id") or "").strip()
        if not sku_id:
            continue
        name = compose_display_name(row) or sku_id
        category = (row.get("category") or "").strip() or None
        meta[sku_id] = {"name": name, "category": category}
    return meta


class SyntheticAugmentor:
    """Generate shelf-like domain-shift variants from clean product images."""

    def __init__(self, variants_per_image: int = 20) -> None:
        self.variants_per_image = max(1, int(variants_per_image))

    def _random_affine(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w * 0.5, h * 0.5)
        angle = float(rng.uniform(-22, 22))
        scale = float(rng.uniform(0.9, 1.08))
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        tx = float(rng.uniform(-0.06, 0.06) * w)
        ty = float(rng.uniform(-0.06, 0.06) * h)
        mat[:, 2] += (tx, ty)
        return cv2.warpAffine(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _random_perspective(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        max_jitter = 0.08
        dst = src.copy()
        jitter = np.array(
            [
                [rng.uniform(-max_jitter, max_jitter) * w, rng.uniform(-max_jitter, max_jitter) * h],
                [rng.uniform(-max_jitter, max_jitter) * w, rng.uniform(-max_jitter, max_jitter) * h],
                [rng.uniform(-max_jitter, max_jitter) * w, rng.uniform(-max_jitter, max_jitter) * h],
                [rng.uniform(-max_jitter, max_jitter) * w, rng.uniform(-max_jitter, max_jitter) * h],
            ],
            dtype=np.float32,
        )
        dst += jitter
        mat = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _add_gaussian_noise(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sigma = float(rng.uniform(6, 20))
        noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
        out = image.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def _add_highlight(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        out = image.astype(np.float32)
        overlay = np.zeros((h, w), dtype=np.float32)
        num_spots = int(rng.integers(1, 4))
        for _ in range(num_spots):
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (
                int(rng.integers(max(8, w // 12), max(12, w // 3))),
                int(rng.integers(max(8, h // 12), max(12, h // 3))),
            )
            angle = float(rng.uniform(0, 180))
            intensity = float(rng.uniform(0.25, 0.6))
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, center, axes, angle, 0, 360, intensity, -1)
            overlay = np.maximum(overlay, mask)

        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=9, sigmaY=9)
        out = out + overlay[..., None] * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _add_uneven_lighting(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        angle = float(rng.uniform(0, np.pi * 2.0))
        grad = np.cos(angle) * xx + np.sin(angle) * yy
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        amp = float(rng.uniform(0.35, 0.75))
        lighting = 1.0 - amp + grad * amp
        out = image.astype(np.float32) * lighting[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _motion_blur(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = int(rng.choice([3, 5, 7, 9, 11]))
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        return cv2.filter2D(image, -1, kernel)

    def _augment_once(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = image
        if rng.random() < 0.95:
            out = self._random_affine(out, rng)
        if rng.random() < 0.70:
            out = self._random_perspective(out, rng)
        if rng.random() < 0.85:
            out = self._add_uneven_lighting(out, rng)
        if rng.random() < 0.75:
            out = self._add_highlight(out, rng)
        if rng.random() < 0.85:
            out = self._add_gaussian_noise(out, rng)
        if rng.random() < 0.65:
            out = self._motion_blur(out, rng)
        return out

    def generate(self, image: np.ndarray, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        variants = [image]
        for _ in range(self.variants_per_image):
            variants.append(self._augment_once(image, rng))
        return variants


def parse_args() -> argparse.Namespace:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--sku-dir", default=cfg["matcher"].get("sku_images_dir", "database/sku_images"))
    parser.add_argument("--index-out", default=cfg["matcher"].get("index_path", "index/faiss_clip.index"))
    parser.add_argument("--db", default=cfg["matcher"].get("db_path", "sku_runtime.db"))
    parser.add_argument("--csv", default="my_sku_full.csv")
    parser.add_argument("--device", default=cfg["embedder"].get("device", "cuda:0"))
    parser.add_argument("--model-name", default=cfg["embedder"].get("model_name", "openai/clip-vit-base-patch32"))
    parser.add_argument("--aug-per-image", type=int, default=20, help="Synthetic variants per source image")
    parser.add_argument("--seed", type=int, default=20260422)
    return parser.parse_args()


def iter_image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(folder.glob(pattern)))
    return files


def count_source_images(sku_dir: Path) -> tuple[int, int]:
    sku_count = 0
    image_count = 0
    for folder in sorted(sku_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = iter_image_files(folder)
        if not images:
            continue
        sku_count += 1
        image_count += len(images)
    return sku_count, image_count


def rebuild_index(
    sku_dir: Path,
    db_path: Path,
    index_out: Path,
    embedder,
    sku_meta: dict[str, dict[str, str]],
    aug_per_image: int,
    seed: int,
) -> tuple[int, int, int]:
    index_out.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")

    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS skus (
            sku_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            barcode TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sku_embeddings (
            faiss_idx INTEGER PRIMARY KEY,
            sku_id TEXT NOT NULL,
            FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
        );
        """
    )

    if sku_meta:
        con.executemany(
            """
            INSERT INTO skus (sku_id, name, barcode, category)
            VALUES (?, ?, NULL, ?)
            ON CONFLICT(sku_id) DO UPDATE SET
                name=excluded.name,
                category=COALESCE(excluded.category, skus.category)
            """,
            [(sid, item["name"], item["category"]) for sid, item in sku_meta.items()],
        )
    con.execute("DELETE FROM sku_embeddings")
    con.commit()

    augmentor = SyntheticAugmentor(variants_per_image=aug_per_image)
    index = faiss.IndexFlatIP(int(embedder.embedding_dim))

    faiss_idx = 0
    source_count = 0
    variant_count = 0

    sku_folders = [d for d in sorted(sku_dir.iterdir()) if d.is_dir()]
    for s_idx, sku_folder in enumerate(sku_folders, 1):
        sku_id = sku_folder.name
        image_files = iter_image_files(sku_folder)
        if not image_files:
            continue

        for i_idx, image_path in enumerate(image_files):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            source_count += 1

            variants = augmentor.generate(
                image=image,
                seed=seed + s_idx * 10_000 + i_idx,
            )

            for var in variants:
                emb = embedder.embed_single(var)
                vec = emb.reshape(1, -1).astype(np.float32)
                index.add(vec)

                con.execute(
                    "INSERT INTO sku_embeddings (faiss_idx, sku_id) VALUES (?, ?)",
                    (faiss_idx, sku_id),
                )
                faiss_idx += 1
                variant_count += 1

        if s_idx % 20 == 0:
            con.commit()
            print(f"  progress: {s_idx}/{len(sku_folders)} sku folders")

    con.commit()
    con.close()

    faiss.write_index(index, str(index_out))
    return index.ntotal, source_count, variant_count


def main() -> None:
    args = parse_args()

    sku_dir = Path(args.sku_dir)
    if not sku_dir.exists():
        print(f"ERROR: SKU image directory not found: {sku_dir}")
        print(f"Expected structure: {sku_dir}/{{sku_id}}/001.jpg")
        sys.exit(1)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP embedder...")
    from src.embedder import SKUEmbedder

    embedder = SKUEmbedder(device=args.device, model_name=args.model_name)

    sku_meta = load_sku_meta(Path(args.csv))
    sku_count, source_image_count = count_source_images(sku_dir)
    estimated_vectors = source_image_count * (args.aug_per_image + 1)

    print(f"\nRebuilding index from: {sku_dir}")
    print(f"CSV metadata rows: {len(sku_meta)}")
    print(f"SKU folders with images: {sku_count}")
    print(f"Source images found: {source_image_count}")
    print(f"Synthetic variants per image: {args.aug_per_image}")
    print(f"Estimated vectors after augmentation: {estimated_vectors}")

    t0 = time.perf_counter()
    vectors, source_images, total_variants = rebuild_index(
        sku_dir=sku_dir,
        db_path=db_path,
        index_out=Path(args.index_out),
        embedder=embedder,
        sku_meta=sku_meta,
        aug_per_image=args.aug_per_image,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    print("\nIndex build completed")
    print(f"  source images: {source_images}")
    print(f"  variants (including originals): {total_variants}")
    print(f"  vectors: {vectors}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  index file: {args.index_out}")
    print(f"  db file: {db_path}")


if __name__ == "__main__":
    main()
