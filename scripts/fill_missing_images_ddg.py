from __future__ import annotations

import argparse
import csv
import hashlib
import io
import time
from pathlib import Path

import requests
from duckduckgo_search import DDGS
from PIL import Image, UnidentifiedImageError

CSV_PATH = Path("my_sku_full.csv")
DEST_ROOT = Path("database/sku_images")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def existing_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def next_idx(folder: Path) -> int:
    idx = 1
    if not folder.exists():
        return idx
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.isdigit():
            idx = max(idx, int(p.stem) + 1)
    return idx


def save_jpeg(content: bytes, path: Path) -> bool:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        return False
    except Exception:
        return False
    w, h = image.size
    if w < 180 or h < 180:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92)
    return True


def download(url: str, timeout: int = 20) -> bytes | None:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
    except Exception:
        return None
    if resp.status_code != 200 or len(resp.content) < 2500:
        return None
    return resp.content


def query_variants(keyword: str) -> list[str]:
    key = " ".join((keyword or "").split())
    if not key:
        return []
    return [
        f"{key} Malaysia product",
        f"{key} Malaysia supermarket shelf",
        f"{key} official front view",
        key,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing SKU images via DuckDuckGo image search.")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--dest-root", default=str(DEST_ROOT))
    parser.add_argument("--target-images", type=int, default=3)
    parser.add_argument("--max-skus", type=int, default=0, help="0 = no limit")
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv_path).open(encoding="utf-8-sig", newline="")))
    root = Path(args.dest_root)

    needed = []
    for row in rows:
        sku_id = (row.get("sku_id") or "").strip()
        if not sku_id:
            continue
        c = existing_count(root / sku_id)
        if c < args.target_images:
            needed.append(row)

    if args.max_skus > 0:
        needed = needed[: args.max_skus]

    print(f"Need fill: {len(needed)} SKUs (target={args.target_images})")
    done = 0
    fail = 0

    with DDGS() as ddgs:
        for idx, row in enumerate(needed, 1):
            sku_id = row["sku_id"]
            keyword = row.get("search_keyword", "").strip()
            dst = root / sku_id
            have = existing_count(dst)
            want = args.target_images - have
            if want <= 0:
                print(f"[{idx}/{len(needed)}] {sku_id:<35s} skip(existing)")
                continue

            saved = 0
            seen_url: set[str] = set()
            seen_hash: set[str] = set()
            image_idx = next_idx(dst)
            queries = query_variants(keyword)

            print(f"[{idx}/{len(needed)}] {sku_id:<35s} ", end="", flush=True)
            for q in queries:
                if saved >= want:
                    break
                try:
                    results = list(ddgs.images(q, region="my-en", safesearch="off", max_results=30))
                except Exception:
                    continue
                for r in results:
                    if saved >= want:
                        break
                    url = (r.get("image") or "").strip()
                    if not url or url in seen_url:
                        continue
                    seen_url.add(url)

                    content = download(url)
                    if content is None:
                        continue
                    digest = hashlib.sha1(content).hexdigest()
                    if digest in seen_hash:
                        continue
                    seen_hash.add(digest)
                    if not save_jpeg(content, dst / f"{image_idx:04d}.jpg"):
                        continue
                    image_idx += 1
                    saved += 1
                    time.sleep(max(0.0, args.sleep))

            if saved >= want:
                done += 1
                print(f"ok(saved={saved})")
            else:
                fail += 1
                print(f"fail(saved={saved})")

    print("")
    print("Run summary")
    print(f"  processed: {len(needed)}")
    print(f"  success:   {done}")
    print(f"  failed:    {fail}")


if __name__ == "__main__":
    main()
