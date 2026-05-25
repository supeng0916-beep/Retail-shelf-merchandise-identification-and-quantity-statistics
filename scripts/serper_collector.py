from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError


DEFAULT_API_KEY = "a92bbcfcd4fd0988b5fee8e454c9050893d5069b"
API_URL = "https://google.serper.dev/images"
CSV_PATH = Path("my_sku_full.csv")
DEST_ROOT = Path("database/sku_images")
LOG_FILE = Path("failed_skus.log")

SEARCH_SUFFIXES = (
    "official front view",
    "real life shot",
    "supermarket shelf",
)
PRIORITY_DOMAINS = (
    "shopee",
    "lotuss",
    "lazada",
    "tesco",
    "jaya grocer",
    "mamee",
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def existing_image_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_skus(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cleaned: list[dict] = []
    for row in rows:
        sku_id = (row.get("sku_id") or "").strip()
        if not sku_id:
            continue
        row["sku_id"] = sku_id
        row["search_keyword"] = (row.get("search_keyword") or "").strip()
        cleaned.append(row)
    return cleaned


def sanitize_query(text: str, max_len: int = 100) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("'", " ").replace('"', " ")
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
    return text.strip()


def search_images(api_key: str, query: str, gl: str, hl: str, num: int) -> list[dict]:
    resp = requests.post(
        API_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": int(num), "gl": gl, "hl": hl},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("images", [])


def candidate_score(item: dict, suffix_rank: int) -> float:
    url = (item.get("imageUrl") or "").lower()
    title = (item.get("title") or "").lower()
    source = (item.get("source") or "").lower()
    width = int(item.get("imageWidth") or 0)
    height = int(item.get("imageHeight") or 0)

    score = 0.0
    if any(dom in url or dom in source for dom in PRIORITY_DOMAINS):
        score += 30.0
    if "official" in title:
        score += 8.0
    if width >= 500 and height >= 500:
        score += 6.0
    if width >= 300 and height >= 300:
        score += 3.0
    score -= suffix_rank * 1.5
    return score


def collect_candidates(
    api_key: str,
    search_keyword: str,
    gl: str,
    hl: str,
    per_query_num: int,
    log_file: Path,
) -> dict[str, list[dict]]:
    by_suffix: dict[str, list[dict]] = defaultdict(list)
    seen_url: set[str] = set()
    had_success = False

    for suffix_rank, suffix in enumerate(SEARCH_SUFFIXES):
        query = sanitize_query(f"{search_keyword} {suffix}".strip())
        if not query:
            continue
        try:
            items = search_images(api_key=api_key, query=query, gl=gl, hl=hl, num=per_query_num)
        except Exception as e:
            append_log(log_file, f"SEARCH_FAIL | query={query} | error={e}")
            continue

        ranked: list[dict] = []
        for item in items:
            url = (item.get("imageUrl") or "").strip()
            if not url or url in seen_url:
                continue
            seen_url.add(url)
            ranked.append(
                {
                    "url": url,
                    "title": item.get("title") or "",
                    "source": item.get("source") or "",
                    "score": candidate_score(item, suffix_rank=suffix_rank),
                    "suffix": suffix,
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        by_suffix[suffix] = ranked
        if ranked:
            had_success = True

    # Fallback: try a simpler base query once when all suffix queries fail.
    if not had_success:
        base_query = sanitize_query(search_keyword)
        if base_query:
            try:
                items = search_images(api_key=api_key, query=base_query, gl=gl, hl=hl, num=max(8, per_query_num))
                ranked = []
                for item in items:
                    url = (item.get("imageUrl") or "").strip()
                    if not url or url in seen_url:
                        continue
                    seen_url.add(url)
                    ranked.append(
                        {
                            "url": url,
                            "title": item.get("title") or "",
                            "source": item.get("source") or "",
                            "score": candidate_score(item, suffix_rank=0),
                            "suffix": "fallback",
                        }
                    )
                ranked.sort(key=lambda x: x["score"], reverse=True)
                if ranked:
                    by_suffix[SEARCH_SUFFIXES[0]] = ranked
            except Exception as e:
                append_log(log_file, f"SEARCH_FAIL | query={base_query} | error={e}")

    return by_suffix


def download_image(url: str, timeout: int = 20) -> bytes | None:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
    except Exception:
        return None
    if resp.status_code != 200 or len(resp.content) < 3000:
        return None
    return resp.content


def save_jpeg(content: bytes, save_path: Path) -> tuple[bool, str]:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        return False, "not_image"
    except Exception:
        return False, "decode_error"

    w, h = image.size
    if w < 180 or h < 180:
        return False, "too_small"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path, format="JPEG", quality=92)
    return True, "ok"


def next_image_index(folder: Path) -> int:
    idx = 1
    if not folder.exists():
        return idx
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.stem.isdigit():
                idx = max(idx, int(p.stem) + 1)
    return idx


def download_for_sku(
    row: dict,
    api_key: str,
    dest_root: Path,
    target_images: int,
    gl: str,
    hl: str,
    per_query_num: int,
    sleep_s: float,
    log_file: Path,
) -> tuple[int, int]:
    sku_id = row["sku_id"]
    keyword = row.get("search_keyword", "").strip()
    if not keyword:
        append_log(log_file, f"SKU_FAIL | {sku_id} | empty search_keyword")
        return 0, 0

    dst = dest_root / sku_id
    existing = existing_image_count(dst)
    if existing > 0:
        return -1, existing

    by_suffix = collect_candidates(
        api_key=api_key,
        search_keyword=keyword,
        gl=gl,
        hl=hl,
        per_query_num=per_query_num,
        log_file=log_file,
    )

    suffix_queues = {k: list(v) for k, v in by_suffix.items()}
    hashes: set[str] = set()
    saved = 0
    failures = 0
    image_idx = next_image_index(dst)

    while saved < target_images:
        progress = False
        for suffix in SEARCH_SUFFIXES:
            queue = suffix_queues.get(suffix, [])
            while queue:
                cand = queue.pop(0)
                content = download_image(cand["url"])
                if content is None:
                    failures += 1
                    continue
                digest = hashlib.sha1(content).hexdigest()
                if digest in hashes:
                    continue
                hashes.add(digest)

                ok, _ = save_jpeg(content, dst / f"{image_idx:04d}.jpg")
                if not ok:
                    failures += 1
                    continue

                saved += 1
                image_idx += 1
                progress = True
                time.sleep(max(0.0, sleep_s))
                break
            if saved >= target_images:
                break
        if not progress:
            break

    if saved == 0:
        append_log(log_file, f"SKU_FAIL | {sku_id} | no images saved")
    elif saved < target_images:
        append_log(log_file, f"SKU_WARN | {sku_id} | saved={saved} target={target_images}")

    return saved, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SKU images from Serper (Malaysia-focused)")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--dest-root", default=str(DEST_ROOT))
    parser.add_argument("--log-file", default=str(LOG_FILE))
    parser.add_argument("--api-key", default=os.getenv("SERPER_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--gl", default="my", help="Country code for search")
    parser.add_argument("--hl", default="en", help="Language for search")
    parser.add_argument("--per-query-num", type=int, default=12)
    parser.add_argument("--target-images", type=int, default=6, help="Target images per new SKU")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=15, help="How many SKU rows to process")
    parser.add_argument("--sleep", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing Serper API key. Set --api-key or SERPER_API_KEY.")
    if args.target_images < 1:
        raise ValueError("--target-images must be >= 1")

    csv_path = Path(args.csv_path)
    dest_root = Path(args.dest_root)
    log_file = Path(args.log_file)
    log_file.write_text("", encoding="utf-8")

    rows = load_skus(csv_path)
    if not rows:
        print("No SKU rows found.")
        return

    begin = max(0, int(args.offset))
    if args.limit > 0:
        selected = rows[begin : begin + int(args.limit)]
    else:
        selected = rows[begin:]

    print(f"Total rows in CSV: {len(rows)}")
    print(f"Selected rows: {len(selected)} (offset={begin}, limit={args.limit})")
    print(f"Destination root: {dest_root}")
    print(f"Country lock: gl={args.gl}, language={args.hl}")
    print(f"Suffix rotation: {', '.join(SEARCH_SUFFIXES)}")
    print("")

    done_skip = 0
    done_ok = 0
    done_fail = 0
    saved_total = 0

    for i, row in enumerate(selected, 1):
        sku_id = row["sku_id"]
        print(f"[{i}/{len(selected)}] {sku_id:<35s} ", end="", flush=True)
        saved, failures = download_for_sku(
            row=row,
            api_key=args.api_key,
            dest_root=dest_root,
            target_images=args.target_images,
            gl=args.gl,
            hl=args.hl,
            per_query_num=args.per_query_num,
            sleep_s=args.sleep,
            log_file=log_file,
        )
        if saved == -1:
            done_skip += 1
            print("skip(existing)")
            continue
        if saved > 0:
            done_ok += 1
            saved_total += saved
            print(f"ok(saved={saved}, fail_try={failures})")
        else:
            done_fail += 1
            print("fail")

    print("")
    print("Run summary")
    print(f"  processed: {len(selected)}")
    print(f"  success:   {done_ok}")
    print(f"  skipped:   {done_skip}")
    print(f"  failed:    {done_fail}")
    print(f"  saved:     {saved_total}")
    print(f"  log:       {log_file}")


if __name__ == "__main__":
    main()
