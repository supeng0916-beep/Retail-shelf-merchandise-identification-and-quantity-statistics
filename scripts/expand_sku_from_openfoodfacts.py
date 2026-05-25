from __future__ import annotations

import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path

import requests

CSV_PATH = Path("my_sku_full.csv")
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_SEARCH_URL_V2 = "https://world.openfoodfacts.org/api/v2/search"

CSV_FIELDS = [
    "sku_id",
    "brand",
    "product_name",
    "variant",
    "size",
    "category",
    "search_keyword",
]

CATEGORY_RULES = [
    ("Noodles", "NOD", ("noodle", "ramen", "mee", "mi goreng", "laksa", "pasta cup")),
    ("Beverages", "BEV", ("drink", "beverage", "coffee", "tea", "juice", "soda", "water", "isotonic", "cola")),
    ("Biscuits", "BIS", ("biscuit", "cookie", "cracker", "wafer")),
    ("Snacks", "SNK", ("snack", "chips", "crisps", "corn snack", "puff")),
    ("Chocolate", "CHO", ("chocolate", "cocoa", "candy", "sweet")),
    ("Dairy", "DAI", ("milk", "yogurt", "yoghurt", "cheese", "dairy")),
]

SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|mg|l|ml|cl|oz|pack|packs|pcs|pc|x\d+)", re.IGNORECASE)


def to_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def clean_text(text: str) -> str:
    text = to_ascii(text or "")
    text = text.replace("|", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_code(text: str, count: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", clean_text(text).upper())
    if not tokens:
        return "UNK"
    pieces = []
    for token in tokens[:count]:
        pieces.append(token[:3])
    return "_".join(pieces)


def normalize_size(quantity: str, name: str) -> str:
    quantity = clean_text(quantity)
    if quantity:
        m = SIZE_PATTERN.search(quantity)
        if m:
            return f"{m.group(1)}{m.group(2).lower()}"
        return quantity[:24]

    m = SIZE_PATTERN.search(name)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    return "1unit"


def infer_category(name: str, categories: str) -> tuple[str, str]:
    text = f"{name} {categories}".lower()
    for label, code, keys in CATEGORY_RULES:
        if any(k in text for k in keys):
            return label, code
    return "Others", "OTR"


def split_product_variant(product_name: str, brand: str, size: str) -> tuple[str, str]:
    name = clean_text(product_name)
    brand_clean = clean_text(brand)
    if brand_clean and name.lower().startswith(brand_clean.lower()):
        name = name[len(brand_clean) :].strip(" -")

    if size and size != "1unit":
        name = re.sub(re.escape(size), "", name, flags=re.IGNORECASE).strip(" -")

    tokens = re.findall(r"[A-Za-z0-9]+", name)
    if not tokens:
        return "Product", "Original"
    if len(tokens) <= 3:
        return " ".join(tokens), "Original"
    return " ".join(tokens[:3]), " ".join(tokens[3:8])


def make_unique_sku_id(base_id: str, used_ids: set[str]) -> str:
    if base_id not in used_ids:
        return base_id
    idx = 2
    while True:
        candidate = f"{base_id}_{idx:02d}"
        if candidate not in used_ids:
            return candidate
        idx += 1


def off_search(country: str, page: int, page_size: int = 100) -> list[dict]:
    headers = {
        "User-Agent": "ShelfSKUBuilder/1.0 (contact: local-script)",
        "Accept": "application/json",
    }
    common_fields = "code,product_name,brands,quantity,categories,image_front_url"

    params_v2 = {
        "countries_tags_en": country,
        "fields": common_fields,
        "page": page,
        "page_size": page_size,
    }
    try:
        resp = requests.get(OFF_SEARCH_URL_V2, params=params_v2, headers=headers, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        products = payload.get("products", [])
        if products:
            return products
    except Exception:
        pass

    params_v1 = {
        "action": "process",
        "json": 1,
        "page": page,
        "page_size": page_size,
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": country,
        "fields": common_fields,
    }
    for url in (OFF_SEARCH_URL, "https://my.openfoodfacts.org/cgi/search.pl"):
        resp = requests.get(url, params=params_v1, headers=headers, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        products = payload.get("products", [])
        if products:
            return products
    return []


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalized_key(brand: str, product_name: str, variant: str, size: str) -> str:
    return "|".join(
        clean_text(x).lower() for x in (brand, product_name, variant, size)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand my_sku_full.csv using Open Food Facts products.")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--target-total", type=int, default=500)
    parser.add_argument("--country", default="malaysia")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in CSV: {csv_path}")

    used_ids = {clean_text(r.get("sku_id", "")).upper() for r in rows if r.get("sku_id")}
    used_keys = {
        normalized_key(
            r.get("brand", ""),
            r.get("product_name", ""),
            r.get("variant", ""),
            r.get("size", ""),
        )
        for r in rows
    }

    needed = max(0, args.target_total - len(rows))
    if needed == 0:
        print(f"Current rows={len(rows)} already >= target={args.target_total}.")
        return

    appended: list[dict] = []

    for page in range(1, args.max_pages + 1):
        if len(appended) >= needed:
            break
        try:
            products = off_search(country=args.country, page=page, page_size=100)
        except Exception as exc:
            print(f"page={page} fetch failed: {exc}")
            continue

        if not products:
            continue

        for p in products:
            if len(appended) >= needed:
                break
            if not p.get("image_front_url"):
                continue

            brand_raw = clean_text((p.get("brands") or "").split(",")[0])
            name_raw = clean_text(p.get("product_name") or "")
            if not brand_raw or not name_raw:
                continue

            size = normalize_size(p.get("quantity", ""), name_raw)
            product_name, variant = split_product_variant(name_raw, brand_raw, size)
            category, cat_code = infer_category(name_raw, p.get("categories") or "")

            key = normalized_key(brand_raw, product_name, variant, size)
            if key in used_keys:
                continue

            brand_code = token_code(brand_raw, count=1)
            prod_code = token_code(product_name, count=1)
            var_code = token_code(variant, count=1)
            base_id = f"{cat_code}_{brand_code}_{prod_code}_{var_code}"
            sku_id = make_unique_sku_id(base_id, used_ids)

            row = {
                "sku_id": sku_id,
                "brand": brand_raw[:40],
                "product_name": product_name[:64],
                "variant": variant[:64] if variant else "Original",
                "size": size[:20],
                "category": category,
                "search_keyword": f"{brand_raw} {name_raw} Malaysia".strip()[:140],
            }
            appended.append(row)
            used_ids.add(sku_id)
            used_keys.add(key)

        time.sleep(max(0.0, args.sleep))

    if not appended:
        print("No new SKU rows were discovered.")
        return

    all_rows = rows + appended
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Original rows: {len(rows)}")
    print(f"Appended rows: {len(appended)}")
    print(f"Final rows:    {len(all_rows)}")


if __name__ == "__main__":
    main()
