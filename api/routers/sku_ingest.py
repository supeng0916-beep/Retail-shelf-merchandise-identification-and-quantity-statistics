from __future__ import annotations

import csv
import hashlib
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.preprocessor import ImagePreprocessor

router = APIRouter(prefix="/sku", tags=["sku-ingest"])

CSV_FIELDS = [
    "sku_id",
    "brand",
    "product_name",
    "variant",
    "size",
    "category",
    "search_keyword",
]

CATEGORY_PREFIX = {
    "beverages": "BEV",
    "noodles": "NOD",
    "biscuits": "BIS",
    "snacks": "SNK",
    "chocolate": "CHO",
    "dairy": "DAI",
    "others": "OTR",
}

INGEST_ROOT = Path("database/ingest_jobs")
JOB_LOCK = threading.RLock()
INGEST_JOBS: dict[str, dict] = {}
MIN_SKU_IMAGES = 3
MAX_SKU_IMAGES = 8
SINGLE_SKU_SIMILARITY_THRESHOLD = 0.78


class PublishRequest(BaseModel):
    job_id: str
    brand: str = Field(default="")
    product_name: str = Field(default="")
    variant: str = Field(default="")
    size: str = Field(default="")
    category: str = Field(default="Others")


def _now() -> float:
    return time.time()


def _job_snapshot(job_id: str) -> dict:
    with JOB_LOCK:
        job = INGEST_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ingest job not found")
        return dict(job)


def _update_job(job_id: str, **fields) -> None:
    with JOB_LOCK:
        if job_id not in INGEST_JOBS:
            return
        INGEST_JOBS[job_id].update(fields)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text.replace("/", " ").replace("|", " ")).strip()
    return text


def _slug_code(text: str, limit: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", _clean_text(text).upper())
    if not tokens:
        return "UNK"
    return "_".join(token[:3] for token in tokens[:limit])


def _compose_name(brand: str, product_name: str, variant: str, size: str) -> str:
    parts = [brand.strip(), product_name.strip(), variant.strip(), size.strip()]
    return " ".join(part for part in parts if part).strip()


def _build_search_keyword(brand: str, product_name: str, variant: str, size: str) -> str:
    return _compose_name(brand, product_name, variant, size)


def _load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _generate_sku_id(csv_path: Path, brand: str, product_name: str, variant: str, category: str) -> str:
    category_code = CATEGORY_PREFIX.get((category or "Others").strip().lower(), "OTR")
    brand_code = _slug_code(brand, 1)
    name_code = _slug_code(product_name, 2)
    variant_code = _slug_code(variant, 1) if variant.strip() else "ORI"
    base = f"{category_code}_{brand_code}_{name_code}_{variant_code}"

    rows = _load_csv_rows(csv_path)
    used = {str(row.get("sku_id", "")).strip().upper() for row in rows}
    candidate = base
    suffix = 2
    while candidate.upper() in used:
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _append_csv_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _iter_existing_image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.rglob(pattern))
    return paths


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _existing_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    if not root.exists():
        return hashes
    for path in _iter_existing_image_paths(root):
        try:
            hashes.add(_md5_file(path))
        except OSError:
            continue
    return hashes


def _suggest_metadata(texts: list[str]) -> dict[str, str]:
    if not texts:
        return {
            "brand": "",
            "product_name": "",
            "ocr_text": "",
        }

    merged = " ".join(texts).strip()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+-]*", merged)
    brand = tokens[0] if tokens else ""
    product_name = " ".join(tokens[:6]).strip()
    return {
        "brand": brand,
        "product_name": product_name,
        "ocr_text": merged,
    }


def _build_validation_result(
    accepted_count: int,
    flagged_files: list[str] | None = None,
    ocr_texts: list[str] | None = None,
) -> tuple[bool, str, str]:
    flagged_files = flagged_files or []
    ocr_texts = [text.strip() for text in (ocr_texts or []) if text.strip()]

    if accepted_count < MIN_SKU_IMAGES:
        return (
            False,
            "too_few",
            f"当前有效图片仅 {accepted_count} 张。请至少上传 {MIN_SKU_IMAGES} 张同一商品的不同角度照片后再发布。",
        )

    if accepted_count > MAX_SKU_IMAGES:
        return (
            False,
            "too_many",
            f"当前有效图片为 {accepted_count} 张，超过上限 {MAX_SKU_IMAGES} 张。请先删除多余图片后重新预分析。",
        )

    if flagged_files:
        message = (
            "系统检测到这批图片里可能混入了不同商品，已禁止发布。"
            f" 请移除可疑图片后重新预分析：{', '.join(flagged_files)}"
        )
        if ocr_texts:
            message += f"。OCR 提示：{' | '.join(ocr_texts[:3])}"
        return False, "mixed_sku_suspected", message

    return (
        True,
        "ok",
        "批次校验通过：当前图片数量符合 3-8 张要求，且内容看起来属于同一商品，可继续确认信息并发布。",
    )


def _validate_single_sku_batch(
    job_dir: Path,
    accepted: list[dict],
    embedder,
    ocr_booster=None,
) -> tuple[list[str], list[str]]:
    if len(accepted) < MIN_SKU_IMAGES or len(accepted) > MAX_SKU_IMAGES:
        return [], []

    embeddings: list[np.ndarray] = []
    flagged_ocr_texts: list[str] = []

    for item in accepted:
        image = cv2.imread(str(job_dir / item["saved_name"]))
        if image is None:
            embeddings.append(np.zeros((embedder.embedding_dim,), dtype=np.float32))
            continue
        embeddings.append(embedder.embed_single(image))

    matrix = np.stack(embeddings, axis=0).astype(np.float32)
    similarity = matrix @ matrix.T
    mean_scores = []
    for index in range(len(accepted)):
        others = np.delete(similarity[index], index)
        mean_scores.append(float(others.mean()) if others.size else 1.0)
    ref_index = int(np.argmax(np.asarray(mean_scores, dtype=np.float32)))

    flagged_files: list[str] = []
    for index, item in enumerate(accepted):
        if index == ref_index:
            continue
        score = float(similarity[ref_index, index])
        if score < SINGLE_SKU_SIMILARITY_THRESHOLD:
            flagged_files.append(item["original_name"])
            if ocr_booster is not None and getattr(ocr_booster, "enabled", False):
                image = cv2.imread(str(job_dir / item["saved_name"]))
                if image is not None:
                    try:
                        texts = ocr_booster.read_texts(image)
                    except Exception:
                        texts = []
                    if texts:
                        flagged_ocr_texts.append(" ".join(texts[:3]))

    return flagged_files, flagged_ocr_texts


@router.post("/ingest-preview")
async def ingest_preview(
    files: list[UploadFile] = File(..., description="Multi-angle product images"),
):
    from api.main import cfg, embedder, ocr_booster

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    sku_dir = Path(cfg["matcher"]["sku_images_dir"])
    existing_hashes = _existing_hashes(sku_dir)
    batch_hashes: set[str] = set()
    job_id = uuid.uuid4().hex[:12]
    job_dir = INGEST_ROOT / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    duplicate_files: list[str] = []
    ocr_texts: list[str] = []

    for index, upload in enumerate(files, start=1):
        file_bytes = await upload.read()
        digest = _md5_bytes(file_bytes)
        if digest in existing_hashes or digest in batch_hashes:
            duplicate_files.append(upload.filename or f"image_{index}")
            continue

        try:
            image = ImagePreprocessor.decode_upload(file_bytes, max_dim=1280)
        except HTTPException:
            duplicate_files.append(upload.filename or f"image_{index}")
            continue

        batch_hashes.add(digest)
        save_path = job_dir / f"{index:04d}.jpg"
        cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if ocr_booster is not None and getattr(ocr_booster, "enabled", False) and len(ocr_texts) < 6:
            try:
                ocr_texts.extend(ocr_booster.read_texts(image))
            except Exception:
                pass

        accepted.append(
            {
                "original_name": upload.filename or save_path.name,
                "saved_name": save_path.name,
                "md5": digest,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }
        )

    suggestion = _suggest_metadata(ocr_texts)
    flagged_files, flagged_ocr_texts = _validate_single_sku_batch(
        job_dir=job_dir,
        accepted=accepted,
        embedder=embedder,
        ocr_booster=ocr_booster,
    )
    is_publishable, validation_code, validation_message = _build_validation_result(
        accepted_count=len(accepted),
        flagged_files=flagged_files,
        ocr_texts=flagged_ocr_texts,
    )

    with JOB_LOCK:
        INGEST_JOBS[job_id] = {
            "job_id": job_id,
            "status": "previewed",
            "phase": "preview",
            "progress": 100,
            "created_at": _now(),
            "updated_at": _now(),
            "job_dir": str(job_dir),
            "accepted": accepted,
            "duplicate_files": duplicate_files,
            "suggestion": suggestion,
            "flagged_files": flagged_files,
            "is_publishable": is_publishable,
            "validation_code": validation_code,
            "validation_message": validation_message,
            "message": validation_message,
        }

    return {
        "job_id": job_id,
        "accepted_count": len(accepted),
        "duplicate_count": len(duplicate_files),
        "files": accepted,
        "duplicate_files": duplicate_files,
        "suggestion": suggestion,
        "is_publishable": is_publishable,
        "validation_code": validation_code,
        "validation_message": validation_message,
        "flagged_files": flagged_files,
    }


async def _process_ingest_job(job_id: str, payload: PublishRequest) -> None:
    from api.main import cfg, embedder, matcher

    try:
        job = _job_snapshot(job_id)
        accepted = list(job.get("accepted", []))
        if not accepted:
            _update_job(job_id, status="failed", phase="publish", message="No accepted files to publish")
            return

        csv_path = Path("my_sku_full.csv")
        brand = payload.brand.strip()
        product_name = payload.product_name.strip()
        variant = payload.variant.strip()
        size = payload.size.strip()
        category = payload.category.strip() or "Others"

        if not product_name:
            product_name = (job.get("suggestion") or {}).get("product_name", "").strip()
        if not brand:
            brand = (job.get("suggestion") or {}).get("brand", "").strip()
        if not product_name:
            raise ValueError("Product name is required")

        sku_id = _generate_sku_id(csv_path, brand, product_name, variant, category)
        display_name = _compose_name(brand, product_name, variant, size) or sku_id
        target_dir = Path(cfg["matcher"]["sku_images_dir"]) / sku_id
        target_dir.mkdir(parents=True, exist_ok=True)

        _update_job(
            job_id,
            status="running",
            phase="embedding",
            progress=5,
            sku_id=sku_id,
            message="Publishing images and generating CLIP features",
            updated_at=_now(),
        )

        start_index = len(list(target_dir.glob("*.jpg")))
        total = len(accepted)

        for idx, item in enumerate(accepted, start=1):
            src_path = Path(job["job_dir"]) / item["saved_name"]
            image = cv2.imread(str(src_path))
            if image is None:
                continue

            out_path = target_dir / f"{start_index + idx:04d}.jpg"
            cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            embedding = embedder.embed_single(image)
            await matcher.add_sku_async(
                sku_id=sku_id,
                name=display_name,
                embedding=embedding,
                barcode=None,
                category=category,
            )
            _update_job(
                job_id,
                progress=10 + int((idx / total) * 80),
                processed=idx,
                total=total,
                updated_at=_now(),
            )

        matcher.save_index()
        _append_csv_row(
            csv_path,
            {
                "sku_id": sku_id,
                "brand": brand,
                "product_name": product_name,
                "variant": variant,
                "size": size,
                "category": category,
                "search_keyword": _build_search_keyword(brand, product_name, variant, size),
            },
        )

        shutil.rmtree(job["job_dir"], ignore_errors=True)
        _update_job(
            job_id,
            status="completed",
            phase="done",
            progress=100,
            message="Publish and sync completed",
            updated_at=_now(),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            phase="error",
            message=str(exc),
            updated_at=_now(),
        )


@router.post("/ingest")
async def ingest_publish(payload: PublishRequest, background_tasks: BackgroundTasks):
    job = _job_snapshot(payload.job_id)
    if job.get("status") != "previewed":
        raise HTTPException(status_code=400, detail="Job is not ready for publish")
    if not bool(job.get("is_publishable")):
        raise HTTPException(
            status_code=400,
            detail=str(job.get("validation_message") or "This batch is blocked and cannot be published."),
        )

    _update_job(
        payload.job_id,
        status="queued",
        phase="queue",
        progress=0,
        message="Queued for publish",
        updated_at=_now(),
    )
    background_tasks.add_task(_process_ingest_job, payload.job_id, payload)
    return {
        "job_id": payload.job_id,
        "status": "queued",
    }


@router.get("/ingest/jobs/{job_id}")
async def ingest_job_status(job_id: str):
    return _job_snapshot(job_id)
