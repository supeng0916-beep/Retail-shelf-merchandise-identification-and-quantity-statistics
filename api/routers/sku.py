from __future__ import annotations
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, File, Form, Query, UploadFile
from loguru import logger

from src.preprocessor import ImagePreprocessor
from src.schemas import SKUItem, SKUListResponse

router = APIRouter(prefix="/sku", tags=["sku-management"])


def _db_uri(db_path: str) -> str:
    return f"file:{Path(db_path).as_posix()}?mode=rwc"


@router.post("/add", summary="新增 SKU 样张")
async def add_sku(
    sku_id: str = Form(..., description="SKU 唯一 ID，如 cola_355ml"),
    name: str = Form(..., description="商品名称"),
    barcode: str | None = Form(None, description="条形码"),
    category: str | None = Form(None, description="商品分类"),
    image: UploadFile = File(..., description="商品正面样张（JPEG/PNG）"),
):
    """
    上传 SKU 样张 + 元数据。
    自动提取 CLIP embedding，热更新 FAISS 索引（无需重启服务）。
    """
    from api.main import cfg, embedder, matcher

    file_bytes = await image.read()
    img = ImagePreprocessor.decode_upload(file_bytes, max_dim=512)

    # 提取 embedding
    embedding = embedder.embed_single(img)

    # 保存样张图片
    images_dir = Path(cfg["matcher"]["sku_images_dir"]) / sku_id
    images_dir.mkdir(parents=True, exist_ok=True)
    existing = list(images_dir.glob("*.jpg"))
    img_path = images_dir / f"{len(existing):03d}.jpg"
    import cv2
    cv2.imwrite(str(img_path), img)

    # 热更新索引
    await matcher.add_sku_async(
        sku_id=sku_id,
        name=name,
        embedding=embedding,
        barcode=barcode,
        category=category,
    )
    matcher.save_index()

    return {
        "success": True,
        "sku_id": sku_id,
        "name": name,
        "index_size": matcher.index_size,
    }


@router.get("/list", response_model=SKUListResponse, summary="列出所有 SKU")
async def list_skus(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    from api.main import cfg
    db_path = cfg["matcher"]["db_path"]

    async with aiosqlite.connect(_db_uri(db_path), uri=True) as db:
        await db.execute("PRAGMA journal_mode=MEMORY")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA temp_store=MEMORY")
        async with db.execute("SELECT COUNT(*) FROM skus") as cur:
            total = (await cur.fetchone())[0]
        offset = (page - 1) * page_size
        async with db.execute(
            """SELECT s.sku_id, s.name, s.barcode, s.category,
                      COUNT(e.faiss_idx) as image_count
               FROM skus s
               LEFT JOIN sku_embeddings e ON s.sku_id = e.sku_id
               GROUP BY s.sku_id
               ORDER BY s.created_at DESC
               LIMIT ? OFFSET ?""",
            (page_size, offset),
        ) as cur:
            rows = await cur.fetchall()

    items = [
        SKUItem(
            sku_id=r[0], name=r[1], barcode=r[2], category=r[3], image_count=r[4]
        )
        for r in rows
    ]
    return SKUListResponse(total=total, items=items)


@router.delete("/{sku_id}", summary="删除 SKU（需重建索引）")
async def delete_sku(sku_id: str):
    from api.main import cfg
    db_path = cfg["matcher"]["db_path"]

    async with aiosqlite.connect(_db_uri(db_path), uri=True) as db:
        await db.execute("PRAGMA journal_mode=MEMORY")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA temp_store=MEMORY")
        await db.execute("DELETE FROM sku_embeddings WHERE sku_id = ?", (sku_id,))
        await db.execute("DELETE FROM skus WHERE sku_id = ?", (sku_id,))
        await db.commit()

    # 删除样张图片
    images_dir = Path(cfg["matcher"]["sku_images_dir"]) / sku_id
    if images_dir.exists():
        import shutil
        shutil.rmtree(images_dir)

    logger.info(f"SKU 已删除: {sku_id}，请调用 /sku/rebuild 重建索引")
    return {"success": True, "message": f"{sku_id} 已删除，请调用 /api/v1/sku/rebuild 重建索引"}


@router.post("/rebuild", summary="全量重建 FAISS 索引")
async def rebuild_index():
    """扫描 database/sku_images/ 重建 FAISS 索引，适用于批量删改后。"""
    from api.main import cfg, embedder, matcher

    count = await matcher.rebuild_index_async(
        embedder=embedder,
        sku_images_dir=cfg["matcher"]["sku_images_dir"],
    )
    return {"success": True, "index_size": count, "message": f"索引重建完成，共 {count} 个向量"}
