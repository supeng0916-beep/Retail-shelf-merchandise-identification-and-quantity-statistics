from __future__ import annotations
import asyncio
import base64
import time
from typing import TypeVar

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.preprocessor import ImagePreprocessor
from src.schemas import BoundingBox, DetectResponse

router = APIRouter(tags=["detection"])
T = TypeVar("T")


def _build_center_boxes(boxes: list[BoundingBox], scale: float) -> list[BoundingBox]:
    center_boxes: list[BoundingBox] = []
    s = max(0.2, min(1.0, float(scale)))
    for b in boxes:
        cx = (b.x1 + b.x2) * 0.5
        cy = (b.y1 + b.y2) * 0.5
        w = (b.x2 - b.x1) * s
        h = (b.y2 - b.y1) * s
        x1 = max(0.0, cx - w * 0.5)
        y1 = max(0.0, cy - h * 0.5)
        x2 = min(1.0, cx + w * 0.5)
        y2 = min(1.0, cy + h * 0.5)
        center_boxes.append(
            BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=b.confidence,
                sku_id=None,
                sku_name=None,
                match_score=None,
            )
        )
    return center_boxes


def _select_items(items: list[T], indices: list[int]) -> list[T]:
    return [items[i] for i in indices]


@router.post("/detect", response_model=DetectResponse)
async def detect_products(
    file: UploadFile = File(..., description="货架图片（JPEG / PNG / WEBP）"),
    conf: float | None = Query(None, ge=0.01, le=1.0, description="置信度阈值（覆盖 config 默认值）"),
    return_image: bool = Query(True, description="是否返回 base64 标注图"),
):
    """
    Two-stage shelf product detection:
    1. YOLO detects product boxes.
    2. CLIP embeddings are matched in FAISS to identify SKU IDs.
    Returns per-SKU counts and an optional annotated image.
    """
    from api.main import counter, detector, embedder, matcher, ocr_booster, cfg

    if detector is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    total_start = time.perf_counter()

    # 1. 读取 + 校验 + 解码
    file_bytes = await file.read()
    loop = asyncio.get_running_loop()   # 3.10+ 推荐用法，不再用 get_event_loop()
    image = await loop.run_in_executor(
        None,
        lambda: ImagePreprocessor.preprocess(
            file_bytes,
            max_dim=cfg["preprocessing"]["max_image_dimension"],
            max_mb=cfg["api"]["max_image_size_mb"],
            auto_enhance=cfg["preprocessing"]["auto_enhance_lighting"],
            clahe_clip_limit=cfg["preprocessing"]["clahe_clip_limit"],
            clahe_tile_grid_size=cfg["preprocessing"]["clahe_tile_grid_size"],
        ),
    )

    # 2. YOLO 检测
    boxes, inference_ms = await loop.run_in_executor(
        None,
        lambda: detector.detect(image, conf=conf),
    )

    # 3. Batch CLIP embedding extraction + FAISS matching
    embedding_ms = 0.0
    ocr_ms = 0.0
    if boxes and matcher.index_size > 0:
        embeddings, embedding_ms = await loop.run_in_executor(
            None,
            lambda: embedder.embed_crops(image, boxes),
        )
        clip_top1_scores = matcher.top1_clip_scores(embeddings)
        ocr_cfg = cfg.get("ocr", {})
        ocr_gate_low = float(ocr_cfg.get("clip_gate_low", 0.25))
        ocr_gate_high = float(ocr_cfg.get("clip_gate_high", 0.70))
        det_confs = np.asarray([b.confidence for b in boxes], dtype=np.float32)

        match_results = matcher.match_batch(
            embeddings,
            detection_confidences=det_confs,
        )

        for i, clip_score in enumerate(clip_top1_scores):
            if float(clip_score) <= ocr_gate_low:
                match_results[i] = {"sku_id": "unknown", "sku_name": "unknown", "score": float(clip_score)}

        ocr_indices: list[int] = []
        if ocr_booster is not None and getattr(ocr_booster, "enabled", False):
            for i, clip_score in enumerate(clip_top1_scores):
                if ocr_gate_low < float(clip_score) < ocr_gate_high:
                    ocr_indices.append(i)

        if ocr_indices:
            t_ocr = time.perf_counter()
            ocr_boxes = _select_items(boxes, ocr_indices)
            ocr_embeddings = embeddings[ocr_indices]
            ocr_det_confs = det_confs[ocr_indices]
            ocr_score_maps = await loop.run_in_executor(
                None,
                lambda: ocr_booster.score_boxes(image, ocr_boxes),
            )
            ocr_ms = (time.perf_counter() - t_ocr) * 1000.0
            ocr_results = matcher.match_batch(
                ocr_embeddings,
                detection_confidences=ocr_det_confs,
                ocr_score_maps=ocr_score_maps,
            )
            for idx, ocr_result in zip(ocr_indices, ocr_results):
                match_results[idx] = ocr_result

        # Optional center-region verification: reduce false positives on cluttered shelves.
        center_verify_cfg = cfg["matcher"].get("center_verify", {})
        if center_verify_cfg.get("enabled", False):
            center_scale = center_verify_cfg.get("scale", 0.72)
            center_boxes = _build_center_boxes(boxes, center_scale)
            center_embeddings, center_embedding_ms = await loop.run_in_executor(
                None,
                lambda: embedder.embed_crops(image, center_boxes),
            )
            embedding_ms += center_embedding_ms
            center_ocr_score_maps: list[dict[str, float]] | None = None
            if ocr_indices:
                center_ocr_score_maps = [{} for _ in center_boxes]
                center_selected_boxes = _select_items(center_boxes, ocr_indices)
                center_selected_ocr = await loop.run_in_executor(
                    None,
                    lambda: ocr_booster.score_boxes(image, center_selected_boxes),
                )
                for idx, score_map in zip(ocr_indices, center_selected_ocr):
                    center_ocr_score_maps[idx] = score_map
            center_match_results = matcher.match_batch(
                center_embeddings,
                detection_confidences=det_confs,
                ocr_score_maps=center_ocr_score_maps,
            )
            center_min_score = float(center_verify_cfg.get("min_match_score", 0.40))
            for i, (m_full, m_center) in enumerate(zip(match_results, center_match_results)):
                same_sku = m_full["sku_id"] == m_center["sku_id"] and m_full["sku_id"] != "unknown"
                center_ok = float(m_center["score"]) >= center_min_score
                if not (same_sku and center_ok):
                    match_results[i] = {"sku_id": "unknown", "sku_name": "unknown", "score": m_full["score"]}
                else:
                    # Conservative score: keep lower one after verification.
                    match_results[i]["score"] = min(float(m_full["score"]), float(m_center["score"]))

        for box, match in zip(boxes, match_results):
            box.sku_id = match["sku_id"]
            box.sku_name = match["sku_name"]
            box.match_score = round(match["score"], 4)
    else:
        # 无 SKU 数据库时，显示为通用商品
        for box in boxes:
            box.sku_id = "product"
            box.sku_name = "商品"
            box.match_score = None

    # 4. 计数
    counts = counter.count(boxes)

    # 5. 标注图（可选）
    annotated_b64: str | None = None
    if return_image:
        annotated = counter.annotate_image(image, boxes, counts)
        quality = cfg["api"].get("jpeg_quality", 85)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    total_ms = (time.perf_counter() - total_start) * 1000

    return DetectResponse(
        total_count=sum(counts.values()),
        by_sku=counts,
        boxes=boxes,
        inference_time_ms=round(inference_ms, 2),
        embedding_time_ms=round(embedding_ms, 2),
        total_time_ms=round(total_ms, 2),
        model_backend=detector.backend,
        annotated_image=annotated_b64,
    )
