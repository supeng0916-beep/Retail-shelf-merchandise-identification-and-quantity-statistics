from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall((text or "").lower()))


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall((text or "").lower()))


def build_sku_token_map(csv_path: str | Path) -> dict[str, set[str]]:
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"OCR token csv not found: {path}")
        return {}

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    token_map: dict[str, set[str]] = {}
    for row in rows:
        sku_id = (row.get("sku_id") or "").strip()
        if not sku_id:
            continue

        text_parts = [
            row.get("brand") or "",
            row.get("product_name") or "",
            row.get("variant") or "",
            row.get("size") or "",
            row.get("search_keyword") or "",
        ]
        tokens = set()
        for part in text_parts:
            tokens.update(_tokenize(part))

        tokens = {t for t in tokens if len(t) >= 2}
        if tokens:
            token_map[sku_id] = tokens

    logger.info(f"OCR token map loaded: {len(token_map)} skus")
    return token_map


class OCRBooster:
    def __init__(
        self,
        enabled: bool = True,
        languages: list[str] | None = None,
        gpu: bool = False,
        model_storage_dir: str | Path | None = None,
        download_enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.languages = languages or ["en"]
        self.gpu = bool(gpu)
        self.model_storage_dir = Path(model_storage_dir) if model_storage_dir else Path("models/easyocr")
        self.download_enabled = bool(download_enabled)

        self._reader = None
        self._sku_tokens: dict[str, set[str]] = {}
        self._token_to_skus: dict[str, set[str]] = {}

        if not self.enabled:
            return

        try:
            import easyocr

            self.model_storage_dir.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False,
                model_storage_directory=str(self.model_storage_dir),
                user_network_directory=str(self.model_storage_dir),
                download_enabled=self.download_enabled,
            )
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR unavailable, OCR boost disabled: {e}")
            self.enabled = False

    def set_sku_token_map(self, token_map: dict[str, set[str]]) -> None:
        self._sku_tokens = token_map or {}
        token_to_skus: dict[str, set[str]] = {}
        for sku_id, tokens in self._sku_tokens.items():
            for t in tokens:
                token_to_skus.setdefault(t, set()).add(sku_id)
        self._token_to_skus = token_to_skus

    @staticmethod
    def _prepare_crop(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        scale = 1.0
        if min(h, w) < 160:
            scale = 160.0 / max(1, float(min(h, w)))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop

    def read_texts(self, image: np.ndarray) -> list[str]:
        if self._reader is None:
            return []

        crop = self._prepare_crop(image)
        texts = self._reader.readtext(crop, detail=0, paragraph=False)
        return [str(t).strip() for t in texts if str(t).strip()]

    def _ocr_tokens(self, crop: np.ndarray) -> set[str]:
        texts = self.read_texts(crop)
        merged = " ".join(str(t) for t in texts)
        return _tokenize(_normalize_text(merged))

    def score_boxes(self, image: np.ndarray, boxes: list) -> list[dict[str, float]]:
        if not self.enabled or self._reader is None or not self._sku_tokens:
            return [{} for _ in boxes]

        h, w = image.shape[:2]
        scored: list[dict[str, float]] = []

        for box in boxes:
            x1 = max(0, int(float(box.x1) * w))
            y1 = max(0, int(float(box.y1) * h))
            x2 = min(w, int(float(box.x2) * w))
            y2 = min(h, int(float(box.y2) * h))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                scored.append({})
                continue

            ocr_tokens = self._ocr_tokens(crop)
            if not ocr_tokens:
                scored.append({})
                continue

            candidate_skus: set[str] = set()
            for t in ocr_tokens:
                candidate_skus.update(self._token_to_skus.get(t, set()))

            if not candidate_skus:
                scored.append({})
                continue

            sku_scores: dict[str, float] = {}
            for sku_id in candidate_skus:
                sku_tokens = self._sku_tokens.get(sku_id, set())
                if not sku_tokens:
                    continue
                overlap = len(ocr_tokens & sku_tokens)
                if overlap <= 0:
                    continue
                denom = max(1, min(6, len(sku_tokens)))
                score = min(1.0, overlap / float(denom))
                if score > 0:
                    sku_scores[sku_id] = float(score)

            scored.append(sku_scores)

        return scored
