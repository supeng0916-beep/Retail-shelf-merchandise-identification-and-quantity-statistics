from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import aiosqlite
import faiss
import numpy as np
from loguru import logger


def _db_uri(db_path: Path) -> str:
    return f"file:{db_path.as_posix()}?mode=rwc"


class SKUMatcher:
    """FAISS cosine-similarity matcher with optional top-k voting."""

    def __init__(
        self,
        index_path: str,
        db_path: str,
        embedding_dim: int = 512,
        vote_top_k: int = 1,
        min_score_margin: float = 0.0,
        min_vote_count: int = 1,
        min_combined_score: float = 0.0,
        clip_weight: float = 0.7,
        ocr_weight: float = 0.3,
    ) -> None:
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.vote_top_k = max(1, int(vote_top_k))
        self.min_score_margin = max(0.0, float(min_score_margin))
        self.min_vote_count = max(1, int(min_vote_count))
        self.min_combined_score = max(0.0, float(min_combined_score))
        self.clip_weight = max(0.0, float(clip_weight))
        self.ocr_weight = max(0.0, float(ocr_weight))
        total_weight = self.clip_weight + self.ocr_weight
        if total_weight <= 0:
            self.clip_weight = 1.0
            self.ocr_weight = 0.0
        else:
            self.clip_weight /= total_weight
            self.ocr_weight /= total_weight

        self._lock = threading.RLock()
        self._index: faiss.IndexFlatIP | None = None
        self._idx_to_sku_id: list[str] = []
        self._sku_meta: dict[str, dict] = {}

        self._load()

    def _load(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            logger.info(f"Loading FAISS index: {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            logger.success(f"FAISS index loaded, vectors={self._index.ntotal}")
        else:
            logger.info("FAISS index not found, creating empty index")
            self._index = faiss.IndexFlatIP(self.embedding_dim)

        self._load_meta_sync()

    def _load_meta_sync(self) -> None:
        con = sqlite3.connect(_db_uri(self.db_path), uri=True)
        try:
            con.execute("PRAGMA journal_mode=MEMORY")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA temp_store=MEMORY")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS skus (
                    sku_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    barcode TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sku_embeddings (
                    faiss_idx INTEGER PRIMARY KEY,
                    sku_id TEXT NOT NULL,
                    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
                )
                """
            )
            con.commit()

            rows = con.execute("SELECT sku_id, name, barcode, category FROM skus").fetchall()
            self._sku_meta = {
                row[0]: {"name": row[1], "barcode": row[2], "category": row[3]}
                for row in rows
            }

            emb_rows = con.execute(
                "SELECT faiss_idx, sku_id FROM sku_embeddings ORDER BY faiss_idx"
            ).fetchall()
            max_idx = max((r[0] for r in emb_rows), default=-1)
            self._idx_to_sku_id = [""] * (max_idx + 1)
            for faiss_idx, sku_id in emb_rows:
                self._idx_to_sku_id[faiss_idx] = sku_id
        finally:
            con.close()

        logger.info(f"SKU metadata loaded, sku_count={len(self._sku_meta)}")

    def save_index(self) -> None:
        with self._lock:
            faiss.write_index(self._index, str(self.index_path))
        logger.info(f"FAISS index saved: {self.index_path}")

    def _aggregate_candidates(
        self,
        score_row: np.ndarray,
        index_row: np.ndarray,
        ocr_scores: dict[str, float] | None = None,
    ) -> list[dict]:
        sku_states: dict[str, dict[str, float | int]] = {}
        ocr_scores = ocr_scores or {}
        use_ocr = bool(ocr_scores)

        for score, idx in zip(score_row, index_row):
            idx = int(idx)
            if idx < 0 or idx >= len(self._idx_to_sku_id):
                continue
            sku_id = self._idx_to_sku_id[idx]
            if not sku_id:
                continue

            state = sku_states.setdefault(
                sku_id,
                {"votes": 0, "best_clip_score": -1.0},
            )
            state["votes"] = int(state["votes"]) + 1
            state["best_clip_score"] = max(float(state["best_clip_score"]), float(score))

        candidates: list[dict] = []
        for sku_id, state in sku_states.items():
            best_clip = float(state["best_clip_score"])
            ocr_score = float(ocr_scores.get(sku_id, 0.0))
            if use_ocr:
                combined = self.clip_weight * best_clip + self.ocr_weight * ocr_score
            else:
                # Preserve pure-CLIP behavior when OCR is unavailable for this crop.
                combined = best_clip
            candidates.append(
                {
                    "sku_id": sku_id,
                    "votes": int(state["votes"]),
                    "clip_score": best_clip,
                    "ocr_score": ocr_score,
                    "combined_score": combined,
                }
            )

        candidates.sort(
            key=lambda x: (-int(x["votes"]), -float(x["combined_score"]), -float(x["clip_score"]), x["sku_id"])
        )
        return candidates

    def top1_clip_scores(self, embeddings: np.ndarray) -> np.ndarray:
        if self._index.ntotal == 0 or len(embeddings) == 0:
            return np.zeros((len(embeddings),), dtype=np.float32)

        with self._lock:
            scores, _ = self._index.search(embeddings.astype(np.float32), 1)
        return scores[:, 0].astype(np.float32)

    def match_batch(
        self,
        embeddings: np.ndarray,
        top_k: int | None = None,
        detection_confidences: np.ndarray | None = None,
        ocr_score_maps: list[dict[str, float]] | None = None,
    ) -> list[dict]:
        results: list[dict] = []

        if self._index.ntotal == 0 or len(embeddings) == 0:
            for _ in range(len(embeddings)):
                results.append({"sku_id": "unknown", "sku_name": "unknown", "score": 0.0})
            return results

        search_k = int(top_k) if top_k is not None else self.vote_top_k
        search_k = max(1, min(search_k, self._index.ntotal))

        with self._lock:
            scores, indices = self._index.search(embeddings.astype(np.float32), search_k)

        for i in range(len(embeddings)):
            ocr_scores = ocr_score_maps[i] if ocr_score_maps is not None and i < len(ocr_score_maps) else {}
            candidates = self._aggregate_candidates(scores[i], indices[i], ocr_scores)
            if not candidates:
                results.append({"sku_id": "unknown", "sku_name": "unknown", "score": 0.0})
                continue

            winner = candidates[0]
            runner_up_combined = float(candidates[1]["combined_score"]) if len(candidates) > 1 else 0.0
            margin = float(winner["combined_score"]) - runner_up_combined

            if margin < self.min_score_margin:
                results.append(
                    {
                        "sku_id": "unknown",
                        "sku_name": "unknown",
                        "score": float(winner["combined_score"]),
                    }
                )
                continue

            if int(winner["votes"]) < self.min_vote_count:
                results.append(
                    {
                        "sku_id": "unknown",
                        "sku_name": "unknown",
                        "score": float(winner["combined_score"]),
                    }
                )
                continue

            if float(winner["combined_score"]) < self.min_combined_score:
                results.append(
                    {
                        "sku_id": "unknown",
                        "sku_name": "unknown",
                        "score": float(winner["combined_score"]),
                    }
                )
                continue

            best_sku = str(winner["sku_id"])

            meta = self._sku_meta.get(best_sku, {})
            results.append(
                {
                    "sku_id": best_sku,
                    "sku_name": meta.get("name", best_sku),
                    "score": float(winner["combined_score"]),
                    "clip_score": float(winner["clip_score"]),
                    "ocr_score": float(winner["ocr_score"]),
                    "votes": int(winner["votes"]),
                }
            )

        return results

    async def add_sku_async(
        self,
        sku_id: str,
        name: str,
        embedding: np.ndarray,
        barcode: str | None = None,
        category: str | None = None,
    ) -> None:
        vec = embedding.reshape(1, -1).astype(np.float32)
        async with aiosqlite.connect(_db_uri(self.db_path), uri=True) as db:
            await db.execute("PRAGMA journal_mode=MEMORY")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA temp_store=MEMORY")
            await db.execute(
                """
                INSERT INTO skus (sku_id, name, barcode, category)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sku_id) DO UPDATE SET
                  name=excluded.name, barcode=excluded.barcode, category=excluded.category
                """,
                (sku_id, name, barcode, category),
            )
            with self._lock:
                faiss_idx = self._index.ntotal
                self._index.add(vec)
                if faiss_idx >= len(self._idx_to_sku_id):
                    self._idx_to_sku_id.append(sku_id)
                else:
                    self._idx_to_sku_id[faiss_idx] = sku_id

            await db.execute(
                "INSERT INTO sku_embeddings (faiss_idx, sku_id) VALUES (?, ?)",
                (faiss_idx, sku_id),
            )
            await db.commit()

        self._sku_meta[sku_id] = {"name": name, "barcode": barcode, "category": category}
        logger.info(f"SKU updated: {sku_id} ({name}), faiss_idx={faiss_idx}")

    async def rebuild_index_async(self, embedder, sku_images_dir: str) -> int:
        import cv2 as _cv2

        images_root = Path(sku_images_dir)
        if not images_root.exists():
            logger.warning(f"SKU images directory not found: {images_root}")
            return 0

        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_idx_map: list[str] = []

        async with aiosqlite.connect(_db_uri(self.db_path), uri=True) as db:
            await db.execute("PRAGMA journal_mode=MEMORY")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA temp_store=MEMORY")
            await db.execute("DELETE FROM sku_embeddings")
            await db.commit()

            for sku_dir in sorted(images_root.iterdir()):
                if not sku_dir.is_dir():
                    continue

                sku_id = sku_dir.name
                image_files = (
                    list(sku_dir.glob("*.jpg"))
                    + list(sku_dir.glob("*.jpeg"))
                    + list(sku_dir.glob("*.png"))
                    + list(sku_dir.glob("*.webp"))
                )
                if not image_files:
                    continue

                for img_path in image_files:
                    img = _cv2.imread(str(img_path))
                    if img is None:
                        continue

                    emb = embedder.embed_single(img)
                    vec = emb.reshape(1, -1).astype(np.float32)
                    faiss_idx = new_index.ntotal
                    new_index.add(vec)
                    new_idx_map.append(sku_id)

                    await db.execute(
                        "INSERT INTO sku_embeddings (faiss_idx, sku_id) VALUES (?, ?)",
                        (faiss_idx, sku_id),
                    )
            await db.commit()

        with self._lock:
            self._index = new_index
            self._idx_to_sku_id = new_idx_map

        self.save_index()
        logger.success(f"FAISS index rebuilt, vectors={new_index.ntotal}")
        return new_index.ntotal

    @property
    def sku_count(self) -> int:
        return len(self._sku_meta)

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0
