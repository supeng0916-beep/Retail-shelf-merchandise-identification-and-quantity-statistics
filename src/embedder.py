from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from loguru import logger

from src.schemas import BoundingBox


class SKUEmbedder:
    """CLIP image embedder for SKU matching."""

    EMBEDDING_DIM = 512

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 64,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.model_name = model_name

        try:
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "transformers is required for CLIP embedder. "
                "Please install: pip install transformers"
            ) from exc

        logger.info(f"Loading CLIP model: {model_name}")
        # Prefer local cache to avoid startup failures in restricted networks.
        try:
            self._processor = CLIPImageProcessor.from_pretrained(
                model_name, local_files_only=True
            )
            self._model = CLIPModel.from_pretrained(
                model_name,
                local_files_only=True,
                use_safetensors=True,
            )
        except Exception:
            self._processor = CLIPImageProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(
                model_name,
                use_safetensors=True,
            )
        self._model.eval().to(self.device)

        if self.device.type == "cuda":
            self._model.half()

        self.embedding_dim = int(self._model.config.projection_dim)
        self.EMBEDDING_DIM = self.embedding_dim
        logger.success(
            f"CLIP loaded on {self.device} | embedding_dim={self.embedding_dim}"
        )

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    @staticmethod
    def _to_feature_tensor(output: object) -> torch.Tensor:
        # transformers versions may return Tensor, ModelOutput, tuple, etc.
        if torch.is_tensor(output):
            return output

        image_embeds = getattr(output, "image_embeds", None)
        if torch.is_tensor(image_embeds):
            return image_embeds

        pooler_output = getattr(output, "pooler_output", None)
        if torch.is_tensor(pooler_output):
            return pooler_output

        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]

        raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")

    def _to_rgb_crops(
        self,
        image: np.ndarray,
        boxes: list[BoundingBox],
    ) -> list[np.ndarray]:
        h, w = image.shape[:2]
        crops: list[np.ndarray] = []

        for box in boxes:
            x1 = max(0, int(box.x1 * w))
            y1 = max(0, int(box.y1 * h))
            x2 = min(w, int(box.x2 * w))
            y2 = min(h, int(box.y2 * h))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        return crops

    def embed_crops(
        self,
        image: np.ndarray,
        boxes: list[BoundingBox],
    ) -> tuple[np.ndarray, float]:
        if not boxes:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), 0.0

        t0 = time.perf_counter()
        rgb_crops = self._to_rgb_crops(image, boxes)
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(rgb_crops), self.batch_size):
                batch_crops = rgb_crops[i : i + self.batch_size]
                batch = self._processor(images=batch_crops, return_tensors="pt")
                pixel_values = batch["pixel_values"].to(self.device)
                if self.device.type == "cuda":
                    pixel_values = pixel_values.half()

                try:
                    output = self._model.get_image_features(pixel_values=pixel_values)
                except Exception:
                    output = self._model(pixel_values=pixel_values)

                features = self._to_feature_tensor(output)
                if features.dim() > 2:
                    features = features[:, 0, :]
                features = features.float().cpu().numpy()
                all_embeddings.append(features)

        embeddings = np.concatenate(all_embeddings, axis=0)
        embeddings = self._l2_normalize(embeddings)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return embeddings, elapsed_ms

    def embed_single(self, image: np.ndarray) -> np.ndarray:
        embeddings, _ = self.embed_crops(
            image,
            [BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=1.0)],
        )
        return embeddings[0]
