from __future__ import annotations
import cv2
import numpy as np
from fastapi import HTTPException

# JPEG/PNG/WEBP magic bytes
_MAGIC_BYTES: dict[bytes, str] = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"RIFF": "webp",
}


class ImagePreprocessor:

    @staticmethod
    def validate_image(file_bytes: bytes, max_mb: float = 10.0) -> None:
        """快速校验文件大小和格式，在解码前调用以快速失败。"""
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_mb:
            raise HTTPException(
                status_code=413,
                detail=f"图片大小 {size_mb:.1f}MB 超过限制 {max_mb}MB",
            )
        header = file_bytes[:4]
        valid = any(header.startswith(magic) for magic in _MAGIC_BYTES)
        if not valid:
            raise HTTPException(
                status_code=400,
                detail="不支持的图片格式，请上传 JPEG / PNG / WEBP",
            )

    @staticmethod
    def decode_upload(
        file_bytes: bytes,
        max_dim: int = 1920,
    ) -> np.ndarray:
        """全内存解码 + 等比例缩放，不写临时文件。"""
        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="图片解码失败，文件可能已损坏")

        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def enhance_lighting(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: int = 8,
    ) -> np.ndarray:
        """
        CLAHE 光照均衡，解决货架局部过亮/过暗问题。
        仅对 LAB 色彩空间的 L（亮度）通道处理，不改变色调。
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def should_enhance(image: np.ndarray, std_threshold: float = 40.0) -> bool:
        """
        自动检测是否需要光照增强。
        图像亮度标准差低于阈值（光线昏暗或不均匀）时返回 True。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.std()) < std_threshold

    @classmethod
    def preprocess(
        cls,
        file_bytes: bytes,
        max_dim: int = 1920,
        max_mb: float = 10.0,
        auto_enhance: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> np.ndarray:
        """
        完整预处理管线：校验 → 解码 → 缩放 → 条件 CLAHE。
        供 API 路由直接调用。
        """
        cls.validate_image(file_bytes, max_mb=max_mb)
        image = cls.decode_upload(file_bytes, max_dim=max_dim)

        if auto_enhance and cls.should_enhance(image):
            image = cls.enhance_lighting(
                image,
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_grid_size,
            )
        return image
