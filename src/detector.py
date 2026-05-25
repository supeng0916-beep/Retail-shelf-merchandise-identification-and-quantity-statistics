from __future__ import annotations
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.schemas import BoundingBox


class ShelfDetector:
    """
    YOLO26n 推理封装。
    加载策略：TensorRT engine（优先）→ 导出 engine → 原生 PyTorch（降级）。
    线程安全：threading.Lock 保护 GPU 推理临界区。
    单例：由 FastAPI lifespan 管理，整个进程共享一个实例。
    """

    def __init__(self, config: dict) -> None:
        self.weights_path = Path(config["weights_path"])
        self.engine_path = Path(config["engine_path"])
        self.use_tensorrt = config.get("use_tensorrt", True)
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.min_box_area_ratio = config.get("min_box_area_ratio", 0.0)
        self.device = config.get("device", "cuda:0")
        self.warmup_iters = config.get("warmup_iterations", 3)

        self._lock = threading.Lock()
        self._backend: str = "unknown"
        self._model = None
        self._start_time = time.time()

        self._load_model()
        self.warmup(self.warmup_iters)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from ultralytics import YOLO

        # 1. 尝试直接加载已有 TensorRT engine
        if self.use_tensorrt and self.engine_path.exists():
            logger.info(f"加载 TensorRT engine: {self.engine_path}")
            try:
                self._model = YOLO(str(self.engine_path))
                self._backend = "tensorrt"
                logger.success("TensorRT engine 加载成功")
                return
            except Exception as e:
                logger.warning(f"TensorRT engine 加载失败: {e}，尝试重新导出")

        # 2. 从 .pt 导出 TensorRT engine
        if self.use_tensorrt and self.weights_path.exists():
            try:
                logger.info("首次导出 TensorRT engine（约需 2-5 分钟）...")
                pt_model = YOLO(str(self.weights_path))
                pt_model.export(
                    format="engine",
                    half=True,
                    device=self.device.replace("cuda:", ""),
                    workspace=4,
                    batch=1,
                )
                if self.engine_path.exists():
                    self._model = YOLO(str(self.engine_path))
                    self._backend = "tensorrt"
                    logger.success("TensorRT engine 导出并加载成功")
                    return
            except Exception as e:
                logger.warning(f"TensorRT 导出失败: {e}，降级到 PyTorch")

        # 3. 降级：原生 PyTorch
        if self.weights_path.exists():
            logger.info(f"加载 PyTorch 权重: {self.weights_path}")
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            self._backend = "pytorch"
            logger.success("PyTorch 模型加载成功")
            return

        # 4. 自动下载（ultralytics 会从官方下载 yolo26n.pt）
        logger.info("本地未找到权重文件，自动下载 yolo26n.pt ...")
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        self._model = YOLO("yolo26n.pt")
        # 将下载的权重复制到 models/ 目录
        import shutil
        downloaded = Path.home() / ".ultralytics" / "assets" / "yolo26n.pt"
        if downloaded.exists():
            shutil.copy(downloaded, self.weights_path)
        self._backend = "pytorch"
        logger.success("yolo26n.pt 下载并加载成功")

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------

    def warmup(self, iterations: int = 3) -> None:
        """推理虚拟数据消除 CUDA kernel 冷启动延迟（200-500ms）。"""
        import numpy as np
        logger.info(f"模型预热中（{iterations} 次）...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(iterations):
            self._model.predict(
                dummy,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.success(f"预热完成，耗时 {elapsed:.1f}ms（共 {iterations} 次）")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> tuple[list[BoundingBox], float]:
        """
        单张图片推理。
        返回 (boxes, inference_time_ms)。
        boxes 中的坐标为归一化值 [0, 1]。
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        h, w = image.shape[:2]

        with self._lock:
            t0 = time.perf_counter()
            results = self._model.predict(
                image,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
                half=(self._backend == "tensorrt"),
            )
            inference_ms = (time.perf_counter() - t0) * 1000

        boxes: list[BoundingBox] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                nx1 = max(0.0, x1 / w)
                ny1 = max(0.0, y1 / h)
                nx2 = min(1.0, x2 / w)
                ny2 = min(1.0, y2 / h)
                area_ratio = (nx2 - nx1) * (ny2 - ny1)
                if area_ratio < self.min_box_area_ratio:
                    continue
                boxes.append(
                    BoundingBox(
                        x1=nx1,
                        y1=ny1,
                        x2=nx2,
                        y2=ny2,
                        confidence=float(box.conf[0]),
                        sku_id=None,
                        sku_name=None,
                        match_score=None,
                    )
                )

        return boxes, inference_ms

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def gpu_memory_info(self) -> tuple[float | None, float | None]:
        """返回 (used_mb, total_mb)，无 CUDA 时返回 (None, None)。"""
        try:
            import torch
            if torch.cuda.is_available():
                idx = int(self.device.split(":")[-1])
                used = torch.cuda.memory_allocated(idx) / 1024 ** 2
                total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 2
                return round(used, 1), round(total, 1)
        except Exception:
            pass
        return None, None
