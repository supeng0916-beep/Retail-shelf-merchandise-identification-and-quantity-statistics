from __future__ import annotations
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0, description="归一化左边界")
    y1: float = Field(..., ge=0.0, le=1.0, description="归一化上边界")
    x2: float = Field(..., ge=0.0, le=1.0, description="归一化右边界")
    y2: float = Field(..., ge=0.0, le=1.0, description="归一化下边界")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO 检测置信度")
    sku_id: str | None = Field(None, description="匹配到的 SKU ID，None 表示未识别")
    sku_name: str | None = Field(None, description="SKU 商品名称")
    match_score: float | None = Field(None, description="FAISS 余弦相似度分数")


class DetectResponse(BaseModel):
    total_count: int = Field(..., description="检测到的商品总数量")
    by_sku: dict[str, int] = Field(..., description="按 SKU 分类的数量统计")
    boxes: list[BoundingBox] = Field(..., description="所有检测框详情")
    inference_time_ms: float = Field(..., description="YOLO 推理耗时（ms）")
    embedding_time_ms: float = Field(..., description="特征提取耗时（ms）")
    total_time_ms: float = Field(..., description="端到端总耗时（ms）")
    model_backend: str = Field(..., description="推理后端（tensorrt / pytorch）")
    annotated_image: str | None = Field(None, description="标注结果图片（base64 JPEG）")


class SKUItem(BaseModel):
    sku_id: str = Field(..., description="SKU 唯一标识符")
    name: str = Field(..., description="商品名称")
    barcode: str | None = Field(None, description="条形码 / EAN 码")
    category: str | None = Field(None, description="商品分类")
    image_count: int = Field(0, description="已录入样张数量")


class SKUAddRequest(BaseModel):
    sku_id: str
    name: str
    barcode: str | None = None
    category: str | None = None


class SKUListResponse(BaseModel):
    total: int
    items: list[SKUItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    device: str
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    sku_count: int
    index_size: int
    uptime_seconds: float
    matcher_params: dict[str, object] | None = None
