from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import warnings

import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", message=r".*torch.load.*weights_only=False.*")

# 全局共享对象（lifespan 注入，整个进程生命周期内共享）
detector = None
embedder = None
matcher = None
counter = None
ocr_booster = None
sku_token_map: dict[str, set[str]] = {}
cfg: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==================== Startup ====================
    global detector, embedder, matcher, counter, ocr_booster, sku_token_map, cfg

    config_path = Path("config.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # 初始化数据库（同步，避免 event loop 嵌套问题）
    from database.init_db import init_db_sync
    init_db_sync(cfg["matcher"]["db_path"])
    logger.info("数据库初始化完成")

    # 确保目录存在
    Path("models").mkdir(exist_ok=True)
    Path("index").mkdir(exist_ok=True)

    # 加载 YOLO 检测器（含预热）
    from src.detector import ShelfDetector
    logger.info("正在加载 YOLO 检测模型...")
    detector = ShelfDetector(cfg["model"])

    # 加载 CLIP 特征提取器
    from src.embedder import SKUEmbedder
    logger.info("正在加载 CLIP 特征提取器...")
    embedder = SKUEmbedder(
        device=cfg["embedder"]["device"],
        batch_size=cfg["embedder"]["batch_size"],
        model_name=cfg["embedder"].get("model_name", "openai/clip-vit-base-patch32"),
    )

    # 加载 FAISS 匹配器
    from src.matcher import SKUMatcher
    logger.info("正在加载 FAISS 匹配器...")
    matcher = SKUMatcher(
        index_path=cfg["matcher"]["index_path"],
        db_path=cfg["matcher"]["db_path"],
        embedding_dim=cfg["embedder"]["embedding_dim"],
        vote_top_k=cfg["matcher"].get("vote_top_k", 1),
        min_score_margin=cfg["matcher"].get("min_score_margin", 0.0),
        min_vote_count=cfg["matcher"].get("min_vote_count", 1),
        min_combined_score=cfg["matcher"].get("min_combined_score", 0.0),
        clip_weight=cfg["matcher"].get("clip_weight", 0.7),
        ocr_weight=cfg["matcher"].get("ocr_weight", 0.3),
    )
    logger.info(
        "Matcher params | min_score_margin={} | min_combined_score={} | vote_top_k={} | min_vote_count={}",
        cfg["matcher"].get("min_score_margin", 0.0),
        cfg["matcher"].get("min_combined_score", 0.0),
        cfg["matcher"].get("vote_top_k", 1),
        cfg["matcher"].get("min_vote_count", 1),
    )

    from src.ocr_booster import OCRBooster, build_sku_token_map

    ocr_cfg = cfg.get("ocr", {})
    ocr_booster = OCRBooster(
        enabled=ocr_cfg.get("enabled", True),
        languages=ocr_cfg.get("languages", ["en"]),
        gpu=ocr_cfg.get("gpu", False),
        model_storage_dir=ocr_cfg.get("model_storage_dir", "models/easyocr"),
        download_enabled=ocr_cfg.get("download_enabled", True),
    )
    sku_token_map = build_sku_token_map(ocr_cfg.get("sku_csv", "my_sku_full.csv"))
    if ocr_booster is not None:
        ocr_booster.set_sku_token_map(sku_token_map)

    # 初始化计数器
    from src.counter import ProductCounter
    counter = ProductCounter()

    logger.success(
        f"所有模块加载完成 | 检测后端: {detector.backend} "
        f"| SKU数量: {matcher.sku_count} | 索引向量: {matcher.index_size}"
    )

    yield

    # ==================== Shutdown ====================
    logger.info("服务关闭，释放 GPU 内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="零售货架商品识别 API",
    description="基于 YOLO26n + CLIP + FAISS 的两阶段货架商品识别与计数系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（允许手机浏览器、本地前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# 注册路由
from api.routers import detect, health, sku, sku_ingest
app.include_router(detect.router, prefix="/api/v1")
app.include_router(sku.router, prefix="/api/v1")
app.include_router(sku_ingest.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

# 挂载前端静态文件（http://localhost:8000/demo）
demo_dir = Path("demo")
if demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "零售货架识别 API", "docs": "/docs", "demo": "/demo"}
