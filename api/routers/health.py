from fastapi import APIRouter
from src.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from api.main import detector, matcher, ocr_booster, cfg

    if detector is None:
        return HealthResponse(
            status="starting",
            model_loaded=False,
            model_backend="unknown",
            device="unknown",
            gpu_memory_used_mb=None,
            gpu_memory_total_mb=None,
            sku_count=0,
            index_size=0,
            uptime_seconds=0.0,
            matcher_params=None,
        )

    used_mb, total_mb = detector.gpu_memory_info()
    matcher_cfg = cfg.get("matcher", {}) if cfg else {}

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_backend=detector.backend,
        device=str(detector.device),
        gpu_memory_used_mb=used_mb,
        gpu_memory_total_mb=total_mb,
        sku_count=matcher.sku_count if matcher else 0,
        index_size=matcher.index_size if matcher else 0,
        uptime_seconds=round(detector.uptime_seconds, 1),
        matcher_params={
            "min_score_margin": matcher_cfg.get("min_score_margin"),
            "min_combined_score": matcher_cfg.get("min_combined_score"),
            "vote_top_k": matcher_cfg.get("vote_top_k"),
            "min_vote_count": matcher_cfg.get("min_vote_count"),
            "clip_weight": matcher_cfg.get("clip_weight", 0.7),
            "ocr_weight": matcher_cfg.get("ocr_weight", 0.3),
            "ocr_enabled": bool(getattr(ocr_booster, "enabled", False)),
            "index_path": matcher_cfg.get("index_path"),
            "db_path": matcher_cfg.get("db_path"),
        },
    )
