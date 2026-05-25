"""
Download a shelf-product detection model and save it as `models/yolo_shelf.pt`.

Default:
  `keremberke/yolov8s-grocery-recognition` from Hugging Face

You can also pass `--model` with another HF model id or a local `.pt` path.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def download_grocery_model(
    hf_model_id: str = "keremberke/yolov8s-grocery-recognition",
    save_path: str = "models/yolo_shelf.pt",
) -> Path:
    from ultralytics import YOLO

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 方法1：ultralytics hf:// 前缀
    hf_uri = f"hf://{hf_model_id}" if not hf_model_id.startswith("hf://") else hf_model_id
    print(f"正在从 Hugging Face 下载模型: {hf_uri}")
    print("（首次下载需要几分钟，请耐心等待）\n")

    try:
        model = YOLO(hf_uri)
    except Exception as e:
        print(f"hf:// 方式失败: {e}")
        print("尝试使用 huggingface_hub 直接下载...")
        try:
            from huggingface_hub import hf_hub_download
            pt_file = hf_hub_download(
                repo_id=hf_model_id,
                filename="best.pt",
                local_dir=str(save_path.parent),
            )
            shutil.copy(pt_file, save_path)
            print(f"✓ 模型已保存到: {save_path}")
            model = YOLO(str(save_path))
        except Exception as e2:
            raise RuntimeError(
                f"两种下载方式均失败。\n"
                f"  hf://  错误: {e}\n"
                f"  hf_hub 错误: {e2}\n"
                f"请手动从 https://huggingface.co/{hf_model_id} 下载 best.pt，\n"
                f"保存到 {save_path} 后重新运行并加 --no-download 参数。"
            ) from e2

    # 保存模型文件
    downloaded = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
    if downloaded and downloaded.exists() and downloaded != save_path:
        shutil.copy(downloaded, save_path)
    elif not save_path.exists():
        model.save(str(save_path))
    print(f"✓ 模型已保存到: {save_path}")

    # 打印模型类别
    print(f"\n模型检测类别 ({len(model.names)} 类):")
    for idx, name in model.names.items():
        print(f"  {idx:3d}: {name}")

    return save_path


def patch_config(model_path: str = "models/yolo_shelf.pt") -> None:
    """自动更新 config.yaml 的 weights_path"""
    import yaml
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("未找到 config.yaml，请手动修改 weights_path")
        return

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    old_path = cfg["model"]["weights_path"]
    cfg["model"]["weights_path"] = model_path
    cfg["model"]["engine_path"] = model_path.replace(".pt", ".engine")

    config_file.write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n✓ config.yaml 已更新：{old_path} → {model_path}")
    print("  重启服务后生效：uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载货架商品检测模型")
    parser.add_argument(
        "--model",
        default="hf://keremberke/yolov8s-grocery-recognition",
        help="Hugging Face 模型 ID 或本地 .pt 路径",
    )
    parser.add_argument(
        "--save",
        default="models/yolo_shelf.pt",
        help="保存路径",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="不修改 config.yaml",
    )
    args = parser.parse_args()

    save_path = download_grocery_model(args.model, args.save)

    if not args.no_patch:
        patch_config(str(save_path))
