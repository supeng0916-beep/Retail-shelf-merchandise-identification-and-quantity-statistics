"""
下载 YOLO26n.pt 并（可选）导出 TensorRT FP16 engine。

用法：
    python scripts/download_model.py                 # 下载 + 导出 TensorRT
    python scripts/download_model.py --skip-tensorrt  # 仅下载 .pt
"""
import argparse
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "yolo26n.pt"
ENGINE_PATH = MODELS_DIR / "yolo26n.engine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tensorrt", action="store_true", help="跳过 TensorRT 导出")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Step 1: 下载 yolo26n.pt ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 下载 YOLO26n 权重")
    print("=" * 60)

    if WEIGHTS_PATH.exists():
        print(f"✓ 已存在: {WEIGHTS_PATH}（跳过下载）")
    else:
        try:
            from ultralytics import YOLO
            print("正在下载 yolo26n.pt ...")
            model = YOLO("yolo26n.pt")  # ultralytics 自动下载到 ~/.ultralytics/assets/

            # 复制到 models/ 目录
            downloaded = Path.home() / ".ultralytics" / "assets" / "yolo26n.pt"
            if downloaded.exists():
                shutil.copy(downloaded, WEIGHTS_PATH)
                print(f"✓ 权重已保存到: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"✓ 模型已加载（ultralytics 缓存目录）")
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            sys.exit(1)

    # ── Step 2: 导出 TensorRT engine ──────────────────────────────────────
    if args.skip_tensorrt:
        print("\nStep 2: 已跳过 TensorRT 导出（--skip-tensorrt）")
    else:
        print("\n" + "=" * 60)
        print("Step 2: 导出 TensorRT FP16 engine（首次约需 2-5 分钟）")
        print("=" * 60)

        import torch
        if not torch.cuda.is_available():
            print("⚠ 未检测到 CUDA GPU，跳过 TensorRT 导出")
            print("  服务将以 PyTorch 模式运行（速度较慢）")
        elif ENGINE_PATH.exists():
            print(f"✓ TensorRT engine 已存在: {ENGINE_PATH}（跳过导出）")
        else:
            try:
                from ultralytics import YOLO
                model = YOLO(str(WEIGHTS_PATH))
                print("正在编译 TensorRT engine ...")
                model.export(
                    format="engine",
                    half=True,          # FP16
                    device=0,
                    workspace=4,        # 4 GB TensorRT workspace
                    batch=1,
                )
                # ultralytics 导出的 engine 在权重同目录
                exported = WEIGHTS_PATH.with_suffix(".engine")
                if exported.exists() and not ENGINE_PATH.exists():
                    shutil.move(str(exported), str(ENGINE_PATH))

                if ENGINE_PATH.exists():
                    print(f"✓ TensorRT engine 已保存: {ENGINE_PATH} ({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)")
                else:
                    print("⚠ engine 文件位置未知，请检查 models/ 目录")
            except Exception as e:
                print(f"⚠ TensorRT 导出失败: {e}")
                print("  服务将以 PyTorch 模式运行")

    # ── Step 3: 验证 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 验证模型可以正常推理")
    print("=" * 60)

    import numpy as np
    from ultralytics import YOLO

    load_path = ENGINE_PATH if ENGINE_PATH.exists() else WEIGHTS_PATH
    print(f"加载: {load_path}")
    model = YOLO(str(load_path))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.predict(dummy, verbose=False)
    print(f"✓ 验证通过，黑色虚拟图检测到 {len(result[0].boxes)} 个目标（预期为 0）")

    print("\n" + "=" * 60)
    print("准备完成！启动服务：")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
