# Retail Shelf Merchandise Identification

一个面向零售货架场景的商品识别与数量统计项目。系统基于 `YOLO + CLIP + FAISS + OCR`，能够对货架图中的商品进行检测、SKU 匹配、`Unknown` 拒识和数量统计，并提供一个可直接演示的 Web 页面。

## 1. Project Summary

本项目的目标是验证这样一条技术路线是否可行：

1. 使用 `YOLO` 从货架图中检测商品框
2. 使用 `CLIP` 提取每个商品裁剪图的视觉特征
3. 使用 `FAISS` 在 SKU 特征库中进行相似度检索
4. 在必要时结合 `OCR` 读取包装文字，辅助区分相似商品
5. 输出识别结果、商品名称和数量统计

当前仓库保留的是适合展示和二次开发的核心代码，不包含本地大模型文件、SKU 图片库、索引库、数据库和测试图片。

## 2. Core Capabilities

- 货架商品目标检测
- SKU 级别商品匹配
- `Unknown` 拒识，避免低置信度误匹配
- OCR 语义加权，辅助区分相似包装商品
- 多 SKU 数量统计与可视化标注
- SKU 批量导入与索引重建流程
- 前端中文演示页面

## 3. System Architecture

### Detection Layer

- `src/detector.py`
- 使用 YOLO 模型检测货架中的商品位置

### Feature Matching Layer

- `src/embedder.py`
- 使用 CLIP 提取 512 维视觉特征

- `src/matcher.py`
- 使用 FAISS 进行余弦相似度检索
- 支持 Top-K 投票、分数边界检查和 `Unknown` 判定

### OCR Boosting Layer

- `src/ocr_booster.py`
- 仅在 CLIP 分数落入指定区间时触发 OCR
- 通过品牌、口味、容量等关键词提升细粒度区分能力

### API and Demo Layer

- `api/main.py`
- `api/routers/`
- `demo/`

提供完整的后端接口和一个前端演示页面，用于上传货架图、查看结果以及管理 SKU 导入流程。

## 4. Repository Structure

```text
api/                    FastAPI application and routes
src/                    Core detection / embedding / matching logic
scripts/                Utility scripts for index building and data preparation
demo/                   Frontend demo page
database/init_db.py     SQLite schema initialization
config.yaml             Default runtime config (Full Library Mode)
config.full.yaml        Full library config backup
config.focus.yaml       Focused demo config backup
my_sku_full.csv         SKU metadata table
report.md               Boss-facing project report in Markdown
requirements.txt        Python dependencies
```

## 5. Runtime Modes

项目目前保留两套配置：

### Full Library Mode

默认配置文件：`config.yaml`

用于评估完整 SKU 库表现，主要路径如下：

- `index/faiss_clip.index`
- `sku_runtime.db`
- `database/sku_images`

### Focused Demo Mode

备用配置文件：`config.focus.yaml`

用于针对少量重点 SKU 做专项演示，主要路径如下：

- `index/faiss_clip_focus.index`
- `sku_runtime_focus.db`
- `database/sku_images_focus`

如果需要切换，可将对应配置内容覆盖到 `config.yaml`，或自行扩展为启动时读取不同配置文件。

## 6. Setup

### Environment

- Python 3.10+
- CUDA 环境可选，推荐用于加速 YOLO 和 CLIP

### Install

```bash
pip install -r requirements.txt
```

如果本地尚未准备模型文件，需要自行补充：

- YOLO 权重
- 本地 CLIP 模型目录
- EasyOCR 模型缓存

仓库中保留了相关脚本：

- `scripts/download_shelf_model.py`
- `scripts/download_model.py`

## 7. Run the Project

启动后端服务：

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

访问地址：

- Demo 页面：`http://127.0.0.1:8000/demo/`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`

## 8. Rebuild the Index

当你更新了 SKU 图片库后，需要重建索引：

```bash
python scripts/build_index.py
```

如果你希望显式指定主库路径：

```bash
python scripts/build_index.py --sku-dir database/sku_images --index-out index/faiss_clip.index --db sku_runtime.db
```

## 9. Data and Asset Notes

出于体积和隐私考虑，以下内容默认不上传到 GitHub：

- `database/sku_images/`
- `database/sku_images_backup/`
- `database/sku_images_focus/`
- `database/真实货架图/`
- `models/clip-vit-base-patch32-local/`
- `models/easyocr/`
- `models/*.pt`
- `index/*.index`
- `*.db`
- `datasets/`

也就是说，这个仓库更偏向于：

- 保留核心代码
- 保留配置与说明
- 保留 SKU 元数据结构
- 由使用者在本地补齐模型、图片和索引

## 10. Current Limitation

当前全库模式的瓶颈主要不在架构本身，而在 SKU 参考图质量：

- 网图与真实货架图之间存在明显域差距
- 部分 SKU 缺少多角度、真实场景参考图
- 相似包装商品仍然容易混淆

因此，后续提升准确率最有效的方式是：

- 提高 SKU 参考图清晰度
- 增加不同角度与真实场景图片
- 优先补强高频 SKU 和易混淆 SKU

## 11. Related Files

- 项目汇报：`report.md`
- 主配置：`config.yaml`
- 全库配置备份：`config.full.yaml`
- 聚焦演示配置备份：`config.focus.yaml`

---

如果你希望把这个项目继续扩展为完整的门店商品识别系统，可以在现有代码基础上继续补充：

- 更高质量的 SKU 图片库
- 更稳定的评估集
- 更细粒度的训练或对比学习策略
- 更完整的部署与用户管理流程
