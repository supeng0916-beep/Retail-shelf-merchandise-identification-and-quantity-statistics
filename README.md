# 零售货架商品识别与数量统计系统

基于 **YOLO26n + CLIP + FAISS + OCR** 的两阶段零售货架商品识别与计数系统。上传一张货架图片，系统自动检测所有商品、识别 SKU 并按类别统计数量。

## 项目概览

| 项目 | 说明 |
|------|------|
| 项目名称 | 零售货架商品识别与数量统计 |
| 版本 | 1.0.0 |
| 核心能力 | 目标检测 → 特征匹配 → OCR 辅助 → 计数统计 |
| SKU 数据库 | 386 个 SKU，覆盖饮料、泡面、饼干、零食、巧克力、乳制品等品类 |
| 前端 | 内置 Web 控制台（识别中心 + 商品管理） |

## 技术栈

### 核心模型

| 组件 | 技术 | 作用 |
|------|------|------|
| 目标检测 | **YOLO26n** (Ultralytics) | 从货架图片中检测并裁剪出每个商品区域 |
| 特征提取 | **CLIP ViT-B/32** (OpenAI, HuggingFace Transformers) | 将商品裁剪图编码为 512 维语义向量 |
| 向量检索 | **FAISS** (GPU/CPU) | 在 SKU 向量库中执行余弦相似度 Top-K 搜索 |
| 文字识别 | **EasyOCR** | 识别商品包装上的文字，作为 CLIP 匹配的辅助信号 |
| 推理加速 | **TensorRT** (可选) | YOLO 推理加速，自动从 .pt 导出 engine |

### 后端框架

| 组件 | 技术 | 说明 |
|------|------|------|
| API 框架 | **FastAPI** + **Uvicorn** | 异步 RESTful API，自动生成 OpenAPI 文档 |
| 数据库 | **SQLite** (aiosqlite) | 存储 SKU 元数据和 FAISS 索引映射 |
| 图像处理 | **OpenCV** + **Pillow** | 图片解码、缩放、CLAHE 光照增强 |
| 日志 | **Loguru** | 结构化日志输出 |
| 配置 | **PyYAML** | `config.yaml` 统一管理所有参数 |

### 前端

| 组件 | 技术 | 说明 |
|------|------|------|
| UI | 原生 HTML/CSS/JS | 零依赖，内嵌于 FastAPI 静态文件服务 |
| 功能 | 识别中心 + 商品管理 | 拖拽上传、实时识别、SKU 批量导入 |

## 项目结构

```
零售货架商品识别与数量统计/
├── api/                          # FastAPI 应用
│   ├── main.py                   # 应用入口，lifespan 管理所有模块生命周期
│   └── routers/
│       ├── detect.py             # POST /api/v1/detect — 核心识别接口
│       ├── sku.py                # SKU 增删查 + 索引重建
│       ├── sku_ingest.py         # 批量导入 SKU（预分析 → 发布 → 同步）
│       └── health.py             # GET /api/v1/health — 健康检查
├── src/                          # 核心业务逻辑
│   ├── detector.py               # ShelfDetector — YOLO26n 推理封装
│   ├── embedder.py               # SKUEmbedder — CLIP 特征提取
│   ├── matcher.py                # SKUMatcher — FAISS 余弦匹配 + 投票机制
│   ├── ocr_booster.py            # OCRBooster — EasyOCR 文字辅助匹配
│   ├── counter.py                # ProductCounter — 计数 + 标注图绘制
│   ├── preprocessor.py           # ImagePreprocessor — 校验/解码/CLAHE 增强
│   └── schemas.py                # Pydantic 数据模型
├── database/
│   ├── init_db.py                # SQLite schema 初始化（幂等）
│   └── sku_images/               # SKU 样张图片库（按 sku_id 分文件夹）
├── index/
│   ├── faiss_clip.index          # FAISS 向量索引（完整版）
│   └── faiss_clip_focus.index    # FAISS 向量索引（精简版）
├── models/
│   ├── yolo_shelf.pt             # YOLO26n 检测权重
│   └── clip-vit-base-patch32-local/  # 本地 CLIP 模型文件
├── demo/                         # Web 前端
│   ├── index.html                # 单页应用
│   └── assets/
│       ├── app.js                # 前端逻辑
│       └── style.css             # 样式
├── scripts/                      # 工具脚本
│   ├── build_index.py            # 合成增强 + 全量重建 FAISS 索引
│   ├── download_model.py         # 下载 CLIP 模型到本地
│   ├── download_shelf_model.py   # 下载 YOLO 权重
│   ├── expand_sku_from_openfoodfacts.py  # 从 OpenFoodFacts 扩充 SKU
│   ├── fill_missing_images_ddg.py        # 用 DuckDuckGo 补全缺失图片
│   ├── make_curated_demo_shelf.py        # 生成精选合成货架图
│   ├── make_synthetic_demo_shelf.py      # 生成合成货架演示图
│   └── serper_collector.py       # Serper 图片采集
├── config.yaml                   # 全局配置文件
├── requirements.txt              # Python 依赖
├── my_sku_full.csv               # SKU 元数据库（386 条）
├── sku_runtime.db                # 运行时 SQLite 数据库
└── yolo26n.pt                    # YOLO26n 预训练权重
```

## 系统架构

### 两阶段识别流水线

```
货架图片
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: YOLO 目标检测                              │
│  ─ 输入: 原图 (≤1920px, 自动 CLAHE 增强)             │
│  ─ 输出: 归一化边界框列表 + 置信度                     │
│  ─ 后端: TensorRT (优先) → PyTorch (降级)            │
└───────────────────────┬─────────────────────────────┘
                        │ 裁剪每个检测框
                        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: CLIP 特征匹配 + OCR 辅助                   │
│  ─ CLIP 编码裁剪图 → 512 维向量                       │
│  ─ FAISS Top-K 余弦搜索 → 投票聚合                    │
│  ─ OCR 门控: CLIP 分数在 [0.25, 0.70] 区间时启用      │
│  ─ 最终得分 = 0.7×CLIP + 0.3×OCR                    │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  计数 + 标注                                         │
│  ─ 按 SKU 分组统计数量                                │
│  ─ 生成标注图（绿框=已识别, 灰框=未识别）              │
│  ─ 右上角统计面板: Total + Top-5 SKU 明细             │
└─────────────────────────────────────────────────────┘
```

### 匹配策略

| 机制 | 说明 |
|------|------|
| **Top-K 投票** | 每个裁剪图取 Top-5 候选，按 SKU 聚合投票数 |
| **分数边距** | 冠军与亚军的 combined_score 差值须 ≥ 0.02 |
| **最低票数** | 获胜 SKU 的投票数须 ≥ 3 |
| **最低分数** | 获胜 SKU 的 combined_score 须 ≥ 0.3 |
| **OCR 门控** | 仅当 CLIP Top-1 分数在 [0.25, 0.70] 时才调用 OCR |
| **中心验证** | 可选：对裁剪框中心区域再次匹配，降低杂乱货架误检 |

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1（推荐，GPU 加速）
- 8GB+ 显存（CLIP + YOLO 同时加载）

### 安装

```bash
# 1. 克隆项目
cd 零售货架商品识别与数量统计

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 下载模型权重（首次运行自动下载，也可手动）
python scripts/download_model.py
python scripts/download_shelf_model.py
```

### 启动服务

```bash
# 默认启动（端口 8000）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 生产环境（多 worker）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

启动后访问：

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/demo` | Web 控制台（识别中心 + 商品管理） |
| `http://localhost:8000/api/v1/health` | 健康检查 |

## API 接口

### 核心识别

```bash
# 上传货架图片进行识别
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@shelf_photo.jpg" \
  -F "conf=0.25" \
  -F "return_image=true"
```

**响应示例：**
```json
{
  "total_count": 15,
  "by_sku": {
    "BEV_NES_MIL_REF": 4,
    "BIS_ORE_CHOC_CRE": 3,
    "unknown": 2
  },
  "inference_time_ms": 45.2,
  "embedding_time_ms": 120.5,
  "total_time_ms": 350.8,
  "model_backend": "tensorrt",
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### SKU 管理

```bash
# 列出所有 SKU
curl http://localhost:8000/api/v1/sku/list?page=1&page_size=20

# 新增 SKU 样张（热更新索引，无需重启）
curl -X POST http://localhost:8000/api/v1/sku/add \
  -F "sku_id=my_new_sku" \
  -F "name=My Product" \
  -F "image=@product_photo.jpg"

# 删除 SKU
curl -X DELETE http://localhost:8000/api/v1/sku/{sku_id}

# 全量重建 FAISS 索引
curl -X POST http://localhost:8000/api/v1/sku/rebuild
```

### 批量导入 SKU

```bash
# 1. 预分析（上传 3-8 张同商品多角度照片）
curl -X POST http://localhost:8000/api/v1/sku/ingest-preview \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg" -F "files=@photo3.jpg"

# 2. 确认发布（使用预分析返回的 job_id）
curl -X POST http://localhost:8000/api/v1/sku/ingest \
  -H "Content-Type: application/json" \
  -d '{"job_id": "返回的job_id", "brand": "Nestle", "product_name": "Milo", "category": "Beverages"}'

# 3. 查询发布进度
curl http://localhost:8000/api/v1/sku/ingest/jobs/{job_id}
```

## 配置说明

所有配置集中在 [config.yaml](config.yaml)：

```yaml
model:
  weights_path: models/yolo_shelf.pt    # YOLO 权重路径
  use_tensorrt: false                    # 是否启用 TensorRT 加速
  conf_threshold: 0.05                   # 检测置信度阈值
  iou_threshold: 0.35                    # NMS IoU 阈值
  device: cuda:0                         # 推理设备

embedder:
  model_name: models/clip-vit-base-patch32-local  # CLIP 模型路径
  embedding_dim: 512                     # 向量维度
  batch_size: 32                         # 批处理大小

matcher:
  index_path: index/faiss_clip_focus.index  # FAISS 索引路径
  db_path: sku_runtime_focus.db             # SQLite 数据库路径
  vote_top_k: 5                            # 每个裁剪图取 Top-K 候选
  min_score_margin: 0.02                   # 冠军-亚军最低分数差
  min_vote_count: 3                        # 最低投票数
  min_combined_score: 0.3                  # 最低综合分数
  clip_weight: 0.7                         # CLIP 权重
  ocr_weight: 0.3                          # OCR 权重

ocr:
  enabled: true                            # 是否启用 OCR 辅助
  clip_gate_low: 0.25                      # OCR 门控下限
  clip_gate_high: 0.70                     # OCR 门控上限

preprocessing:
  max_image_dimension: 1920                # 图片最大边长
  auto_enhance_lighting: false             # 自动 CLAHE 光照增强
```

## 工具脚本

| 脚本 | 用途 |
|------|------|
| [build_index.py](scripts/build_index.py) | 合成增强 + 全量重建 FAISS 索引（每张原图生成 20 个变体） |
| [download_model.py](scripts/download_model.py) | 下载 CLIP 模型到本地 |
| [download_shelf_model.py](scripts/download_shelf_model.py) | 下载 YOLO 检测权重 |
| [expand_sku_from_openfoodfacts.py](scripts/expand_sku_from_openfoodfacts.py) | 从 OpenFoodFacts 开放数据扩充 SKU |
| [fill_missing_images_ddg.py](scripts/fill_missing_images_ddg.py) | 用 DuckDuckGo 搜索补全缺失的 SKU 图片 |
| [make_curated_demo_shelf.py](scripts/make_curated_demo_shelf.py) | 生成精选合成货架图用于演示 |
| [make_synthetic_demo_shelf.py](scripts/make_synthetic_demo_shelf.py) | 生成合成货架演示图 |
| [serper_collector.py](scripts/serper_collector.py) | Serper API 图片采集 |

### 重建索引

```bash
# 默认参数重建（每张图 20 个合成变体）
python scripts/build_index.py

# 自定义变体数量
python scripts/build_index.py --aug-per-image 30

# 指定设备和 SKU 目录
python scripts/build_index.py --device cuda:0 --sku-dir database/sku_images
```

## 数据格式

### SKU 图片目录结构

```
database/sku_images/
├── BEV_NES_MIL_REF/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── 0003.jpg
├── BIS_ORE_CHOC_CRE/
│   ├── 0001.jpg
│   └── 0002.jpg
└── .../
```

### SKU CSV 格式 ([my_sku_full.csv](my_sku_full.csv))

| 字段 | 说明 | 示例 |
|------|------|------|
| `sku_id` | 唯一标识符 | `BEV_NES_MIL_REF` |
| `brand` | 品牌 | `Nestle` |
| `product_name` | 商品名 | `Milo Powder` |
| `variant` | 规格/口味 | `Activ-Go Refill` |
| `size` | 容量/重量 | `1kg` |
| `category` | 分类 | `Beverage` |
| `search_keyword` | 搜索关键词 | `Milo Activ-Go Softpack 1kg` |

### SKU ID 命名规则

```
{分类前缀}_{品牌缩写}_{商品名缩写}_{变体缩写}

示例: BEV_NES_MIL_REF
  BEV = Beverages
  NES = Nestle
  MIL = Milo
  REF = Refill
```

## 性能指标

| 指标 | 参考值 |
|------|--------|
| YOLO 推理 (TensorRT) | ~45ms/张 (1080p) |
| CLIP 特征提取 (batch=32) | ~120ms |
| FAISS 检索 (Top-5) | <5ms |
| 端到端延迟 | ~350ms/张 |
| YOLO mAP50 | 0.892（训练集评估） |

## 许可证

本项目仅供学术研究与学习使用。
