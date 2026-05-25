from __future__ import annotations
from collections import defaultdict

import cv2
import numpy as np

from src.schemas import BoundingBox


# 颜色：已识别=绿色，未识别=浅绿色（BGR）
_COLOR_IDENTIFIED = (0, 200, 60)
_COLOR_UNKNOWN = (0, 180, 60)
_COLOR_PANEL_BG = (30, 30, 30)
_COLOR_TEXT = (255, 255, 255)


class ProductCounter:

    def count(self, boxes: list[BoundingBox]) -> dict[str, int]:
        """
        按 sku_id 统计数量。
        未识别的商品归入 'unknown' 类别。
        """
        counts: dict[str, int] = defaultdict(int)
        for box in boxes:
            key = box.sku_id if box.sku_id and box.sku_id != "unknown" else "unknown"
            counts[key] += 1
        return dict(counts)

    def annotate_image(
        self,
        image: np.ndarray,
        boxes: list[BoundingBox],
        counts: dict[str, int],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        在原图上绘制检测框 + 统计面板。
        已识别商品：绿框 + SKU 名称
        未识别商品：灰框 + "?"
        右上角：总数量 + top-5 SKU 明细面板
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # --- 绘制 bbox ---
        for box in boxes:
            x1 = int(box.x1 * w)
            y1 = int(box.y1 * h)
            x2 = int(box.x2 * w)
            y2 = int(box.y2 * h)

            is_identified = box.sku_id and box.sku_id not in ("unknown", "product", None)
            color = _COLOR_IDENTIFIED if is_identified else _COLOR_UNKNOWN
            thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if draw_labels:
                if is_identified:
                    label = box.sku_name or box.sku_id
                else:
                    # 无 SKU 库时只显示置信度
                    label = f"{box.confidence:.2f}"
                # 截断过长的标签
                if len(label) > 15:
                    label = label[:14] + "."
                font_scale = max(0.3, min(0.5, (x2 - x1) / 120))
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                # 标签背景
                label_y = max(y1 - 2, th + 2)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - th - baseline),
                    (x1 + tw, label_y),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    _COLOR_TEXT,
                    1,
                    cv2.LINE_AA,
                )

        # --- 右上角统计面板 ---
        total = sum(counts.values())
        # 按数量降序排列，取 top-5
        sku_name_map: dict[str, str] = {}
        for box in boxes:
            if box.sku_id and box.sku_id not in ("unknown", "product") and box.sku_name:
                sku_name_map.setdefault(box.sku_id, box.sku_name)

        top_skus = sorted(
            [(k, v) for k, v in counts.items() if k != "unknown"],
            key=lambda x: -x[1],
        )[:5]

        panel_lines = [f"Total: {total}"]
        for sku_id, cnt in top_skus:
            display_name = sku_name_map.get(sku_id, "Product")
            panel_lines.append(f"  {display_name[:18]}: {cnt}")
        # 有 SKU 数据时才显示 unknown 行
        if "unknown" in counts and top_skus:
            panel_lines.append(f"  unknown: {counts['unknown']}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 从 0.55 增大到 0.7
        line_h = 28       # 从 22 增大到 28
        padding = 10      # 从 8 增大到 10
        panel_w = 240     # 从 180 增大到 240
        panel_h = len(panel_lines) * line_h + padding * 2

        px1 = w - panel_w - 10
        py1 = 10
        px2 = w - 10
        py2 = py1 + panel_h

        # 半透明背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), _COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        for i, line in enumerate(panel_lines):
            y_pos = py1 + padding + (i + 1) * line_h - 4
            text_color = (100, 220, 100) if i == 0 else _COLOR_TEXT
            cv2.putText(
                annotated,
                line,
                (px1 + 6, y_pos),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated
