from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class OnnxDocumentDetector:
    """ONNX single-document detector with overlapping-window multi-document search.

    yolo_doc_v1 is a single-class YOLO Detect model trained at 960px. It predicts
    axis-aligned document boxes, not precise quadrilateral corners. For flatbed
    scans containing several separate receipts/cards, the image is additionally
    evaluated in overlapping horizontal and vertical windows; this gives the
    single-document model a better chance to focus on one receipt at a time.
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(
                "ONNX Runtime가 설치되지 않았습니다. onnxruntime-directml 설치가 필요합니다."
            ) from exc

        self.model_path = Path(model_path)
        available = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )
        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name
        shape = self.input_info.shape

        # Upstream yolo_doc_v1 training/export uses imgsz=960.
        default_size = 960 if self.model_path.name.lower() == "yolo_doc_v1.onnx" else 640
        self.input_h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else default_size
        self.input_w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else default_size
        self.provider = self.session.get_providers()[0]
        self.output_shapes = [tuple(item.shape) for item in self.session.get_outputs()]
        self.last_diagnostics = {}

    @staticmethod
    def _letterbox(image_rgb: np.ndarray, width: int, height: int):
        source_h, source_w = image_rgb.shape[:2]
        scale = min(width / source_w, height / source_h)
        resized_w = max(1, int(round(source_w * scale)))
        resized_h = max(1, int(round(source_h * scale)))
        resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (width - resized_w) // 2
        pad_y = (height - resized_h) // 2
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _rows_from_output(output: np.ndarray) -> np.ndarray:
        pred = np.asarray(output, dtype=np.float32)
        if pred.ndim == 3:
            if pred.shape[0] != 1:
                raise RuntimeError(f"ONNX batch 크기 1만 지원합니다: {pred.shape}")
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"지원하지 않는 YOLO 출력 차원입니다: {pred.shape}")
        # YOLO11 single-class export normally returns (5, N).
        if pred.shape[0] <= 16 and pred.shape[1] > 16:
            pred = pred.T
        if pred.shape[1] < 5:
            raise RuntimeError(
                f"지원하지 않는 ONNX 출력 형식입니다: {pred.shape}. "
                "YOLO Detect (cx, cy, w, h, confidence) 형식이 필요합니다."
            )
        return pred

    def _infer_window(self, image_bgr: np.ndarray, confidence: float):
        """Run one window and return its candidate xyxy boxes, scores and output shape."""
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_x, pad_y = self._letterbox(rgb, self.input_w, self.input_h)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            raise RuntimeError("ONNX 모델이 출력 텐서를 반환하지 않았습니다.")
        raw = np.asarray(outputs[0])
        rows = self._rows_from_output(raw)
        scores = rows[:, 4] if rows.shape[1] == 5 else np.max(rows[:, 4:], axis=1)
        finite = np.isfinite(rows[:, :5]).all(axis=1) & np.isfinite(scores)
        scores = np.where(finite, scores, 0.0)

        boxes = []
        kept_scores = []
        rejected_full = 0
        for row, score in zip(rows, scores):
            if score < confidence:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1 = (cx - bw / 2.0 - pad_x) / scale
            y1 = (cy - bh / 2.0 - pad_y) / scale
            x2 = (cx + bw / 2.0 - pad_x) / scale
            y2 = (cy + bh / 2.0 - pad_y) / scale
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, 0, width - 1))
            y2 = float(np.clip(y2, 0, height - 1))
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 18 or box_h < 18:
                continue

            area_ratio = (box_w * box_h) / max(float(width * height), 1.0)
            # A box covering nearly the complete scan/window is typically the
            # page/backdrop in this use case, not an individual receipt.
            if area_ratio >= 0.88:
                rejected_full += 1
                continue
            boxes.append([x1, y1, box_w, box_h])
            kept_scores.append(float(score))

        raw_max = float(np.max(scores)) if scores.size else 0.0
        valid_max = max(kept_scores) if kept_scores else 0.0
        return boxes, kept_scores, tuple(raw.shape), raw_max, rejected_full, valid_max

    @staticmethod
    def _overlapping_windows(width: int, height: int):
        """Full-height columns and full-width rows with generous overlap."""
        windows = []
        fraction = 0.62
        if width >= 900:
            tile_w = min(width, max(720, int(round(width * fraction))))
            if tile_w < width:
                x_positions = sorted(set([0, width - tile_w]))
                for x in x_positions:
                    windows.append((x, 0, tile_w, height, "세로 분할"))
        if height >= 1100:
            tile_h = min(height, max(900, int(round(height * fraction))))
            if tile_h < height:
                y_positions = sorted(set([0, height - tile_h]))
                for y in y_positions:
                    windows.append((0, y, width, tile_h, "가로 분할"))
        return windows

    @staticmethod
    def _global_nms(boxes, scores, threshold: float = 0.48):
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.001, threshold)
        return [int(i) for i in np.asarray(indices).reshape(-1)] if len(indices) else []

    def detect(self, image: np.ndarray, confidence: float = 0.25) -> list[np.ndarray]:
        """Return individual axis-aligned boxes in original image coordinates."""
        if image is None or image.size == 0:
            self.last_diagnostics = {"error": "empty image", "kept_count": 0}
            return []

        confidence = float(np.clip(confidence, 0.01, 0.95))
        height, width = image.shape[:2]
        full_boxes, full_scores, output_shape, max_conf, full_rejected, valid_max = self._infer_window(
            image, confidence
        )

        mapped_boxes = [list(box) for box in full_boxes]
        mapped_scores = list(full_scores)
        tile_count = 0
        rejected_tile_boxes = 0
        tile_names = []

        # This model was trained to find one document in an image. If a full
        # scan only yields one candidate, search overlapping sub-windows so
        # individual receipts/cards can become the dominant object per window.
        windows = self._overlapping_windows(width, height) if len(full_boxes) < 2 else []
        for x, y, tile_w, tile_h, name in windows:
            tile = image[y:y + tile_h, x:x + tile_w]
            boxes, scores, _, tile_max, rejected, tile_valid_max = self._infer_window(tile, confidence)
            tile_count += 1
            tile_names.append(name)
            max_conf = max(max_conf, tile_max)
            valid_max = max(valid_max, tile_valid_max)
            rejected_tile_boxes += rejected
            for box, score in zip(boxes, scores):
                bx, by, bw, bh = box
                mapped_boxes.append([bx + x, by + y, bw, bh])
                mapped_scores.append(score)

        # Merge overlapping detections produced by adjacent tiles.
        indices = self._global_nms(mapped_boxes, mapped_scores, 0.48)
        result = []
        for idx in indices:
            x, y, box_w, box_h = mapped_boxes[idx]
            result.append(np.array([
                [x, y], [x + box_w, y],
                [x + box_w, y + box_h], [x, y + box_h]
            ], dtype=np.float32))

        # Prefer top-to-bottom, then left-to-right display order.
        result.sort(key=lambda corners: (
            float(np.min(corners[:, 1])),
            float(np.min(corners[:, 0]))
        ))
        self.last_diagnostics = {
            "input_size": (self.input_w, self.input_h),
            "output_shape": output_shape,
            "max_confidence": max_conf,
            "max_valid_confidence": valid_max,
            "full_frame_candidates": len(full_boxes),
            "full_frame_rejected": full_rejected,
            "tile_count": tile_count,
            "tile_rejected": rejected_tile_boxes,
            "candidate_count": len(mapped_boxes),
            "kept_count": len(result),
            "threshold": confidence,
            "tile_modes": tile_names,
        }
        return result
