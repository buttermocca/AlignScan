from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class OnnxDocumentDetector:
    """ONNX document detector with a pagescan-compatible YOLO11 path.

    yolo_doc_v1 was exported as a single-class YOLO Detect model. Its training
    recipe uses 960px letterboxing and output (1, 5, N), not a four-corner
    regression head. It predicts coarse axis-aligned boxes; it does not return
    perspective-corrected document corners.
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

        # This model was trained/exported at 960px. If the ONNX graph marks
        # spatial axes dynamic, preserve that training size instead of silently
        # falling back to 640px (which noticeably weakens small-document recall).
        model_name = self.model_path.name.lower()
        default_size = 960 if model_name == "yolo_doc_v1.onnx" else 640
        self.input_h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else default_size
        self.input_w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else default_size
        self.provider = self.session.get_providers()[0]
        self.output_shapes = [tuple(output.shape) for output in self.session.get_outputs()]
        self.last_diagnostics = {
            "input_size": (self.input_w, self.input_h),
            "output_shape": None,
            "max_confidence": 0.0,
            "candidate_count": 0,
            "kept_count": 0,
        }

    @staticmethod
    def _letterbox(image_rgb: np.ndarray, width: int, height: int):
        """Match pagescan's training-time letterbox: long edge to 960, pad 114."""
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
        """Normalize common YOLO single-class layouts to (N, 5+) rows."""
        pred = np.asarray(output, dtype=np.float32)
        if pred.ndim == 3:
            if pred.shape[0] != 1:
                raise RuntimeError(f"ONNX batch 크기 1만 지원합니다: {pred.shape}")
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"지원하지 않는 YOLO 출력 차원입니다: {pred.shape}")

        # YOLO11 single-class detect exports as (5, N); alternate exporters
        # may expose (N, 5). Do not transpose the latter.
        if pred.shape[0] <= 16 and pred.shape[1] > pred.shape[0]:
            pred = pred.T
        if pred.shape[1] < 5:
            raise RuntimeError(
                f"지원하지 않는 ONNX 출력 형식입니다: {pred.shape}. "
                "yolo_doc_v1은 YOLO Detect 출력 (1, 5, N)을 사용합니다."
            )
        return pred

    def detect(self, image: np.ndarray, confidence: float = 0.25) -> list[np.ndarray]:
        """Return axis-aligned document boxes in original image coordinates.

        confidence=0.25 matches the upstream pagescan detector's threshold.
        """
        if image is None or image.size == 0:
            self.last_diagnostics.update({
                "output_shape": None, "max_confidence": 0.0,
                "candidate_count": 0, "kept_count": 0,
            })
            return []

        confidence = float(np.clip(confidence, 0.01, 0.95))
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_x, pad_y = self._letterbox(rgb, self.input_w, self.input_h)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]

        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            raise RuntimeError("ONNX 모델이 출력 텐서를 반환하지 않았습니다.")
        raw = np.asarray(outputs[0])
        rows = self._rows_from_output(raw)

        # This model has one class: [cx, cy, width, height, document_score].
        # Other single-class exports can append metadata, so use the maximum
        # class score only when extra score columns exist.
        scores = rows[:, 4] if rows.shape[1] == 5 else np.max(rows[:, 4:], axis=1)
        finite = np.isfinite(rows[:, :5]).all(axis=1) & np.isfinite(scores)
        scores = np.where(finite, scores, 0.0)
        max_confidence = float(np.max(scores)) if scores.size else 0.0

        boxes, kept_scores = [], []
        for row, score in zip(rows, scores):
            if score < confidence:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1 = (cx - bw / 2.0 - pad_x) / scale
            y1 = (cy - bh / 2.0 - pad_y) / scale
            x2 = (cx + bw / 2.0 - pad_x) / scale
            y2 = (cy + bh / 2.0 - pad_y) / scale
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            if x2 - x1 < 18 or y2 - y1 < 18:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            kept_scores.append(float(score))

        self.last_diagnostics = {
            "input_size": (self.input_w, self.input_h),
            "output_shape": tuple(raw.shape),
            "max_confidence": max_confidence,
            "candidate_count": len(boxes),
            "kept_count": 0,
            "threshold": confidence,
        }
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, kept_scores, confidence, 0.45)
        result = []
        for idx in np.asarray(indices).reshape(-1) if len(indices) else []:
            x, y, bw, bh = boxes[int(idx)]
            result.append(np.array([
                [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
            ], dtype=np.float32))
        self.last_diagnostics["kept_count"] = len(result)
        return result
