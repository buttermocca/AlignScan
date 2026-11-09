from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


class OnnxDocumentDetector:
    """Optional YOLOv8-style ONNX detector. Supply a model trained for documents/cards."""
    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("ONNX Runtime가 설치되지 않았습니다. pip install onnxruntime-directml") from exc
        available = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(Path(model_path)), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_h = int(shape[2]) if isinstance(shape[2], int) else 640
        self.input_w = int(shape[3]) if isinstance(shape[3], int) else 640
        self.provider = self.session.get_providers()[0]

    def detect(self, image: np.ndarray, confidence: float = 0.30) -> list[np.ndarray]:
        h, w = image.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_w - nw) // 2, (self.input_h - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        output = self.session.run(None, {self.input_name: tensor})[0]
        pred = np.asarray(output)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
            pred = pred.T
        if pred.shape[1] < 5:
            raise RuntimeError("지원하지 않는 ONNX 출력 형식입니다. YOLOv8 detection 형식을 사용하세요.")
        boxes, scores = [], []
        for row in pred:
            if len(row) < 5:
                continue
            score = float(np.max(row[4:]))
            if score < confidence:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1]); scores.append(score)
        ids = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45)
        result = []
        for idx in np.asarray(ids).reshape(-1) if len(ids) else []:
            x, y, bw, bh = boxes[int(idx)]
            result.append(np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32))
        return result
