from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSplitter, QFrame, QListWidget, QListWidgetItem,
    QSlider, QSpinBox, QComboBox, QGroupBox, QTabWidget, QScrollArea, QLineEdit)
from .geometry import DocumentRegion, detect_document_regions, rectify_document
from .imaging import enhance_image
from .canvas import ScanCanvas
from .ai_detector import OnnxDocumentDetector


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlignScan · 영수증 정렬 스튜디오")
        self.resize(1320, 860); self.setMinimumSize(980, 680)
        self.image: np.ndarray | None = None
        self.image_path: str | None = None
        self.regions: list[DocumentRegion] = []
        self.selected = -1
        self.ai_detector: OnnxDocumentDetector | None = None
        self._build_ui()
        self._connect_canvas()
        self.preview_timer = QTimer(self); self.preview_timer.setSingleShot(True); self.preview_timer.setInterval(130)
        self.preview_timer.timeout.connect(self.update_preview)
        self._set_controls_enabled(False)

    def _build_ui(self):
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#f8faf8;color:#34483f;font-family:'Malgun Gothic','맑은 고딕';font-size:10pt}
        QFrame#topbar{background:#fffaf7;border:1px solid #f2e4de;border-radius:16px}
        QLabel#brand{font-size:19pt;font-weight:900;color:#314b43}
        QLabel#hint{color:#82958d;font-size:9pt}
        QPushButton{background:#fff;border:1px solid #d8e5de;border-radius:10px;padding:8px 13px;color:#3c5b50;font-weight:600}
        QPushButton:hover{background:#eff8f3;border-color:#8ecbb3}
        QPushButton#primary{background:#ff8796;color:white;border:0;font-weight:800}
        QPushButton#primary:hover{background:#f36e82}
        QPushButton#mint{background:#54b99a;color:white;border:0}
        QGroupBox{border:1px solid #e0ebe4;border-radius:12px;margin-top:12px;padding:12px 9px 8px;font-weight:700;background:#fff}
        QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px;color:#527267}
        QListWidget{border:1px solid #e0ebe4;border-radius:10px;background:white;padding:5px}
        QListWidget::item{padding:10px;border-radius:7px;margin:2px}
        QListWidget::item:selected{background:#fff0f1;color:#78434b}
        QSlider::groove:horizontal{height:6px;background:#e8eee9;border-radius:3px}
        QSlider::handle:horizontal{background:#ff8796;width:15px;margin:-5px 0;border-radius:7px}
        QSpinBox,QComboBox,QLineEdit{background:white;border:1px solid #dce8e1;border-radius:8px;padding:6px}
        QTabWidget::pane{border:1px solid #e0ebe4;border-radius:10px;background:#fff}
        QTabBar::tab{padding:8px 14px;background:#edf3ef;border-radius:7px;margin:3px}
        QTabBar::tab:selected{background:#ffe8eb;color:#9a4b58;font-weight:700}
        QProgressBar{border:0;background:#edf3ef;border-radius:4px;height:7px}
        QProgressBar::chunk{background:#61b99a;border-radius:4px}
        """)
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(16, 12, 16, 14); outer.setSpacing(10)
        top = QFrame(objectName="topbar"); top_layout = QHBoxLayout(top); top_layout.setContentsMargins(16, 11, 16, 11)
        brand_col = QVBoxLayout(); title = QLabel("AlignScan  ♡"); title.setObjectName("brand")
        subtitle = QLabel("여러 장의 영수증·카드를 자동으로 찾아 반듯하게 저장해요"); subtitle.setObjectName("hint")
        brand_col.addWidget(title); brand_col.addWidget(subtitle); top_layout.addLayout(brand_col); top_layout.addStretch()
        self.btn_open = QPushButton("＋ 이미지 열기")
        self.btn_detect = QPushButton("✦ 자동 찾기")
        self.btn_add = QPushButton("✎ 영역 직접 지정")
        self.btn_save_all = QPushButton("모두 개별 저장"); self.btn_save_all.setObjectName("primary")
        for b in (self.btn_open, self.btn_detect, self.btn_add, self.btn_save_all): top_layout.addWidget(b)
        outer.addWidget(top)

        splitter = QSplitter(Qt.Horizontal); splitter.setChildrenCollapsible(False)
        left = QWidget(); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(7)
        view_head = QHBoxLayout(); view_head.addWidget(QLabel("원본 스캔")); view_head.addStretch()
        self.source_info = QLabel("이미지를 열면 자동 검출을 시작할 수 있어요"); self.source_info.setStyleSheet("color:#899b93;font-size:9pt")
        view_head.addWidget(self.source_info); left_layout.addLayout(view_head)
        self.canvas = ScanCanvas(); left_layout.addWidget(self.canvas, 1)
        footer = QHBoxLayout()
        self.btn_delete = QPushButton("선택 영역 삭제")
        self.btn_redetect = QPushButton("다시 자동 찾기")
        footer.addWidget(self.btn_delete); footer.addWidget(self.btn_redetect); footer.addStretch()
        left_layout.addLayout(footer)

        right = QWidget(); right.setMinimumWidth(350); right.setMaximumWidth(460)
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(7)
        tabs = QTabWidget(); right_layout.addWidget(tabs, 1)
        docs_tab = QWidget(); docs_layout = QVBoxLayout(docs_tab); docs_layout.setContentsMargins(9,9,9,9)
        docs_layout.addWidget(QLabel("검출한 종이 / 카드"))
        self.doc_list = QListWidget(); docs_layout.addWidget(self.doc_list, 1)
        list_buttons = QHBoxLayout(); self.btn_select_all = QPushButton("전체 선택"); self.btn_ai = QPushButton("AI 모델로 찾기")
        list_buttons.addWidget(self.btn_select_all); list_buttons.addWidget(self.btn_ai); docs_layout.addLayout(list_buttons)
        self.ai_status = QLabel("AI 가속: 사용 가능한 모델 없음 · 기본 검출은 CPU에서 동작")
        self.ai_status.setWordWrap(True); self.ai_status.setStyleSheet("color:#7d9489;font-size:8.5pt")
        docs_layout.addWidget(self.ai_status)
        tabs.addTab(docs_tab, "문서 목록")

        adjust_tab = QWidget(); adj_scroll = QScrollArea(); adj_scroll.setWidgetResizable(True); adj_scroll.setFrameShape(QFrame.NoFrame)
        adj_content = QWidget(); adj = QVBoxLayout(adj_content); adj.setContentsMargins(10,10,10,10)
        self.preview = QLabel("영역을 선택하면\n보정 미리보기가 표시됩니다")
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumHeight(260)
        self.preview.setStyleSheet("background:#f5f8f6;border:1px dashed #cddfd4;border-radius:12px;color:#82958d")
        adj.addWidget(self.preview)
        self.slider_controls = {}
        self._add_slider(adj, "밝기", "brightness", -100, 100, 0, "조금씩 밝기 보정")
        self._add_slider(adj, "대비", "contrast", 0, 200, 100, "글자와 종이의 명암")
        self._add_slider(adj, "채도", "saturation", 0, 200, 100, "색상 강도")
        self._add_slider(adj, "노이즈 제거", "denoise", 0, 100, 0, "얼룩·스캔 입자 완화")
        self._add_slider(adj, "먼지 제거", "dust", 0, 100, 0, "작은 점 형태의 먼지 완화")
        reset = QPushButton("보정값 초기화"); reset.clicked.connect(self.reset_adjustments); adj.addWidget(reset)
        adj.addStretch(); adj_scroll.setWidget(adj_content); adj_layout = QVBoxLayout(adjust_tab); adj_layout.setContentsMargins(0,0,0,0); adj_layout.addWidget(adj_scroll)
        tabs.addTab(adjust_tab, "이미지 보정")

        output_tab = QWidget(); out = QVBoxLayout(output_tab); out.setContentsMargins(12,12,12,12)
        out.addWidget(QLabel("저장 형식"))
        self.format_combo = QComboBox(); self.format_combo.addItems(["PNG · 무손실", "JPG · 압축"]); out.addWidget(self.format_combo)
        qrow = QHBoxLayout(); self.quality_slider = QSlider(Qt.Horizontal); self.quality_slider.setRange(40,100); self.quality_slider.setValue(95)
        self.quality_value = QSpinBox(); self.quality_value.setRange(40,100); self.quality_value.setValue(95); self.quality_value.setSuffix("%")
        qrow.addWidget(self.quality_slider,1); qrow.addWidget(self.quality_value); out.addLayout(qrow)
        out.addWidget(QLabel("저장 폴더")); folder_row = QHBoxLayout()
        self.output_folder = QLineEdit(); self.output_folder.setPlaceholderText("원본 옆에 자동 생성")
        self.btn_folder = QPushButton("찾기")
        folder_row.addWidget(self.output_folder,1); folder_row.addWidget(self.btn_folder); out.addLayout(folder_row)
        self.btn_save_selected = QPushButton("선택한 문서 저장"); self.btn_save_selected.setObjectName("mint"); self.btn_save_selected.setMinimumHeight(42)
        self.btn_save_all2 = QPushButton("검출한 문서 모두 저장"); self.btn_save_all2.setObjectName("primary"); self.btn_save_all2.setMinimumHeight(44)
        out.addSpacing(12); out.addWidget(self.btn_save_selected); out.addWidget(self.btn_save_all2)
        note = QLabel("※ 원본 스캔 파일은 변경하지 않습니다.\nJPG 화질은 40~100% 범위에서 지정할 수 있어요.")
        note.setWordWrap(True); note.setStyleSheet("color:#84968f;font-size:9pt"); out.addWidget(note); out.addStretch()
        tabs.addTab(output_tab, "저장 설정")

        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([840, 420]); outer.addWidget(splitter, 1)
        self.status = QLabel("준비 완료 · 이미지 파일을 열어주세요 ♡"); self.status.setStyleSheet("color:#81958b;padding:2px 4px")
        outer.addWidget(self.status)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.auto_detect); self.btn_redetect.clicked.connect(self.auto_detect)
        self.btn_add.clicked.connect(self.start_manual_add); self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_save_all.clicked.connect(self.save_all); self.btn_save_all2.clicked.connect(self.save_all)
        self.btn_save_selected.clicked.connect(self.save_selected); self.btn_folder.clicked.connect(self.choose_output_folder)
        self.doc_list.currentRowChanged.connect(self.select_region); self.btn_select_all.clicked.connect(self.select_all_regions)
        self.btn_ai.clicked.connect(self.load_ai_model)
        self.quality_slider.valueChanged.connect(self.quality_value.setValue)
        self.quality_value.valueChanged.connect(self.quality_slider.setValue)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self._format_changed()
        open_action = QAction("이미지 열기", self); open_action.setShortcut("Ctrl+O"); open_action.triggered.connect(self.open_image); self.addAction(open_action)
        save_action = QAction("모두 저장", self); save_action.setShortcut("Ctrl+Shift+S"); save_action.triggered.connect(self.save_all); self.addAction(save_action)

    def _add_slider(self, layout, label, key, minimum, maximum, initial, hint):
        group = QGroupBox(label); form = QVBoxLayout(group)
        top = QHBoxLayout(); top.addWidget(QLabel(hint)); top.addStretch()
        value = QLabel(str(initial)); value.setStyleSheet("color:#e57b8b;font-weight:800"); top.addWidget(value); form.addLayout(top)
        slider = QSlider(Qt.Horizontal); slider.setRange(minimum, maximum); slider.setValue(initial)
        slider.valueChanged.connect(lambda v, lbl=value: lbl.setText(str(v)))
        slider.valueChanged.connect(lambda _: self.preview_timer.start())
        form.addWidget(slider); layout.addWidget(group)
        self.slider_controls[key] = slider

    def _connect_canvas(self):
        self.canvas.region_selected.connect(self._canvas_select)
        self.canvas.polygon_created.connect(self.add_manual_region)
        self.canvas.polygon_changed.connect(self._canvas_region_changed)

    def _set_controls_enabled(self, enabled):
        for widget in (self.btn_detect, self.btn_add, self.btn_save_all, self.btn_save_all2,
                       self.btn_save_selected, self.btn_delete, self.btn_redetect): widget.setEnabled(enabled)

    def _format_changed(self):
        jpg = self.format_combo.currentIndex() == 1
        self.quality_slider.setEnabled(jpg); self.quality_value.setEnabled(jpg)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "스캔 이미지 열기", "",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not path: return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "열기 실패", "이미지를 읽을 수 없습니다."); return
        self.image = image; self.image_path = path; self.regions.clear(); self.selected = -1
        self.canvas.set_image(bgr_to_pixmap(image)); self.canvas.set_regions([])
        self.doc_list.clear(); self.preview.setText("자동 찾기를 눌러주세요")
        self.output_folder.setText(str(Path(path).with_name(Path(path).stem + "_AlignScan")))
        self.source_info.setText(f"{Path(path).name} · {image.shape[1]} × {image.shape[0]} px")
        self._set_controls_enabled(True); self.status.setText("이미지를 불러왔어요. 자동 찾기를 실행해보세요 ♡")
        self.auto_detect()

    def auto_detect(self):
        if self.image is None: return
        self.status.setText("종이와 카드의 테두리를 찾는 중…")
        self.regions = detect_document_regions(self.image)
        self._refresh_regions()
        if self.regions:
            self.select_region(0)
            self.status.setText(f"{len(self.regions)}개 영역을 찾았어요. 모서리 점을 드래그해 미세 조정할 수 있어요.")
        else:
            self.preview.setText("문서 테두리를 찾지 못했어요.\n'영역 직접 지정'으로 네 모서리를 차례로 찍어주세요.")
            self.status.setText("자동 검출 결과가 없어요. 직접 지정도 가능합니다.")

    def _refresh_regions(self):
        self.doc_list.blockSignals(True); self.doc_list.clear()
        for i, region in enumerate(self.regions):
            x, y = np.min(region.corners, axis=0); xx, yy = np.max(region.corners, axis=0)
            item = QListWidgetItem(f"  {i + 1:02d}  {region.label}\n       {int(xx-x)} × {int(yy-y)} px")
            self.doc_list.addItem(item)
        self.doc_list.blockSignals(False)
        self.canvas.set_regions([r.corners for r in self.regions], self.selected)

    def select_region(self, idx):
        if idx < 0 or idx >= len(self.regions): return
        self.selected = idx
        self.doc_list.blockSignals(True); self.doc_list.setCurrentRow(idx); self.doc_list.blockSignals(False)
        self.canvas.set_regions([r.corners for r in self.regions], idx)
        self.update_preview()

    def _canvas_select(self, idx): self.select_region(idx)

    def _canvas_region_changed(self, idx, corners):
        if 0 <= idx < len(self.regions):
            self.regions[idx].corners = corners
            self.preview_timer.start()

    def start_manual_add(self):
        if self.image is None: return
        self.canvas.set_add_mode(True)
        self.status.setText("문서의 모서리 4곳을 시계 방향으로 차례로 클릭하세요. 시작점은 어디든 괜찮아요.")

    def add_manual_region(self, corners):
        self.regions.append(DocumentRegion(corners, f"직접 지정 {len(self.regions)+1}"))
        self.selected = len(self.regions) - 1
        self._refresh_regions(); self.select_region(self.selected)
        self.status.setText("직접 지정 영역을 추가했어요. 모서리 점을 드래그해 조절할 수 있어요.")

    def delete_selected(self):
        if 0 <= self.selected < len(self.regions):
            self.regions.pop(self.selected); self.selected = min(self.selected, len(self.regions)-1)
            self._refresh_regions()
            if self.selected >= 0: self.select_region(self.selected)
            else: self.preview.setText("영역을 선택해주세요")
            self.status.setText("선택한 영역을 삭제했어요.")

    def select_all_regions(self):
        if self.regions:
            self.doc_list.setCurrentRow(0)
            self.status.setText(f"현재 {len(self.regions)}개 영역이 저장 대상입니다.")

    def _values(self): return {k: v.value() for k, v in self.slider_controls.items()}

    def update_preview(self):
        if self.image is None or not (0 <= self.selected < len(self.regions)): return
        try:
            crop = rectify_document(self.image, self.regions[self.selected].corners)
            crop = enhance_image(crop, **self._values())
            pm = bgr_to_pixmap(crop).scaled(self.preview.width()-12, self.preview.height()-12,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(pm)
        except Exception as exc:
            self.preview.setText(f"미리보기 오류: {exc}")

    def reset_adjustments(self):
        defaults = {"brightness":0,"contrast":100,"saturation":100,"denoise":0,"dust":0}
        for key, value in defaults.items(): self.slider_controls[key].setValue(value)

    def choose_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", self.output_folder.text() or "")
        if path: self.output_folder.setText(path)

    def _output_dir(self):
        text = self.output_folder.text().strip()
        if text: return Path(text)
        if self.image_path: return Path(self.image_path).with_name(Path(self.image_path).stem + "_AlignScan")
        return Path.cwd() / "AlignScan_Output"

    def _export_region(self, idx: int, target: Path):
        region = self.regions[idx]
        crop = rectify_document(self.image, region.corners)
        crop = enhance_image(crop, **self._values())
        fmt = "jpg" if self.format_combo.currentIndex() == 1 else "png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jpg":
            quality = self.quality_value.value()
            ok = cv2.imwrite(str(target.with_suffix(".jpg")), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return target.with_suffix(".jpg") if ok else None
        ok = cv2.imwrite(str(target.with_suffix(".png")), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return target.with_suffix(".png") if ok else None

    def save_selected(self):
        if self.image is None or not (0 <= self.selected < len(self.regions)):
            QMessageBox.information(self, "문서 선택", "먼저 저장할 영역을 선택해주세요."); return
        folder = self._output_dir(); base = Path(self.image_path).stem if self.image_path else "Scan"
        target = folder / f"{base}_{self.selected+1:02d}"
        try:
            path = self._export_region(self.selected, target)
            if path:
                self.status.setText(f"저장 완료: {path}")
                QMessageBox.information(self, "저장 완료", f"개별 이미지로 저장했어요.\n\n{path}")
        except Exception as exc: QMessageBox.critical(self, "저장 오류", str(exc))

    def save_all(self):
        if self.image is None or not self.regions:
            QMessageBox.information(self, "저장할 영역 없음", "먼저 자동 찾기 또는 직접 지정을 해주세요."); return
        folder = self._output_dir(); base = Path(self.image_path).stem if self.image_path else "Scan"
        try:
            results = []
            for idx in range(len(self.regions)):
                path = self._export_region(idx, folder / f"{base}_{idx+1:02d}")
                if path: results.append(path)
            self.status.setText(f"저장 완료 · {len(results)}개 파일")
            QMessageBox.information(self, "저장 완료", f"총 {len(results)}개 파일을 저장했습니다.\n\n저장 위치:\n{folder}")
        except Exception as exc: QMessageBox.critical(self, "저장 오류", str(exc))

    def load_ai_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "문서 검출 ONNX 모델 선택", "", "ONNX 모델 (*.onnx)")
        if not path: return
        try:
            self.ai_detector = OnnxDocumentDetector(path)
            self.ai_status.setText(f"AI 모델 준비 완료 · {self.ai_detector.provider}")
            QMessageBox.information(self, "AI 모델 준비", f"모델을 불러왔습니다.\n실행 공급자: {self.ai_detector.provider}\n\n문서/카드 클래스로 학습된 YOLOv8 Detection ONNX 모델을 사용하세요.")
            self._run_ai_detection()
        except Exception as exc:
            self.ai_detector = None
            QMessageBox.warning(self, "AI 모델을 불러올 수 없음", str(exc))

    def _run_ai_detection(self):
        if self.image is None or self.ai_detector is None: return
        try:
            boxes = self.ai_detector.detect(self.image)
            if not boxes:
                QMessageBox.information(self, "AI 검출", "AI 모델이 문서 영역을 찾지 못했습니다."); return
            for corners in boxes:
                if not any(np.mean(np.linalg.norm(r.corners - corners, axis=1)) < 35 for r in self.regions):
                    self.regions.append(DocumentRegion(corners, "AI 검출"))
            self.selected = 0 if self.regions else -1
            self._refresh_regions()
            if self.selected >= 0: self.select_region(self.selected)
            self.status.setText(f"AI 검출 완료 · {len(boxes)}개 후보 · {self.ai_detector.provider}")
        except Exception as exc: QMessageBox.warning(self, "AI 검출 오류", str(exc))

    def closeEvent(self, event):
        event.accept()
