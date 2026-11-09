from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSplitter, QFrame, QListWidget, QListWidgetItem,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QTabWidget, QScrollArea, QLineEdit)
from .geometry import DocumentRegion, detect_document_regions, rectify_document
from .imaging import enhance_image
from .canvas import ScanCanvas
from .preview import ZoomPreview
from .transforms import (rotate_quarter, rotate_expand, flip_image,
                         transform_points, ruler_correction_angle)


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy())


class MainWindow(QMainWindow):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlignScan · 영수증 정렬 스튜디오")
        self.resize(1380, 900)
        self.setMinimumSize(1020, 700)
        self.setAcceptDrops(True)
        self.image: np.ndarray | None = None
        self.preview_image: np.ndarray | None = None
        self.preview_scale = 1.0
        self.image_path: str | None = None
        self.regions: list[DocumentRegion] = []
        self.selected = -1
        self.ruler_line: tuple[np.ndarray, np.ndarray, str] | None = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(160)
        self.preview_timer.timeout.connect(self.update_preview)
        self.sensitivity_timer = QTimer(self)
        self.sensitivity_timer.setSingleShot(True)
        self.sensitivity_timer.setInterval(350)
        self.sensitivity_timer.timeout.connect(self.auto_detect)
        self._build_ui()
        self._connect_canvas()
        self._set_controls_enabled(False)

    def _build_ui(self):
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#f1f4ef;color:#4b5d54;font-family:'Malgun Gothic','맑은 고딕';font-size:10pt}
        QFrame#topbar{background:#fffdf8;border:1px solid #e3e8df;border-radius:20px}
        QLabel#brand{font-size:21pt;font-weight:900;color:#526e62}
        QLabel#hint{color:#8b9b91;font-size:9pt}
        QPushButton{background:#fffefb;border:1px solid #dce5dc;border-radius:12px;padding:8px 13px;color:#597064;font-weight:700}
        QPushButton:hover{background:#f1f7f1;border-color:#afcbbb}
        QPushButton#primary{background:#91bca3;color:#ffffff;border:0;font-weight:900}
        QPushButton#primary:hover{background:#7fae94}
        QPushButton#mint{background:#d6e8da;color:#526e62;border:1px solid #c4ddcb}
        QGroupBox{border:1px solid #e3e9e0;border-radius:15px;margin-top:12px;padding:12px 9px 8px;font-weight:800;background:#fffefb}
        QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px;color:#658272}
        QListWidget{border:1px solid #e3e9e0;border-radius:13px;background:#fffefc;padding:5px}
        QListWidget::item{padding:10px;border-radius:9px;margin:2px}
        QListWidget::item:selected{background:#f6e9eb;color:#805c64}
        QSlider::groove:horizontal{height:6px;background:#e4ebe2;border-radius:3px}
        QSlider::handle:horizontal{background:#91bca3;border:2px solid white;width:16px;margin:-6px 0;border-radius:9px}
        QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#fffefb;border:1px solid #dce5dc;border-radius:9px;padding:6px}
        QTabWidget::pane{border:1px solid #e3e9e0;border-radius:13px;background:#fffefb}
        QTabBar::tab{padding:9px 14px;background:#edf2eb;border-radius:9px;margin:3px;color:#718579}
        QTabBar::tab:selected{background:#f6e9eb;color:#936b73;font-weight:800}
        """)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(15, 12, 15, 12)
        outer.setSpacing(10)

        top = QFrame(objectName="topbar")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(16, 8, 16, 8)
        brand_col = QVBoxLayout()
        brand = QLabel("AlignScan  ♡", objectName="brand")
        hint = QLabel("Smart Scan · 여러 장의 영수증과 카드를 반듯하게", objectName="hint")
        brand_col.addWidget(brand)
        brand_col.addWidget(hint)
        top_row.addLayout(brand_col)
        mascot = QLabel()
        asset_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets"
        banner = QPixmap(str(asset_root / "mascot_banner.jpg"))
        if banner.isNull():
            banner = QPixmap(str(asset_root / "mascot_banner.png"))
        if not banner.isNull():
            mascot.setPixmap(banner.scaledToHeight(100, Qt.SmoothTransformation))
            mascot.setMaximumWidth(285)
            mascot.setMinimumWidth(175)
            top_row.addWidget(mascot)
        top_row.addStretch()
        self.btn_open = QPushButton("＋ 이미지 열기")
        self.btn_detect = QPushButton("✦ 자동 찾기")
        self.btn_add = QPushButton("✎ 직접 지정")
        self.btn_save_all = QPushButton("모두 개별 저장")
        self.btn_save_all.setObjectName("primary")
        for button in (self.btn_open, self.btn_detect, self.btn_add, self.btn_save_all):
            top_row.addWidget(button)
        outer.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 0, 0)
        view_head = QHBoxLayout()
        view_head.addWidget(QLabel("🖼️ 원본 스캔"))
        view_head.addStretch()
        self.source_info = QLabel("스캔 이미지를 이 창에 끌어놓으면 자동으로 찾아요 ♡")
        self.source_info.setStyleSheet("color:#8887ae;font-size:9pt")
        view_head.addWidget(self.source_info)
        left_col.addLayout(view_head)
        self.canvas = ScanCanvas()
        self.canvas.zoom_changed.connect(
            lambda value: self.canvas_zoom_label.setText(f"작업 영역 {value}%")
            if hasattr(self, "canvas_zoom_label") else None)
        left_col.addWidget(self.canvas, 1)

        canvas_tools = QHBoxLayout()
        self.btn_ruler_h = QPushButton("／ 수평 룰러")
        self.btn_ruler_v = QPushButton("│ 수직 룰러")
        self.btn_apply_ruler = QPushButton("룰러 기준 수평 보정")
        self.btn_clear_ruler = QPushButton("기준선 지우기")
        self.btn_fit_canvas = QPushButton("화면 맞춤")
        self.btn_apply_ruler.setEnabled(False)
        self.btn_clear_ruler.setEnabled(False)
        self.canvas_zoom_label = QLabel("작업 영역 100%")
        self.canvas_zoom_label.setStyleSheet("color:#7773df;font-weight:900")
        canvas_tools.addWidget(self.btn_ruler_h)
        canvas_tools.addWidget(self.btn_ruler_v)
        canvas_tools.addWidget(self.btn_apply_ruler)
        canvas_tools.addWidget(self.btn_clear_ruler)
        canvas_tools.addWidget(self.btn_fit_canvas)
        canvas_tools.addStretch()
        canvas_tools.addWidget(self.canvas_zoom_label)
        left_col.addLayout(canvas_tools)

        foot = QHBoxLayout()
        self.btn_delete = QPushButton("선택 영역 삭제")
        self.btn_redetect = QPushButton("다시 자동 찾기")
        foot.addWidget(self.btn_delete)
        foot.addWidget(self.btn_redetect)
        foot.addStretch()
        left_col.addLayout(foot)

        right = QWidget()
        right.setMinimumWidth(355)
        right.setMaximumWidth(480)
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        right_col.addWidget(tabs, 1)

        docs_tab = QWidget()
        docs = QVBoxLayout(docs_tab)
        docs.setContentsMargins(10, 9, 10, 9)
        docs.addWidget(QLabel("검출한 종이 / 카드"))
        sensitivity = QGroupBox("🎯 자동 찾기 감도")
        sens_col = QVBoxLayout(sensitivity)
        sens_line = QHBoxLayout()
        sens_line.addWidget(QLabel("둔감"))
        sens_line.addStretch()
        self.sensitivity_value = QLabel("62 / 100")
        self.sensitivity_value.setStyleSheet("color:#7773df;font-weight:900")
        sens_line.addWidget(self.sensitivity_value)
        sens_line.addStretch()
        sens_line.addWidget(QLabel("민감"))
        sens_col.addLayout(sens_line)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(62)
        self.sensitivity_slider.valueChanged.connect(
            lambda value: self.sensitivity_value.setText(f"{value} / 100"))
        self.sensitivity_slider.valueChanged.connect(lambda _value: self.sensitivity_timer.start())
        sens_col.addWidget(self.sensitivity_slider)
        sens_hint = QLabel("값을 높이면 흐린 경계와 작은 카드를 더 적극적으로 찾아요.")
        sens_hint.setWordWrap(True)
        sens_hint.setStyleSheet("color:#8887ae;font-size:8.5pt")
        sens_col.addWidget(sens_hint)
        docs.addWidget(sensitivity)
        self.doc_list = QListWidget()
        docs.addWidget(self.doc_list, 1)
        buttons = QHBoxLayout()
        self.btn_select_all = QPushButton("전체 선택")
        self.btn_select_all.setMinimumHeight(38)
        buttons.addWidget(self.btn_select_all)
        docs.addLayout(buttons)
        guide = QLabel(
            "영수증을 선택하면 오른쪽 미리보기에서 확대해 확인할 수 있어요.\n"
            "원본 화면의 색상 모서리 점을 움직여 크롭 경계를 세밀하게 맞춰보세요. ♡")
        guide.setWordWrap(True)
        guide.setAlignment(Qt.AlignCenter)
        guide.setStyleSheet(
            "color:#788d80;background:#f1f6ef;border:1px solid #e2eadf;"
            "border-radius:12px;padding:14px;font-size:9pt;")
        docs.addWidget(guide)
        tabs.addTab(docs_tab, "문서 목록")

        adjust_tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        adjust = QWidget()
        adj = QVBoxLayout(adjust)
        adj.setContentsMargins(10, 10, 10, 10)
        preview_head = QHBoxLayout()
        preview_head.addWidget(QLabel("🔎 정렬 미리보기 · 마우스 휠 확대/축소"))
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color:#7773df;font-weight:900")
        preview_head.addStretch()
        preview_head.addWidget(self.zoom_label)
        adj.addLayout(preview_head)
        self.preview = ZoomPreview()
        self.preview.zoom_changed.connect(lambda value: self.zoom_label.setText(f"{value}%"))
        adj.addWidget(self.preview)
        self.slider_controls = {}
        self._add_slider(adj, "밝기", "brightness", -100, 100, 0, "전체 밝기")
        self._add_slider(adj, "대비", "contrast", 0, 200, 100, "글자와 배경의 명암")
        self._add_slider(adj, "채도", "saturation", 0, 200, 100, "색상 강도")
        self._add_slider(adj, "노이즈 제거", "denoise", 0, 100, 0, "스캔 입자 완화")
        self._add_slider(adj, "먼지 제거", "dust", 0, 100, 0, "작은 점 형태의 먼지 완화")
        self._add_slider(adj, "선명도", "sharpness", 0, 100, 0, "글자 가장자리 또렷하게")
        reset = QPushButton("보정값 초기화")
        reset.clicked.connect(self.reset_adjustments)
        adj.addWidget(reset)
        adj.addStretch()
        scroll.setWidget(adjust)
        adjust_layout = QVBoxLayout(adjust_tab)
        adjust_layout.setContentsMargins(0, 0, 0, 0)
        adjust_layout.addWidget(scroll)
        tabs.addTab(adjust_tab, "이미지 보정")

        rotate_tab = QWidget()
        rotate_col = QVBoxLayout(rotate_tab)
        rotate_col.setContentsMargins(13, 13, 13, 13)

        quick_rotate = QGroupBox("↻ 빠른 회전")
        quick_row = QHBoxLayout(quick_rotate)
        self.btn_rotate_90 = QPushButton("90° ↻")
        self.btn_rotate_180 = QPushButton("180°")
        self.btn_rotate_270 = QPushButton("270° ↻")
        self.btn_rotate_ccw = QPushButton("90° ↺")
        for button in (self.btn_rotate_90, self.btn_rotate_180,
                       self.btn_rotate_270, self.btn_rotate_ccw):
            quick_row.addWidget(button)
        rotate_col.addWidget(quick_rotate)

        flip_group = QGroupBox("↔ 이미지 뒤집기")
        flip_row = QHBoxLayout(flip_group)
        self.btn_flip_h = QPushButton("좌우 반전")
        self.btn_flip_v = QPushButton("상하 반전")
        flip_row.addWidget(self.btn_flip_h)
        flip_row.addWidget(self.btn_flip_v)
        rotate_col.addWidget(flip_group)

        angle_group = QGroupBox("🎚 수동 각도 조절")
        angle_col = QVBoxLayout(angle_group)
        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("각도"))
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-1800, 1800)
        self.angle_slider.setValue(0)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(0.5)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setValue(0.0)
        angle_row.addWidget(self.angle_slider, 1)
        angle_row.addWidget(self.angle_spin)
        angle_col.addLayout(angle_row)
        angle_hint = QLabel("양수는 반시계 방향, 음수는 시계 방향입니다. 0.1도 단위로 조절할 수 있어요.")
        angle_hint.setStyleSheet("color:#777da8;font-size:8.5pt")
        angle_hint.setWordWrap(True)
        angle_col.addWidget(angle_hint)
        angle_actions = QHBoxLayout()
        self.btn_apply_angle = QPushButton("각도 적용")
        self.btn_reset_angle = QPushButton("값 초기화")
        angle_actions.addWidget(self.btn_apply_angle)
        angle_actions.addWidget(self.btn_reset_angle)
        angle_col.addLayout(angle_actions)
        self.angle_slider.valueChanged.connect(lambda value: self.angle_spin.setValue(value / 10.0))
        self.angle_spin.valueChanged.connect(lambda value: self.angle_slider.setValue(round(value * 10)))
        rotate_col.addWidget(angle_group)

        ruler_help = QLabel(
            "룰러 사용법: 수평 또는 수직 룰러를 선택한 뒤 작업 이미지 위에서 "
            "기준으로 삼을 선을 마우스로 드래그하세요. 그 선이 수평/수직이 되도록 "
            "이미지 전체를 회전합니다. 작업 캔버스에서는 휠로 확대·축소하고 "
            "마우스 가운데 버튼 또는 오른쪽 버튼으로 화면을 이동할 수 있습니다.")
        ruler_help.setWordWrap(True)
        ruler_help.setStyleSheet("color:#777da8;padding:10px;background:#f6f3ff;border-radius:10px")
        rotate_col.addWidget(ruler_help)
        rotate_col.addStretch()
        tabs.addTab(rotate_tab, "회전·룰러")

        output_tab = QWidget()
        out = QVBoxLayout(output_tab)
        out.setContentsMargins(13, 13, 13, 13)
        out.addWidget(QLabel("저장 형식"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG · 무손실", "JPG · 압축"])
        out.addWidget(self.format_combo)
        quality_row = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(40, 100)
        self.quality_slider.setValue(95)
        self.quality_value = QSpinBox()
        self.quality_value.setRange(40, 100)
        self.quality_value.setValue(95)
        self.quality_value.setSuffix("%")
        quality_row.addWidget(self.quality_slider, 1)
        quality_row.addWidget(self.quality_value)
        out.addLayout(quality_row)
        out.addWidget(QLabel("저장 폴더"))
        folder_row = QHBoxLayout()
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText("원본 옆에 자동 생성")
        self.btn_folder = QPushButton("찾기")
        folder_row.addWidget(self.output_folder, 1)
        folder_row.addWidget(self.btn_folder)
        out.addLayout(folder_row)
        self.btn_save_selected = QPushButton("선택한 문서 저장")
        self.btn_save_selected.setObjectName("mint")
        self.btn_save_selected.setMinimumHeight(42)
        self.btn_save_all2 = QPushButton("검출한 문서 모두 저장")
        self.btn_save_all2.setObjectName("primary")
        self.btn_save_all2.setMinimumHeight(44)
        out.addSpacing(12)
        out.addWidget(self.btn_save_selected)
        out.addWidget(self.btn_save_all2)
        note = QLabel("원본은 유지됩니다. JPG 화질은 40~100%까지 지정할 수 있어요.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8887ae;font-size:9pt")
        out.addWidget(note)
        out.addStretch()
        tabs.addTab(output_tab, "저장 설정")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([850, 440])
        outer.addWidget(splitter, 1)
        self.status = QLabel("준비 완료 · 파일을 열거나 끌어놓아 주세요 ♡")
        self.status.setStyleSheet("color:#777da8;padding:3px 4px")
        outer.addWidget(self.status)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.auto_detect)
        self.btn_redetect.clicked.connect(self.auto_detect)
        self.btn_add.clicked.connect(self.start_manual_add)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_save_all2.clicked.connect(self.save_all)
        self.btn_save_selected.clicked.connect(self.save_selected)
        self.btn_folder.clicked.connect(self.choose_output_folder)
        self.doc_list.currentRowChanged.connect(self.select_region)
        self.btn_select_all.clicked.connect(self.select_all_regions)
        self.btn_ruler_h.clicked.connect(lambda: self._start_ruler("horizontal"))
        self.btn_ruler_v.clicked.connect(lambda: self._start_ruler("vertical"))
        self.btn_apply_ruler.clicked.connect(self._apply_ruler)
        self.btn_clear_ruler.clicked.connect(self._clear_ruler)
        self.btn_fit_canvas.clicked.connect(self.canvas.fit_to_window)
        self.btn_rotate_90.clicked.connect(lambda: self.rotate_by_quarter(90))
        self.btn_rotate_180.clicked.connect(lambda: self.rotate_by_quarter(180))
        self.btn_rotate_270.clicked.connect(lambda: self.rotate_by_quarter(270))
        self.btn_rotate_ccw.clicked.connect(lambda: self.rotate_by_quarter(270))
        self.btn_flip_h.clicked.connect(lambda: self.flip_current_image(True))
        self.btn_flip_v.clicked.connect(lambda: self.flip_current_image(False))
        self.btn_apply_angle.clicked.connect(self.apply_custom_rotation)
        self.btn_reset_angle.clicked.connect(lambda: (self.angle_spin.setValue(0), self.angle_slider.setValue(0)))
        self.quality_slider.valueChanged.connect(self.quality_value.setValue)
        self.quality_value.valueChanged.connect(self.quality_slider.setValue)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        open_action = QAction("이미지 열기", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        self.addAction(open_action)
        save_action = QAction("모두 저장", self)
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self.save_all)
        self.addAction(save_action)
        self._format_changed()

    def _add_slider(self, layout, label, key, minimum, maximum, initial, hint):
        group = QGroupBox(label)
        form = QVBoxLayout(group)
        line = QHBoxLayout()
        line.addWidget(QLabel(hint))
        line.addStretch()
        value_label = QLabel(str(initial))
        value_label.setStyleSheet("color:#7773df;font-weight:900")
        line.addWidget(value_label)
        form.addLayout(line)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(initial)
        slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
        slider.valueChanged.connect(lambda _value: self.preview_timer.start())
        form.addWidget(slider)
        layout.addWidget(group)
        self.slider_controls[key] = slider

    def _connect_canvas(self):
        self.canvas.region_selected.connect(self._canvas_select)
        self.canvas.polygon_created.connect(self.add_manual_region)
        self.canvas.polygon_changed.connect(self._canvas_region_changed)
        self.canvas.image_dropped.connect(self._load_image)
        self.canvas.ruler_created.connect(self._on_ruler_created)

    def _set_controls_enabled(self, enabled: bool):
        for widget in (self.btn_detect, self.btn_add, self.btn_save_all, self.btn_save_all2,
                       self.btn_save_selected, self.btn_delete, self.btn_redetect,
                       self.btn_select_all, self.sensitivity_slider,
                       self.btn_ruler_h, self.btn_ruler_v, self.btn_apply_ruler,
                       self.btn_clear_ruler, self.btn_rotate_90, self.btn_rotate_180,
                       self.btn_rotate_270, self.btn_rotate_ccw, self.btn_flip_h,
                       self.btn_flip_v, self.angle_slider, self.angle_spin,
                       self.btn_apply_angle, self.btn_reset_angle):
            widget.setEnabled(enabled)
    def _format_changed(self):
        is_jpg = self.format_combo.currentIndex() == 1
        self.quality_slider.setEnabled(is_jpg)
        self.quality_value.setEnabled(is_jpg)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "스캔 이미지 열기", "",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if path:
            self._load_image(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in self.IMAGE_EXTS
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in self.IMAGE_EXTS:
                self._load_image(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()

    def _load_image(self, path: str):
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None:
            QMessageBox.warning(self, "열기 실패", "이미지를 읽을 수 없습니다.")
            return
        self.image = image
        self.image_path = path
        self.ruler_line = None
        self.canvas.clear_ruler_line()
        self.canvas.set_add_mode(False)
        self.canvas.set_ruler_mode(None)
        self.btn_apply_ruler.setEnabled(False)
        self.btn_clear_ruler.setEnabled(False)
        self._preview_region = -1
        self.regions = []
        self.selected = -1
        height, width = image.shape[:2]
        self.preview_scale = min(1.0, 1200.0 / max(height, width))
        self.preview_image = (
            cv2.resize(image, None, fx=self.preview_scale, fy=self.preview_scale,
                       interpolation=cv2.INTER_AREA)
            if self.preview_scale < 1.0 else image
        )
        self.canvas.set_image(bgr_to_pixmap(image))
        self.canvas.set_regions([])
        self.doc_list.clear()
        self.preview.set_message("종이와 카드 테두리를 찾고 있어요 ♡")
        self.output_folder.setText(str(Path(path).with_name(Path(path).stem + "_AlignScan")))
        self.source_info.setText(f"{Path(path).name} · {width} × {height} px")
        self._set_controls_enabled(True)
        self.status.setText("이미지를 불러왔어요. 자동 정렬 중…")
        self.auto_detect()

    def auto_detect(self):
        if self.image is None:
            return
        self.status.setText("종이와 카드의 테두리를 찾는 중…")
        self.regions = detect_document_regions(self.image, sensitivity=self.sensitivity_slider.value())
        self._refresh_regions()
        if self.regions:
            self.select_region(0)
            self.status.setText(
                f"{len(self.regions)}개 영역 자동 정렬 완료 ♡ · 모서리 점을 드래그해 수정할 수 있어요.")
        else:
            self.preview.set_message("문서 테두리를 찾지 못했어요. 감도를 높이거나 직접 지정해보세요.")
            self.status.setText("자동 검출 결과가 없어요. 직접 지정도 가능합니다.")

    def _refresh_regions(self):
        self.doc_list.blockSignals(True)
        self.doc_list.clear()
        for index, region in enumerate(self.regions):
            x, y = np.min(region.corners, axis=0)
            xx, yy = np.max(region.corners, axis=0)
            self.doc_list.addItem(QListWidgetItem(
                f"  {index + 1:02d}  {region.label}\n       {int(xx-x)} × {int(yy-y)} px"))
        self.doc_list.blockSignals(False)
        self.canvas.set_regions([region.corners for region in self.regions], self.selected)

    def select_region(self, index: int):
        if index < 0 or index >= len(self.regions):
            return
        self.selected = index
        self.doc_list.blockSignals(True)
        self.doc_list.setCurrentRow(index)
        self.doc_list.blockSignals(False)
        self.canvas.set_regions([region.corners for region in self.regions], index)
        self.update_preview()

    def _canvas_select(self, index):
        self.select_region(index)

    def _canvas_region_changed(self, index, corners):
        if 0 <= index < len(self.regions):
            self.regions[index].corners = corners
            self.preview_timer.start()

    def start_manual_add(self):
        if self.image is None:
            return
        self.canvas.set_add_mode(True)
        self.status.setText("문서 모서리 네 곳을 시계 방향으로 차례로 클릭하세요.")

    def add_manual_region(self, corners):
        self.regions.append(DocumentRegion(corners, f"직접 지정 {len(self.regions) + 1}"))
        self.selected = len(self.regions) - 1
        self._refresh_regions()
        self.select_region(self.selected)

    def delete_selected(self):
        if not (0 <= self.selected < len(self.regions)):
            return
        self.regions.pop(self.selected)
        self.selected = min(self.selected, len(self.regions) - 1)
        self._refresh_regions()
        if self.selected >= 0:
            self.select_region(self.selected)
        else:
            self.preview.set_message("영역을 선택해주세요.")
        self.status.setText("선택한 영역을 삭제했습니다.")

    def select_all_regions(self):
        if self.regions:
            self.doc_list.setCurrentRow(0)
            self.status.setText(f"현재 {len(self.regions)}개 영역이 저장 대상입니다.")

    def _values(self):
        return {key: slider.value() for key, slider in self.slider_controls.items()}

    def update_preview(self):
        if self.preview_image is None or not (0 <= self.selected < len(self.regions)):
            return
        try:
            corners = self.regions[self.selected].corners * self.preview_scale
            crop = rectify_document(self.preview_image, corners, interpolation=cv2.INTER_LINEAR)
            crop = enhance_image(crop, **self._values(), fast=True)
            changed_region = getattr(self, "_preview_region", -1) != self.selected
            self.preview.set_preview(bgr_to_pixmap(crop), reset_zoom=changed_region)
            self._preview_region = self.selected
        except Exception as exc:
            self.preview.set_message(f"미리보기 오류: {exc}")

    def reset_adjustments(self):
        defaults = {"brightness": 0, "contrast": 100, "saturation": 100,
                    "denoise": 0, "dust": 0, "sharpness": 0}
        for key, value in defaults.items():
            self.slider_controls[key].setValue(value)

    def choose_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", self.output_folder.text() or "")
        if path:
            self.output_folder.setText(path)

    def _output_dir(self):
        value = self.output_folder.text().strip()
        if value:
            return Path(value)
        if self.image_path:
            return Path(self.image_path).with_name(Path(self.image_path).stem + "_AlignScan")
        return Path.cwd() / "AlignScan_Output"

    def _export_region(self, index: int, target: Path):
        crop = rectify_document(self.image, self.regions[index].corners)
        crop = enhance_image(crop, **self._values(), fast=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.format_combo.currentIndex() == 1:
            output_path = target.with_suffix(".jpg")
            ok, encoded = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.quality_value.value()])
        else:
            output_path = target.with_suffix(".png")
            ok, encoded = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return None
        encoded.tofile(str(output_path))
        return output_path

    def save_selected(self):
        if self.image is None or not (0 <= self.selected < len(self.regions)):
            QMessageBox.information(self, "문서 선택", "먼저 저장할 영역을 선택해주세요.")
            return
        folder = self._output_dir()
        base = Path(self.image_path).stem if self.image_path else "Scan"
        try:
            path = self._export_region(self.selected, folder / f"{base}_{self.selected + 1:02d}")
            if path:
                self.status.setText(f"저장 완료: {path}")
                QMessageBox.information(self, "저장 완료", str(path))
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def save_all(self):
        if self.image is None or not self.regions:
            QMessageBox.information(self, "저장할 영역 없음", "먼저 자동 찾기 또는 직접 지정을 해주세요.")
            return
        folder = self._output_dir()
        base = Path(self.image_path).stem if self.image_path else "Scan"
        try:
            results = []
            for index in range(len(self.regions)):
                path = self._export_region(index, folder / f"{base}_{index + 1:02d}")
                if path:
                    results.append(path)
            self.status.setText(f"저장 완료 · {len(results)}개 파일")
            QMessageBox.information(self, "저장 완료",
                                    f"총 {len(results)}개 파일을 저장했습니다.\n\n{folder}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def _start_ruler(self, orientation: str):
        if self.image is None:
            return
        self.canvas.set_ruler_mode(orientation)
        label = "수평" if orientation == "horizontal" else "수직"
        self.status.setText(f"{label} 기준 룰러를 작업 이미지 위에서 마우스로 드래그해 그어주세요.")

    def _on_ruler_created(self, start, end, orientation: str):
        self.ruler_line = (
            np.asarray(start, dtype=np.float32).copy(),
            np.asarray(end, dtype=np.float32).copy(),
            orientation,
        )
        self.canvas.set_ruler_line(*self.ruler_line)
        self.btn_apply_ruler.setEnabled(True)
        self.btn_clear_ruler.setEnabled(True)
        self.status.setText("기준선이 만들어졌습니다. '룰러 기준 수평 보정'을 눌러 기울기를 바로잡으세요.")

    def _clear_ruler(self):
        self.ruler_line = None
        self.canvas.clear_ruler_line()
        self.btn_apply_ruler.setEnabled(False)
        self.btn_clear_ruler.setEnabled(False)
        self.status.setText("룰러 기준선을 지웠습니다.")

    def _refresh_transformed_image(self, new_image: np.ndarray, matrix: np.ndarray,
                                   description: str):
        if self.image is None:
            return
        was_empty = not self.regions
        transformed = []
        for region in self.regions:
            corners = transform_points(region.corners, matrix)
            transformed.append(DocumentRegion(corners, region.label))
        self.image = new_image
        self.regions = transformed
        self.ruler_line = None
        self.canvas.clear_ruler_line()
        self.canvas.set_add_mode(False)
        self.canvas.set_ruler_mode(None)
        self.btn_apply_ruler.setEnabled(False)
        self.btn_clear_ruler.setEnabled(False)

        height, width = new_image.shape[:2]
        self.preview_scale = min(1.0, 1200.0 / max(height, width))
        self.preview_image = (
            cv2.resize(new_image, None, fx=self.preview_scale, fy=self.preview_scale,
                       interpolation=cv2.INTER_AREA)
            if self.preview_scale < 1.0 else new_image
        )
        self.canvas.set_image(bgr_to_pixmap(new_image))
        self._preview_region = -1
        self._refresh_regions()
        self.source_info.setText(f"{Path(self.image_path).name if self.image_path else '이미지'} · {width} × {height} px")
        self.canvas.fit_to_window()

        if self.regions:
            self.selected = min(max(self.selected, 0), len(self.regions) - 1)
            self.select_region(self.selected)
        elif was_empty:
            self.selected = -1
            self.auto_detect()
        self.status.setText(f"{description} 적용 완료 · 모서리 위치도 함께 변환했습니다.")

    def rotate_by_quarter(self, clockwise_degrees: int):
        if self.image is None:
            return
        try:
            new_image, matrix = rotate_quarter(self.image, clockwise_degrees)
            self._refresh_transformed_image(new_image, matrix, f"{clockwise_degrees}도 회전")
            self.angle_spin.setValue(0)
            self.angle_slider.setValue(0)
        except Exception as exc:
            QMessageBox.warning(self, "회전 오류", str(exc))

    def flip_current_image(self, horizontal: bool):
        if self.image is None:
            return
        try:
            new_image, matrix = flip_image(self.image, horizontal=horizontal)
            description = "좌우 반전" if horizontal else "상하 반전"
            self._refresh_transformed_image(new_image, matrix, description)
        except Exception as exc:
            QMessageBox.warning(self, "뒤집기 오류", str(exc))

    def apply_custom_rotation(self):
        if self.image is None:
            return
        angle = float(self.angle_spin.value())
        if abs(angle) < 0.05:
            return
        try:
            new_image, matrix = rotate_expand(self.image, angle)
            self._refresh_transformed_image(new_image, matrix, f"수동 {angle:g}도 회전")
            self.angle_spin.setValue(0)
            self.angle_slider.setValue(0)
        except Exception as exc:
            QMessageBox.warning(self, "회전 오류", str(exc))

    def _apply_ruler(self):
        if self.image is None or self.ruler_line is None:
            return
        start, end, orientation = self.ruler_line
        try:
            angle = ruler_correction_angle(start, end, orientation)
            if abs(angle) < 0.05:
                self._clear_ruler()
                self.status.setText("기준선이 이미 수평/수직에 가깝습니다.")
                return
            new_image, matrix = rotate_expand(self.image, angle)
            label = "수평" if orientation == "horizontal" else "수직"
            self._refresh_transformed_image(new_image, matrix, f"{label} 룰러 보정 ({angle:+.2f}도)")
        except Exception as exc:
            QMessageBox.warning(self, "룰러 보정 오류", str(exc))

    def closeEvent(self, event):
        event.accept()
