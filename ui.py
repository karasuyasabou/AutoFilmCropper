"""PySide6 graphics core for interactive crop review (step-2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QBrush,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class CropState:
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float


class CropRectItem(QGraphicsObject):
    """Interactive crop item: move + anchored-edge resize + rotate."""

    changed = Signal()

    def __init__(self, width: float = 400, height: float = 260) -> None:
        super().__init__()
        self._width = max(10.0, width)
        self._height = max(10.0, height)
        self._aspect_ratio: Optional[float] = self._width / self._height

        self._mode = "none"  # none|move|resize
        self._resize_edges = {"l": False, "r": False, "t": False, "b": False}
        self._drag_start = QPointF()
        self._start_pos = QPointF()
        self._start_w = self._width
        self._start_h = self._height

        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)

    def boundingRect(self) -> QRectF:
        m = 2.0
        return QRectF(-self._width / 2 - m, -self._height / 2 - m, self._width + 2 * m, self._height + 2 * m)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 220, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rect = QRectF(-self._width / 2, -self._height / 2, self._width, self._height)
        painter.drawRect(rect)

    def set_aspect_ratio(self, aspect_ratio: Optional[float]) -> None:
        """None means free ratio."""
        self._aspect_ratio = aspect_ratio
        if aspect_ratio is None or aspect_ratio <= 0:
            self.changed.emit()
            return

        current_area = max(20.0 * 20.0, self._width * self._height)
        new_w = max(20.0, (current_area * aspect_ratio) ** 0.5)
        new_h = max(20.0, new_w / aspect_ratio)

        self.prepareGeometryChange()
        self._width = new_w
        self._height = new_h
        self.update()
        self.changed.emit()

    def set_angle(self, angle_deg: float) -> None:
        angle_deg = max(-15.0, min(15.0, angle_deg))
        if abs(angle_deg - self.rotation()) < 1e-6:
            return
        self.setRotation(angle_deg)
        self.changed.emit()

    def nudge(self, dx: float, dy: float) -> None:
        self.setPos(self.pos() + QPointF(dx, dy))
        self.changed.emit()

    def current_state(self) -> CropState:
        p = self.pos()
        return CropState(cx=p.x(), cy=p.y(), width=self._width, height=self._height, angle_deg=self.rotation())

    def set_state(self, cx: float, cy: float, width: float, height: float, angle_deg: float) -> None:
        self.prepareGeometryChange()
        self._width = max(20.0, float(width))
        self._height = max(20.0, float(height))
        self.setPos(float(cx), float(cy))
        self.setRotation(max(-15.0, min(15.0, float(angle_deg))))
        self.update()
        self.changed.emit()

    def polygon_in_scene(self):
        rect = QRectF(-self._width / 2, -self._height / 2, self._width, self._height)
        return self.mapToScene(rect)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._drag_start = event.pos()
        self._start_pos = QPointF(self.pos())
        self._start_w = self._width
        self._start_h = self._height
        self._resize_edges = self._detect_edges(event.pos())

        if any(self._resize_edges.values()):
            self._mode = "resize"
        else:
            self._mode = "move"

        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._mode == "move":
            delta_scene = event.scenePos() - event.lastScenePos()
            self.setPos(self.pos() + delta_scene)
            self.changed.emit()
            event.accept()
            return

        if self._mode == "resize":
            delta = event.pos() - self._drag_start
            self._resize_from_drag(delta)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._mode = "none"
        event.accept()

    def hoverMoveEvent(self, event) -> None:
        edges = self._detect_edges(event.pos())
        if (edges["l"] and edges["t"]) or (edges["r"] and edges["b"]):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (edges["r"] and edges["t"]) or (edges["l"] and edges["b"]):
            self.setCursor(Qt.SizeBDiagCursor)
        elif edges["l"] or edges["r"]:
            self.setCursor(Qt.SizeHorCursor)
        elif edges["t"] or edges["b"]:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def _detect_edges(self, p: QPointF, margin: float = 10.0):
        left = -self._width / 2
        right = self._width / 2
        top = -self._height / 2
        bottom = self._height / 2
        return {
            "l": abs(p.x() - left) <= margin,
            "r": abs(p.x() - right) <= margin,
            "t": abs(p.y() - top) <= margin,
            "b": abs(p.y() - bottom) <= margin,
        }

    def _resize_from_drag(self, delta: QPointF) -> None:
        min_size = 20.0

        left = -self._start_w / 2
        right = self._start_w / 2
        top = -self._start_h / 2
        bottom = self._start_h / 2

        if self._aspect_ratio and self._aspect_ratio > 0:
            left, right, top, bottom = self._resize_with_aspect(
                left,
                right,
                top,
                bottom,
                delta,
                self._aspect_ratio,
                min_size,
            )
        else:
            if self._resize_edges["l"]:
                left = min(left + delta.x(), right - min_size)
            if self._resize_edges["r"]:
                right = max(right + delta.x(), left + min_size)
            if self._resize_edges["t"]:
                top = min(top + delta.y(), bottom - min_size)
            if self._resize_edges["b"]:
                bottom = max(bottom + delta.y(), top + min_size)

        new_w = max(min_size, right - left)
        new_h = max(min_size, bottom - top)
        local_center_shift = QPointF((left + right) / 2.0, (top + bottom) / 2.0)

        angle_rad = math.radians(self.rotation())
        scene_shift = QPointF(
            local_center_shift.x() * math.cos(angle_rad) - local_center_shift.y() * math.sin(angle_rad),
            local_center_shift.x() * math.sin(angle_rad) + local_center_shift.y() * math.cos(angle_rad),
        )

        self.prepareGeometryChange()
        self._width = new_w
        self._height = new_h
        self.setPos(self._start_pos + scene_shift)
        self.update()
        self.changed.emit()

    def _resize_with_aspect(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        delta: QPointF,
        aspect_ratio: float,
        min_size: float,
    ) -> tuple[float, float, float, float]:
        has_horizontal = self._resize_edges["l"] or self._resize_edges["r"]
        has_vertical = self._resize_edges["t"] or self._resize_edges["b"]

        if has_horizontal and has_vertical:
            return self._resize_corner_with_aspect(
                left,
                right,
                top,
                bottom,
                delta,
                aspect_ratio,
                min_size,
            )
        if has_horizontal:
            fixed_x = right if self._resize_edges["l"] else left
            dragged_x = left + delta.x() if self._resize_edges["l"] else right + delta.x()
            new_w = max(min_size, abs(fixed_x - dragged_x))
            new_h = max(min_size, new_w / aspect_ratio)
            center_y = (top + bottom) / 2.0

            if self._resize_edges["l"]:
                left = fixed_x - new_w
                right = fixed_x
            else:
                left = fixed_x
                right = fixed_x + new_w

            top = center_y - new_h / 2.0
            bottom = center_y + new_h / 2.0
            return left, right, top, bottom

        fixed_y = bottom if self._resize_edges["t"] else top
        dragged_y = top + delta.y() if self._resize_edges["t"] else bottom + delta.y()
        new_h = max(min_size, abs(fixed_y - dragged_y))
        new_w = max(min_size, new_h * aspect_ratio)
        center_x = (left + right) / 2.0

        if self._resize_edges["t"]:
            top = fixed_y - new_h
            bottom = fixed_y
        else:
            top = fixed_y
            bottom = fixed_y + new_h

        left = center_x - new_w / 2.0
        right = center_x + new_w / 2.0
        return left, right, top, bottom

    def _resize_corner_with_aspect(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        delta: QPointF,
        aspect_ratio: float,
        min_size: float,
    ) -> tuple[float, float, float, float]:
        fixed_x = right if self._resize_edges["l"] else left
        fixed_y = bottom if self._resize_edges["t"] else top

        dragged_x = left + delta.x() if self._resize_edges["l"] else right + delta.x()
        dragged_y = top + delta.y() if self._resize_edges["t"] else bottom + delta.y()

        candidate_w = max(min_size, abs(fixed_x - dragged_x))
        candidate_h = max(min_size, abs(fixed_y - dragged_y))

        dw = abs(candidate_w - self._start_w) / max(1.0, self._start_w)
        dh = abs(candidate_h - self._start_h) / max(1.0, self._start_h)

        if dw >= dh:
            new_w = candidate_w
            new_h = max(min_size, new_w / aspect_ratio)
        else:
            new_h = candidate_h
            new_w = max(min_size, new_h * aspect_ratio)

        if self._resize_edges["l"]:
            left = fixed_x - new_w
            right = fixed_x
        else:
            left = fixed_x
            right = fixed_x + new_w

        if self._resize_edges["t"]:
            top = fixed_y - new_h
            bottom = fixed_y
        else:
            top = fixed_y
            bottom = fixed_y + new_h

        return left, right, top, bottom


class MaskOverlayItem(QGraphicsObject):
    """Darken overlay with a transparent rotated-rect hole."""

    def __init__(self, scene: QGraphicsScene, crop_item: CropRectItem) -> None:
        super().__init__()
        self._scene = scene
        self._crop_item = crop_item
        self._color = QColor(0, 0, 0, 128)
        self._scene_rect = QRectF(scene.sceneRect())
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(10_000)

        self._scene.sceneRectChanged.connect(self._on_scene_rect_changed)
        self._crop_item.changed.connect(self.update)

    def _on_scene_rect_changed(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._scene_rect = QRectF(rect)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(self._scene_rect)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        scene_rect = self._scene_rect
        full = QPainterPath()
        full.addRect(scene_rect)

        crop_poly = self._crop_item.polygon_in_scene()
        hole = QPainterPath()
        hole.addPolygon(crop_poly)

        combined = QPainterPath()
        combined.setFillRule(Qt.OddEvenFill)
        combined.addPath(full)
        combined.addPath(hole)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawPath(combined)


class FilmGraphicsView(QGraphicsView):
    """Graphics view that hosts image, crop item and mask overlay."""

    approveRequested = Signal()
    sampleRectSelected = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.scene_obj.addItem(self.image_item)

        self.crop_item = CropRectItem(width=700, height=466)
        self.crop_item.setZValue(100)
        self.crop_item.setAcceptHoverEvents(True)
        self.scene_obj.addItem(self.crop_item)

        self.export_preview_item = QGraphicsRectItem(self.crop_item)
        self.export_preview_item.setZValue(110)
        self.export_preview_item.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
        self.export_preview_item.setBrush(Qt.NoBrush)
        self.export_preview_item.setVisible(False)

        self.mask_item = MaskOverlayItem(self.scene_obj, self.crop_item)
        self.scene_obj.addItem(self.mask_item)

        self.sample_rect_item = QGraphicsRectItem()
        self.sample_rect_item.setZValue(20_000)
        self.sample_rect_item.setPen(QPen(QColor(255, 0, 255), 2, Qt.DashLine))
        self.sample_rect_item.setBrush(QBrush(QColor(255, 0, 255, 35)))
        self.sample_rect_item.setVisible(False)
        self.scene_obj.addItem(self.sample_rect_item)

        self._sampling_mode = False
        self._saved_crop_mouse_buttons = Qt.LeftButton
        self._saved_crop_hover_enabled = True
        self._zoom = 1.0
        self._export_inset_preview_ratio = 0.0
        self.set_crop_visible(False)
        self.crop_item.changed.connect(self._update_export_preview)

    def set_proxy_image(self, image: QImage) -> None:
        pix = QPixmap.fromImage(image)
        self.image_item.setPixmap(pix)

        rect = QRectF(0, 0, pix.width(), pix.height())
        self.scene_obj.setSceneRect(rect)

        self.crop_item.setPos(rect.center())
        max_w = rect.width() * 0.75
        max_h = rect.height() * 0.75
        self.crop_item.prepareGeometryChange()
        # keep current aspect, reset to a visible size
        ar = self.crop_item.current_state().width / max(1.0, self.crop_item.current_state().height)
        w = min(max_w, max_h * ar)
        h = w / ar
        if h > max_h:
            h = max_h
            w = h * ar
        self.crop_item._width = max(50.0, w)
        self.crop_item._height = max(50.0, h)
        self.crop_item.update()
        self.crop_item.changed.emit()

        self.fitInView(rect, Qt.KeepAspectRatio)

    def set_crop_state(self, state) -> None:
        if state is None:
            self.set_crop_visible(False)
            return

        self.set_crop_visible(True)
        self.crop_item.set_state(
            cx=state.cx,
            cy=state.cy,
            width=state.width,
            height=state.height,
            angle_deg=state.angle_deg,
        )

    def current_crop_state(self) -> CropState:
        return self.crop_item.current_state()

    def has_crop_state(self) -> bool:
        return self.crop_item.isVisible()

    def set_crop_visible(self, visible: bool) -> None:
        self.crop_item.setVisible(visible)
        self.mask_item.setVisible(visible)
        self._update_export_preview()

    def set_export_inset_preview_ratio(self, inset_ratio_per_side: float) -> None:
        self._export_inset_preview_ratio = max(0.0, min(0.05, float(inset_ratio_per_side)))
        self._update_export_preview()

    def set_aspect_ratio(self, aspect_ratio: Optional[float]) -> None:
        self.crop_item.set_aspect_ratio(aspect_ratio)

    def start_sample_selection(self) -> None:
        self._sampling_mode = True
        self._saved_crop_mouse_buttons = self.crop_item.acceptedMouseButtons()
        self._saved_crop_hover_enabled = self.crop_item.acceptHoverEvents()
        self.crop_item.setAcceptedMouseButtons(Qt.NoButton)
        self.crop_item.setAcceptHoverEvents(False)
        self.crop_item.unsetCursor()
        self.viewport().setCursor(Qt.CrossCursor)

    def cancel_sample_selection(self) -> None:
        self._sampling_mode = False
        self.crop_item.setAcceptedMouseButtons(self._saved_crop_mouse_buttons)
        self.crop_item.setAcceptHoverEvents(self._saved_crop_hover_enabled)
        self.crop_item.unsetCursor()
        self.viewport().unsetCursor()

    def set_sample_rect(self, rect: Optional[tuple[int, int, int, int]]) -> None:
        if rect is None:
            self.sample_rect_item.setVisible(False)
            self.sample_rect_item.setRect(QRectF())
            return

        x, y, w, h = rect
        self.sample_rect_item.setRect(QRectF(float(x), float(y), float(w), float(h)))
        self.sample_rect_item.setVisible(True)

    def mousePressEvent(self, event) -> None:
        if self._sampling_mode and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            rect = self._build_sample_rect(scene_pos)
            self.cancel_sample_selection()
            self.set_sample_rect(rect)
            self.sampleRectSelected.emit(rect)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._sampling_mode:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._sampling_mode:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if self._sampling_mode and key == Qt.Key_Escape:
            self.cancel_sample_selection()
            event.accept()
            return
        if key == Qt.Key_Q and self.has_crop_state():
            self.crop_item.set_angle(self.crop_item.rotation() - 0.1)
            event.accept()
            return
        if key == Qt.Key_E and self.has_crop_state():
            self.crop_item.set_angle(self.crop_item.rotation() + 0.1)
            event.accept()
            return
        if key == Qt.Key_A and self.has_crop_state():
            self.crop_item.nudge(-1, 0)
            event.accept()
            return
        if key == Qt.Key_D and self.has_crop_state():
            self.crop_item.nudge(1, 0)
            event.accept()
            return
        if key == Qt.Key_W and self.has_crop_state():
            self.crop_item.nudge(0, -1)
            event.accept()
            return
        if key == Qt.Key_S and self.has_crop_state():
            self.crop_item.nudge(0, 1)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.approveRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_sample_rect(self, center_scene: QPointF, sample_size: int = 12) -> tuple[int, int, int, int]:
        scene_rect = self.scene_obj.sceneRect()
        half = sample_size / 2.0
        rect = QRectF(
            center_scene.x() - half,
            center_scene.y() - half,
            float(sample_size),
            float(sample_size),
        ).intersected(scene_rect)
        return (
            int(round(rect.x())),
            int(round(rect.y())),
            max(1, int(round(rect.width()))),
            max(1, int(round(rect.height()))),
        )

    def _update_export_preview(self) -> None:
        if not self.has_crop_state() or self._export_inset_preview_ratio <= 0.0:
            self.export_preview_item.setVisible(False)
            self.export_preview_item.setRect(QRectF())
            return

        state = self.crop_item.current_state()
        inset_x = state.width * self._export_inset_preview_ratio
        inset_y = state.height * self._export_inset_preview_ratio
        preview_width = max(1.0, state.width - inset_x * 2.0)
        preview_height = max(1.0, state.height - inset_y * 2.0)
        self.export_preview_item.setRect(
            QRectF(
                -preview_width / 2.0,
                -preview_height / 2.0,
                preview_width,
                preview_height,
            )
        )
        self.export_preview_item.setVisible(True)


class DemoMainWindow(QMainWindow):
    """Single-image demo window for validating the graphics interaction core."""

    ASPECT_MAP = {
        "3:2": 3 / 2,
        "1:1": 1.0,
        "6:7": 6 / 7,
        "自定义比例": "custom",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Film Crop Reviewer - Step2 Demo")
        self.resize(1400, 900)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        control = QWidget()
        control_layout = QVBoxLayout(control)
        form = QFormLayout()

        self.ratio_box = QComboBox()
        self.ratio_box.addItems(list(self.ASPECT_MAP.keys()))
        self.ratio_box.setCurrentText("3:2")
        form.addRow("画幅比例", self.ratio_box)

        self.custom_ratio_widget = QWidget()
        custom_ratio_layout = QHBoxLayout(self.custom_ratio_widget)
        custom_ratio_layout.setContentsMargins(0, 0, 0, 0)

        self.custom_ratio_w = QDoubleSpinBox()
        self.custom_ratio_w.setRange(0.01, 999.0)
        self.custom_ratio_w.setDecimals(2)
        self.custom_ratio_w.setValue(4.0)

        self.custom_ratio_h = QDoubleSpinBox()
        self.custom_ratio_h.setRange(0.01, 999.0)
        self.custom_ratio_h.setDecimals(2)
        self.custom_ratio_h.setValue(3.0)

        custom_ratio_layout.addWidget(self.custom_ratio_w)
        custom_ratio_layout.addWidget(QLabel(":"))
        custom_ratio_layout.addWidget(self.custom_ratio_h)
        form.addRow("自定义比例", self.custom_ratio_widget)

        self.progress_label = QLabel("当前进度: 1 / 1")
        self.export_btn = QPushButton("批量导出 (Export Approved)")

        control_layout.addLayout(form)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.export_btn)
        control_layout.addStretch(1)

        self.viewer = FilmGraphicsView()

        outer.addWidget(control, 0)
        outer.addWidget(self.viewer, 1)

        self.ratio_box.currentTextChanged.connect(self._on_ratio_changed)
        self.custom_ratio_w.valueChanged.connect(self._on_custom_ratio_changed)
        self.custom_ratio_h.valueChanged.connect(self._on_custom_ratio_changed)
        self.viewer.approveRequested.connect(self._on_approve)

        self._load_demo_image()
        self._on_ratio_changed(self.ratio_box.currentText())

    def _load_demo_image(self) -> None:
        w, h = 2200, 1400
        img = QImage(w, h, QImage.Format_RGB888)
        img.fill(QColor("#1f1f1f"))

        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#d9d9d9"))
        p.drawRoundedRect(220, 130, 1760, 1140, 20, 20)
        p.setBrush(QColor("#f4f4f4"))
        p.drawRect(320, 230, 1560, 940)
        p.end()

        self.viewer.set_proxy_image(img)

    def _on_ratio_changed(self, text: str) -> None:
        is_custom = self.ASPECT_MAP.get(text) == "custom"
        self.custom_ratio_widget.setVisible(is_custom)
        self.custom_ratio_w.setEnabled(is_custom)
        self.custom_ratio_h.setEnabled(is_custom)
        self.viewer.crop_item.set_aspect_ratio(self._current_aspect_ratio())

    def _on_custom_ratio_changed(self, _value: float) -> None:
        if self.ratio_box.currentText() != "自定义比例":
            return
        self.viewer.crop_item.set_aspect_ratio(self._current_aspect_ratio())

    def _current_aspect_ratio(self) -> Optional[float]:
        selected = self.ASPECT_MAP.get(self.ratio_box.currentText())
        if selected == "custom":
            return self.custom_ratio_w.value() / self.custom_ratio_h.value()
        return selected

    def _on_approve(self) -> None:
        state = self.viewer.crop_item.current_state()
        self.statusBar().showMessage(
            f"Approved: cx={state.cx:.1f}, cy={state.cy:.1f}, w={state.width:.1f}, h={state.height:.1f}, angle={state.angle_deg:.1f}",
            4000,
        )


def build_main_window() -> DemoMainWindow:
    return DemoMainWindow()


def run_demo() -> int:
    app = QApplication.instance() or QApplication([])
    w = build_main_window()
    w.show()
    return app.exec()
