"""PySide6 graphics core for interactive crop review (step-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
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
    QFormLayout,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
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
    """Interactive crop item: move + resize (ratio lock aware) + rotate."""

    changed = Signal()

    def __init__(self, width: float = 400, height: float = 260) -> None:
        super().__init__()
        self._width = max(10.0, width)
        self._height = max(10.0, height)
        self._aspect_ratio: Optional[float] = self._width / self._height
        self._template_locked = False

        self._mode = "none"  # none|move|resize
        self._resize_edges = {"l": False, "r": False, "t": False, "b": False}
        self._drag_start = QPointF()
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

    def set_template_locked(self, locked: bool) -> None:
        self._template_locked = locked

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

    def polygon_in_scene(self):
        rect = QRectF(-self._width / 2, -self._height / 2, self._width, self._height)
        return self.mapToScene(rect)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._drag_start = event.pos()
        self._start_w = self._width
        self._start_h = self._height
        self._resize_edges = self._detect_edges(event.pos())

        if not self._template_locked and any(self._resize_edges.values()):
            self._mode = "resize"
        else:
            self._mode = "move"

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._mode == "move":
            delta_scene = event.scenePos() - event.lastScenePos()
            self.setPos(self.pos() + delta_scene)
            self.changed.emit()
            return

        if self._mode == "resize" and not self._template_locked:
            delta = event.pos() - self._drag_start
            new_w = self._start_w
            new_h = self._start_h

            if self._resize_edges["l"]:
                new_w = self._start_w - delta.x() * 2
            elif self._resize_edges["r"]:
                new_w = self._start_w + delta.x() * 2

            if self._resize_edges["t"]:
                new_h = self._start_h - delta.y() * 2
            elif self._resize_edges["b"]:
                new_h = self._start_h + delta.y() * 2

            new_w = max(20.0, new_w)
            new_h = max(20.0, new_h)

            if self._aspect_ratio and self._aspect_ratio > 0:
                horizontal_change = self._resize_edges["l"] or self._resize_edges["r"]
                vertical_change = self._resize_edges["t"] or self._resize_edges["b"]
                if horizontal_change and not vertical_change:
                    new_h = max(20.0, new_w / self._aspect_ratio)
                elif vertical_change and not horizontal_change:
                    new_w = max(20.0, new_h * self._aspect_ratio)
                else:
                    # corner resize: choose larger relative movement
                    dw = abs(new_w - self._start_w) / max(1.0, self._start_w)
                    dh = abs(new_h - self._start_h) / max(1.0, self._start_h)
                    if dw >= dh:
                        new_h = max(20.0, new_w / self._aspect_ratio)
                    else:
                        new_w = max(20.0, new_h * self._aspect_ratio)

            self.prepareGeometryChange()
            self._width = new_w
            self._height = new_h
            self.update()
            self.changed.emit()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._mode = "none"
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        if self._template_locked:
            self.setCursor(Qt.OpenHandCursor)
            return

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


class MaskOverlayItem(QGraphicsObject):
    """Darken overlay with a transparent rotated-rect hole."""

    def __init__(self, scene: QGraphicsScene, crop_item: CropRectItem) -> None:
        super().__init__()
        self._scene = scene
        self._crop_item = crop_item
        self._color = QColor(0, 0, 0, 128)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(10_000)

        self._scene.sceneRectChanged.connect(self._on_scene_rect_changed)
        self._crop_item.changed.connect(self.update)

    def _on_scene_rect_changed(self, _rect: QRectF) -> None:
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._scene.sceneRect()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        scene_rect = self._scene.sceneRect()
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
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

        self.mask_item = MaskOverlayItem(self.scene_obj, self.crop_item)
        self.scene_obj.addItem(self.mask_item)

        self._zoom = 1.0

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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Q:
            self.crop_item.set_angle(self.crop_item.rotation() - 0.5)
            event.accept()
            return
        if key == Qt.Key_E:
            self.crop_item.set_angle(self.crop_item.rotation() + 0.5)
            event.accept()
            return
        if key == Qt.Key_Left:
            self.crop_item.nudge(-1, 0)
            event.accept()
            return
        if key == Qt.Key_Right:
            self.crop_item.nudge(1, 0)
            event.accept()
            return
        if key == Qt.Key_Up:
            self.crop_item.nudge(0, -1)
            event.accept()
            return
        if key == Qt.Key_Down:
            self.crop_item.nudge(0, 1)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.approveRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class DemoMainWindow(QMainWindow):
    """Single-image demo window for validating the graphics interaction core."""

    ASPECT_MAP = {
        "3:2": 3 / 2,
        "1:1": 1.0,
        "6:7": 6 / 7,
        "自由比例": None,
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

        self.template_btn = QPushButton("设为全局模板")
        self.progress_label = QLabel("当前进度: 1 / 1")
        self.export_btn = QPushButton("批量导出 (Export Approved)")

        control_layout.addLayout(form)
        control_layout.addWidget(self.template_btn)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.export_btn)
        control_layout.addStretch(1)

        self.viewer = FilmGraphicsView()

        outer.addWidget(control, 0)
        outer.addWidget(self.viewer, 1)

        self.ratio_box.currentTextChanged.connect(self._on_ratio_changed)
        self.template_btn.clicked.connect(self._on_set_template)
        self.viewer.approveRequested.connect(self._on_approve)

        self._template_locked = False
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
        if self._template_locked:
            return
        self.viewer.crop_item.set_aspect_ratio(self.ASPECT_MAP.get(text))

    def _on_set_template(self) -> None:
        self._template_locked = True
        self.viewer.crop_item.set_template_locked(True)
        self.template_btn.setEnabled(False)
        self.ratio_box.setEnabled(False)
        self.statusBar().showMessage("已锁定模板：Width/Height 固定，只允许平移和旋转", 3000)

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
