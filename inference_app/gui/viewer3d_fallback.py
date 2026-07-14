from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class FallbackViewer3DDialog(QDialog):
    """Stable 3D fallback viewer using Matplotlib voxels.

    This is used only when VTK/PyVista fails on a given Linux graphics stack.
    """

    def __init__(
        self,
        pred_dhw: Optional[np.ndarray],
        gt_dhw: Optional[np.ndarray],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Tumor Viewer (Fallback)")
        self.resize(980, 760)

        self.pred_dhw = (pred_dhw > 0) if pred_dhw is not None else None
        self.gt_dhw = (gt_dhw > 0) if gt_dhw is not None else None
        self.last_screenshot_path: Optional[str] = None

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        form = QFormLayout()

        self.visibility_combo = QComboBox()
        self.visibility_combo.addItems(["Predicted only", "Ground truth only", "Both"])

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(60)

        self.downsample_combo = QComboBox()
        self.downsample_combo.addItems(["2", "3", "4", "5"])
        self.downsample_combo.setCurrentText("3")

        self.render_button = QPushButton("Render")
        self.reset_view_button = QPushButton("Reset View")
        self.screenshot_button = QPushButton("Capture Screenshot")

        form.addRow("Show", self.visibility_combo)
        form.addRow("Opacity", self.alpha_slider)
        form.addRow("Downsample", self.downsample_combo)

        controls.addLayout(form)
        controls.addWidget(self.render_button)
        controls.addWidget(self.reset_view_button)
        controls.addWidget(self.screenshot_button)
        controls.addStretch(1)

        root.addLayout(controls)

        self.figure = Figure(figsize=(7, 6), facecolor="#0f1218")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_facecolor("#0f1218")
        root.addWidget(self.canvas)

        self.info_label = QLabel("Fallback mode active: rotate with mouse, zoom with scroll.")
        root.addWidget(self.info_label)

        self.render_button.clicked.connect(self.render_scene)
        self.reset_view_button.clicked.connect(self._reset_view)
        self.screenshot_button.clicked.connect(self._capture_screenshot)

        self.render_scene()

    def _downsample(self, arr: np.ndarray, factor: int) -> np.ndarray:
        return arr[::factor, ::factor, ::factor]

    def render_scene(self) -> None:
        self.axes.clear()
        self.axes.set_facecolor("#0f1218")

        mode = self.visibility_combo.currentText()
        alpha = self.alpha_slider.value() / 100.0
        factor = int(self.downsample_combo.currentText())

        has_any = False
        if self.pred_dhw is not None and mode in {"Predicted only", "Both"}:
            pred = self._downsample(self.pred_dhw, factor)
            if np.any(pred):
                colors = np.empty(pred.shape, dtype=object)
                colors[:] = "#ff3b30"
                self.axes.voxels(pred, facecolors=colors, edgecolor="none", alpha=alpha)
                has_any = True

        if self.gt_dhw is not None and mode in {"Ground truth only", "Both"}:
            gt = self._downsample(self.gt_dhw, factor)
            if np.any(gt):
                colors = np.empty(gt.shape, dtype=object)
                colors[:] = "#2ecc71"
                self.axes.voxels(gt, facecolors=colors, edgecolor="none", alpha=alpha)
                has_any = True

        if not has_any:
            self.axes.text2D(0.2, 0.5, "No tumor voxels to render", transform=self.axes.transAxes, color="white")

        self.axes.set_xlabel("X", color="white")
        self.axes.set_ylabel("Y", color="white")
        self.axes.set_zlabel("Z", color="white")
        self.axes.tick_params(colors="white")

        self.canvas.draw_idle()

    def _reset_view(self) -> None:
        self.axes.view_init(elev=30, azim=-60)
        self.canvas.draw_idle()

    def _capture_screenshot(self) -> None:
        if self.last_screenshot_path:
            self.figure.savefig(self.last_screenshot_path, dpi=160, facecolor=self.figure.get_facecolor(), bbox_inches="tight")
            self.info_label.setText(f"Screenshot saved: {self.last_screenshot_path}")
        else:
            self.info_label.setText("Set screenshot path from main window before capture.")

    def capture_to_path(self, output_path: str) -> None:
        self.last_screenshot_path = output_path
        self.figure.savefig(output_path, dpi=160, facecolor=self.figure.get_facecolor(), bbox_inches="tight")
        self.info_label.setText(f"Screenshot saved: {output_path}")
