from __future__ import annotations

from typing import Dict, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

matplotlib.use("QtAgg")


class MRIViewer2D(QWidget):
    slice_changed = pyqtSignal(int)
    cursor_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.modalities: Dict[str, np.ndarray] = {}
        self.prediction_dhw: Optional[np.ndarray] = None
        self.ground_truth_dhw: Optional[np.ndarray] = None

        self.current_modality = "FLAIR"
        self.current_slice = 0
        self.overlay_opacity = 0.5
        self.show_prediction = True
        self.show_ground_truth = True
        self.brightness = 0
        self.contrast = 0

        self.figure = Figure(figsize=(6, 6), facecolor="#161a22")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.axes = self.figure.add_subplot(111)

        self.axes.set_facecolor("#0f1218")
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def set_data(self, modalities_dhw: Dict[str, np.ndarray], ground_truth_dhw: Optional[np.ndarray] = None) -> None:
        self.modalities = modalities_dhw
        self.ground_truth_dhw = ground_truth_dhw
        if self.current_modality not in self.modalities:
            self.current_modality = next(iter(self.modalities.keys()))
        self.current_slice = self.total_slices // 2
        self.draw()

    def set_prediction(self, prediction_dhw: Optional[np.ndarray]) -> None:
        self.prediction_dhw = prediction_dhw
        self.draw()

    @property
    def total_slices(self) -> int:
        volume = self.modalities.get(self.current_modality)
        return int(volume.shape[0]) if volume is not None else 0

    def set_slice(self, idx: int) -> None:
        if self.total_slices == 0:
            return
        self.current_slice = max(0, min(idx, self.total_slices - 1))
        self.slice_changed.emit(self.current_slice)
        self.draw()

    def set_modality(self, modality: str) -> None:
        if modality not in self.modalities:
            return
        self.current_modality = modality
        self.current_slice = min(self.current_slice, max(0, self.total_slices - 1))
        self.draw()

    def set_overlay_options(self, show_pred: bool, show_gt: bool, opacity: float) -> None:
        self.show_prediction = show_pred
        self.show_ground_truth = show_gt
        self.overlay_opacity = float(np.clip(opacity, 0.0, 1.0))
        self.draw()

    def set_brightness_contrast(self, brightness: int, contrast: int) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.draw()

    def _adjust_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        min_v = float(img.min())
        max_v = float(img.max())
        if max_v - min_v < 1e-8:
            img = np.zeros_like(img)
        else:
            img = (img - min_v) / (max_v - min_v)

        contrast_factor = 1.0 + (self.contrast / 100.0)
        brightness_shift = self.brightness / 100.0

        img = (img - 0.5) * contrast_factor + 0.5 + brightness_shift
        return np.clip(img, 0.0, 1.0)

    def draw(self) -> None:
        self.axes.clear()
        self.axes.set_facecolor("#0f1218")

        if self.total_slices == 0:
            self.axes.text(0.5, 0.5, "Load a patient to start", color="white", ha="center", va="center")
            self.axes.axis("off")
            self.canvas.draw_idle()
            return

        base_slice = self.modalities[self.current_modality][self.current_slice]
        base_slice = self._adjust_image(base_slice)

        self.axes.imshow(base_slice, cmap="gray", origin="lower", interpolation="nearest")

        if self.show_prediction and self.prediction_dhw is not None:
            pred_slice = self.prediction_dhw[self.current_slice]
            pred_mask = np.ma.masked_where(pred_slice <= 0, pred_slice)
            self.axes.imshow(pred_mask, cmap="Reds", alpha=self.overlay_opacity, origin="lower", interpolation="nearest")

        if self.show_ground_truth and self.ground_truth_dhw is not None:
            gt_slice = self.ground_truth_dhw[self.current_slice]
            gt_mask = np.ma.masked_where(gt_slice <= 0, gt_slice)
            self.axes.imshow(gt_mask, cmap="Greens", alpha=self.overlay_opacity, origin="lower", interpolation="nearest")

        self.axes.set_title(f"{self.current_modality} | Slice {self.current_slice + 1}/{self.total_slices}", color="white")
        self.axes.axis("off")
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if self.total_slices == 0:
            return
        if event.button == "up":
            self.set_slice(self.current_slice + 1)
        else:
            self.set_slice(self.current_slice - 1)

    def _on_mouse_move(self, event) -> None:
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        active_volume = self.modalities.get(self.current_modality)
        if active_volume is None or active_volume.ndim != 3:
            return

        x = int(np.clip(round(event.xdata), 0, active_volume.shape[2] - 1))
        y = int(np.clip(round(event.ydata), 0, active_volume.shape[1] - 1))
        z = int(np.clip(self.current_slice, 0, active_volume.shape[0] - 1))
        val = float(active_volume[z, y, x])
        self.cursor_changed.emit(f"x={x}, y={y}, z={z}, value={val:.3f}")

    def export_png(self, output_path: str) -> None:
        self.figure.savefig(output_path, dpi=160, facecolor=self.figure.get_facecolor(), bbox_inches="tight")
