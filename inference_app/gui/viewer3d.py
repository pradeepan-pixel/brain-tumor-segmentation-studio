from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
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


class Viewer3DDialog(QDialog):
    def __init__(
        self,
        brain_dhw: np.ndarray,
        pred_dhw: Optional[np.ndarray],
        gt_dhw: Optional[np.ndarray],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Tumor Viewer")
        self.resize(1100, 760)

        self.brain_dhw = brain_dhw
        self.pred_dhw = pred_dhw
        self.gt_dhw = gt_dhw
        self.last_screenshot_path: Optional[str] = None

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Surface", "Volume", "Wireframe"])

        self.visibility_combo = QComboBox()
        self.visibility_combo.addItems(["Predicted only", "Ground truth only", "Both"])

        self.pred_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.pred_opacity_slider.setRange(5, 100)
        self.pred_opacity_slider.setValue(60)

        self.gt_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.gt_opacity_slider.setRange(5, 100)
        self.gt_opacity_slider.setValue(55)

        self.reset_button = QPushButton("Reset Camera")
        self.screenshot_button = QPushButton("Capture Screenshot")

        controls_form = QFormLayout()
        controls_form.addRow("Render mode", self.mode_combo)
        controls_form.addRow("Show", self.visibility_combo)
        controls_form.addRow("Prediction opacity", self.pred_opacity_slider)
        controls_form.addRow("Ground truth opacity", self.gt_opacity_slider)

        controls.addLayout(controls_form)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.screenshot_button)
        controls.addStretch(1)

        root.addLayout(controls)

        self.plotter = QtInteractor(self)
        root.addWidget(self.plotter)

        self.info_label = QLabel("Use mouse to rotate/zoom and inspect tumor geometry.")
        root.addWidget(self.info_label)

        self.mode_combo.currentTextChanged.connect(self.render_scene)
        self.visibility_combo.currentTextChanged.connect(self.render_scene)
        self.pred_opacity_slider.valueChanged.connect(self.render_scene)
        self.gt_opacity_slider.valueChanged.connect(self.render_scene)
        self.reset_button.clicked.connect(self._reset_camera)
        self.screenshot_button.clicked.connect(self._capture_screenshot)

        self.render_scene()

    def _to_image_data(self, arr: np.ndarray) -> pv.ImageData:
        grid = pv.ImageData(dimensions=arr.shape)
        grid["values"] = arr.flatten(order="F")
        return grid

    def render_scene(self) -> None:
        self.plotter.clear()
        self.plotter.set_background("#0f1218")

        brain = self.brain_dhw.astype(np.float32)
        p5 = np.percentile(brain, 50)
        p95 = np.percentile(brain, 99)
        if p95 <= p5:
            p95 = p5 + 1e-3
        brain_norm = np.clip((brain - p5) / (p95 - p5), 0.0, 1.0)

        mode = self.mode_combo.currentText()
        show_mode = self.visibility_combo.currentText()

        if mode == "Volume":
            self.plotter.add_volume(
                brain_norm,
                cmap="bone",
                opacity="sigmoid_6",
                shade=True,
                blending="composite",
            )
        else:
            brain_mask = (brain_norm > 0.25).astype(np.uint8)
            if np.any(brain_mask):
                brain_grid = self._to_image_data(brain_mask)
                brain_surface = brain_grid.contour(isosurfaces=[0.5], scalars="values")
                style = "wireframe" if mode == "Wireframe" else "surface"
                self.plotter.add_mesh(
                    brain_surface,
                    color="#3a4a63",
                    opacity=0.12,
                    style=style,
                    lighting=True,
                )

        pred_opacity = self.pred_opacity_slider.value() / 100.0
        gt_opacity = self.gt_opacity_slider.value() / 100.0

        if self.pred_dhw is not None and show_mode in {"Predicted only", "Both"} and np.any(self.pred_dhw):
            pred_grid = self._to_image_data((self.pred_dhw > 0).astype(np.uint8))
            pred_surface = pred_grid.contour(isosurfaces=[0.5], scalars="values")
            style = "wireframe" if mode == "Wireframe" else "surface"
            self.plotter.add_mesh(pred_surface, color="#ff3b30", opacity=pred_opacity, style=style)

        if self.gt_dhw is not None and show_mode in {"Ground truth only", "Both"} and np.any(self.gt_dhw):
            gt_grid = self._to_image_data((self.gt_dhw > 0).astype(np.uint8))
            gt_surface = gt_grid.contour(isosurfaces=[0.5], scalars="values")
            style = "wireframe" if mode == "Wireframe" else "surface"
            self.plotter.add_mesh(gt_surface, color="#2ecc71", opacity=gt_opacity, style=style)

        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()
        self.plotter.reset_camera()
        self.plotter.render()

    def _reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def _capture_screenshot(self) -> None:
        if self.last_screenshot_path:
            self.plotter.screenshot(self.last_screenshot_path)
            self.info_label.setText(f"Screenshot updated: {self.last_screenshot_path}")
        else:
            self.info_label.setText("Set screenshot path from main window before capture.")

    def capture_to_path(self, output_path: str) -> None:
        self.last_screenshot_path = output_path
        self.plotter.screenshot(output_path)
        self.info_label.setText(f"Screenshot saved: {output_path}")
