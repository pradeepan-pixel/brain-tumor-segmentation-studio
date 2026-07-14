from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .model_loader import LoadedModel, load_model
from .prediction_worker import PredictionThread
from src.report_visuals import (
    build_feature_patch,
    capture_encoder_feature_maps,
    load_raw_modalities,
    make_classification_figure,
    make_feature_figure,
    make_input_3d_figure,
    make_methodology_figure,
    make_preprocessing_figure,
    make_segmentation_figure,
    select_focus_slice,
    zscore_modalities,
)
from .results_view import ResultsDialog
from .utils import PatientData, compute_comparison_metrics, compute_tumor_stats, load_patient_data
from .viewer2d import MRIViewer2D

# Absolute inference_app folder so exports always land in
# inference_app/export/<patient_id>/ regardless of the launch directory.
APP_DIR = Path(__file__).resolve().parents[1]

# Filenames used for the per-patient organized export bundle.
SECTION_FILENAMES = {
    "Input 3D image": "1_input_3d.png",
    "Preprocessing output image": "2_preprocessing.png",
    "Feature extracted output image": "3_features.png",
    "Segmentation output": "4_segmentation.png",
    "Classification layer output": "5_classification.png",
    "Methodology": "6_methodology.png",
}


# Paper-style explanations shown under each on-screen result heading. These
# describe outputs generated from the real patient prediction, not placeholders.
RESULT_EXPLANATIONS = {
    "input_3d": (
        "The raw FLAIR volume of the loaded BraTS patient is rendered as a 3D point cloud. "
        "The blue cloud is the actual brain tissue, red points are the voxels the model predicted "
        "as tumor, and green points (when a segmentation file exists) are the ground-truth tumor. "
        "This is built directly from this patient's scan."
    ),
    "preprocessing": (
        "Each of the four MRI modalities (FLAIR, T1, T1CE, T2) is z-score normalized independently. "
        "The top row is the raw intensity slice; the bottom row is the normalized slice that is actually "
        "fed to the network. Normalization removes scanner-specific intensity scale so the model sees a "
        "standardized distribution across patients."
    ),
    "features": (
        "Feature maps captured with forward hooks from the first three encoder blocks (enc1-enc3) of the "
        "3D U-Net while it processes this patient's tumor-centered patch. Early layers respond to edges and "
        "intensity; deeper layers respond to larger tumor structures. These are the model's real learned "
        "activations for this patient."
    ),
    "segmentation": (
        "The predicted whole-tumor mask overlaid on the patient's FLAIR slice (red), and the same slice "
        "compared against ground truth (green) when available. The displayed slice is automatically chosen "
        "as the one with the largest predicted tumor cross-section."
    ),
    "classification": (
        "The model has no image-level classifier; instead its final 1x1x1 Conv3d + sigmoid is the classification "
        "layer, which classifies every voxel independently as tumor or non-tumor. Left: the input patch. Middle: the "
        "continuous per-voxel probability map (jet color, 0 to 1) before thresholding. Right: the binary mask at a "
        "0.5 threshold. This shows the network outputs a continuous tumor likelihood, not a mask directly."
    ),
    "methodology": (
        "End-to-end methodology of how the outputs above are produced, from loading the patient's four MRI "
        "modalities through per-modality z-score normalization, patch-based 3D U-Net inference reconstructed with a "
        "sliding window, to the voxel-wise sigmoid classification layer that yields the final tumor probability."
    ),
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Brain Tumor Segmentation Studio")
        self.resize(1600, 940)

        self.patient_data: Optional[PatientData] = None
        self.loaded_model: Optional[LoadedModel] = None
        self.prediction_dhw: Optional[np.ndarray] = None
        self.prediction_prob_dhw: Optional[np.ndarray] = None
        self.prediction_path: Optional[str] = None
        self.inference_time_sec: Optional[float] = None
        self.last_confidence: Optional[float] = None
        self.viewer3d_dialog: Optional[Any] = None
        self.results_dialog: Optional[Any] = None
        self.prediction_thread: Optional[PredictionThread] = None

        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self._build_left_panel())
        top_splitter.addWidget(self._build_center_panel())
        top_splitter.addWidget(self._build_right_panel())
        top_splitter.setStretchFactor(0, 2)
        top_splitter.setStretchFactor(1, 7)
        top_splitter.setStretchFactor(2, 3)

        root_layout.addWidget(top_splitter)
        root_layout.addWidget(self._build_bottom_panel())

        self.setCentralWidget(root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)

        title = QLabel("Model Information")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.model_checkpoint_label = QLabel("Checkpoint: Not loaded")
        self.model_epoch_label = QLabel("Epoch: -")
        self.model_base_label = QLabel("Base channels: -")
        self.model_device_label = QLabel("Device: -")
        self.model_status_label = QLabel("Status: Not loaded")

        form = QFormLayout()
        form.addRow("Checkpoint", self.model_checkpoint_label)
        form.addRow("Epoch", self.model_epoch_label)
        form.addRow("Base", self.model_base_label)
        form.addRow("Device", self.model_device_label)
        form.addRow("State", self.model_status_label)
        layout.addLayout(form)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.on_load_model)
        layout.addWidget(self.load_model_button)

        layout.addSpacing(16)

        patient_title = QLabel("Patient Information")
        patient_title.setObjectName("sectionTitle")
        layout.addWidget(patient_title)

        self.patient_id_label = QLabel("Patient ID: -")
        self.patient_dir_label = QLabel("Folder: -")
        self.inference_time_label = QLabel("Inference time: -")
        self.dice_label = QLabel("Dice: -")

        layout.addWidget(self.patient_id_label)
        layout.addWidget(self.patient_dir_label)
        layout.addWidget(self.inference_time_label)
        layout.addWidget(self.dice_label)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)

        self.viewer2d = MRIViewer2D(self)
        self.viewer2d.slice_changed.connect(self._sync_slice_ui)
        self.viewer2d.cursor_changed.connect(self._on_cursor_changed)

        layout.addWidget(self.viewer2d)

        controls_row = QHBoxLayout()

        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["FLAIR", "T1", "T1CE", "T2", "Prediction", "Ground Truth"])
        self.modality_combo.currentTextChanged.connect(self.on_modality_changed)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        self.slice_label = QLabel("Slice: 0/0")

        controls_row.addWidget(QLabel("Modality"))
        controls_row.addWidget(self.modality_combo)
        controls_row.addWidget(self.slice_slider, 1)
        controls_row.addWidget(self.slice_label)
        layout.addLayout(controls_row)

        overlay_row = QHBoxLayout()
        self.show_pred_checkbox = QCheckBox("Show prediction")
        self.show_pred_checkbox.setChecked(True)
        self.show_gt_checkbox = QCheckBox("Show ground truth")
        self.show_gt_checkbox.setChecked(True)

        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity_slider.setRange(5, 100)
        self.overlay_opacity_slider.setValue(55)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-80, 120)
        self.contrast_slider.setValue(0)

        overlay_row.addWidget(self.show_pred_checkbox)
        overlay_row.addWidget(self.show_gt_checkbox)
        overlay_row.addWidget(QLabel("Opacity"))
        overlay_row.addWidget(self.overlay_opacity_slider)
        overlay_row.addWidget(QLabel("Brightness"))
        overlay_row.addWidget(self.brightness_slider)
        overlay_row.addWidget(QLabel("Contrast"))
        overlay_row.addWidget(self.contrast_slider)
        layout.addLayout(overlay_row)

        self.show_pred_checkbox.toggled.connect(self._update_overlays)
        self.show_gt_checkbox.toggled.connect(self._update_overlays)
        self.overlay_opacity_slider.valueChanged.connect(self._update_overlays)
        self.brightness_slider.valueChanged.connect(self._update_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self._update_brightness_contrast)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)

        status_title = QLabel("Prediction Status")
        status_title.setObjectName("sectionTitle")
        layout.addWidget(status_title)

        self.pred_status_label = QLabel("Decision: -")
        self.pred_target_label = QLabel("Target class: Whole Tumor (binary WT mask)")
        self.pred_summary_label = QLabel("Summary: -")
        self.pred_quality_label = QLabel("Quality note: -")
        self.pred_summary_label.setWordWrap(True)
        self.pred_quality_label.setWordWrap(True)

        layout.addWidget(self.pred_status_label)
        layout.addWidget(self.pred_target_label)
        layout.addWidget(self.pred_summary_label)
        layout.addWidget(self.pred_quality_label)
        layout.addSpacing(8)

        title = QLabel("Prediction Statistics")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.stat_volume = QLabel("Tumor volume (ml): -")
        self.stat_voxels = QLabel("Voxel count: -")
        self.stat_bbox = QLabel("Bounding box: -")
        self.stat_centroid = QLabel("Centroid: -")
        self.stat_diameter = QLabel("Max diameter (mm): -")
        self.stat_confidence = QLabel("Confidence: -")
        self.stat_slice = QLabel("Slice: -")
        self.stat_total_slices = QLabel("Total slices: -")
        self.stat_jaccard = QLabel("Jaccard: -")
        self.stat_sensitivity = QLabel("Sensitivity: -")
        self.stat_precision = QLabel("Precision: -")
        self.metric_hint = QLabel("Dice/Jaccard/Sensitivity/Precision appear when ground truth is available.")
        self.metric_hint.setWordWrap(True)

        for lbl in [
            self.stat_volume,
            self.stat_voxels,
            self.stat_bbox,
            self.stat_centroid,
            self.stat_diameter,
            self.stat_confidence,
            self.stat_slice,
            self.stat_total_slices,
            self.stat_jaccard,
            self.stat_sensitivity,
            self.stat_precision,
            self.metric_hint,
        ]:
            layout.addWidget(lbl)

        layout.addStretch(1)
        return panel

    def _build_bottom_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)

        buttons = QHBoxLayout()
        self.load_patient_button = QPushButton("Load Patient")
        self.run_prediction_button = QPushButton("Run Prediction")
        self.save_prediction_button = QPushButton("Save Prediction")
        self.open_prediction_button = QPushButton("Open Prediction")
        self.generate_3d_button = QPushButton("Generate 3D")
        self.export_png_button = QPushButton("Export PNG")
        self.view_results_button = QPushButton("View Results")
        self.export_report_button = QPushButton("Export Report")

        buttons.addWidget(self.load_patient_button)
        buttons.addWidget(self.run_prediction_button)
        buttons.addWidget(self.save_prediction_button)
        buttons.addWidget(self.open_prediction_button)
        buttons.addWidget(self.generate_3d_button)
        buttons.addWidget(self.export_png_button)
        buttons.addWidget(self.view_results_button)
        buttons.addWidget(self.export_report_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addLayout(buttons)
        layout.addWidget(self.progress_bar)

        self.load_patient_button.clicked.connect(self.on_load_patient)
        self.run_prediction_button.clicked.connect(self.on_run_prediction)
        self.save_prediction_button.clicked.connect(self.on_save_prediction)
        self.open_prediction_button.clicked.connect(self.on_open_prediction)
        self.generate_3d_button.clicked.connect(self.on_generate_3d)
        self.export_png_button.clicked.connect(self.on_export_png)
        self.view_results_button.clicked.connect(self.on_view_results)
        self.export_report_button.clicked.connect(self.on_export_report)

        return panel

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Noto Sans', sans-serif;
                font-size: 13px;
            }
            QFrame#sidePanel {
                background: #121922;
                border: 1px solid #253344;
                border-radius: 10px;
                padding: 8px;
            }
            QLabel#sectionTitle {
                font-size: 16px;
                font-weight: 700;
                color: #7dc8ff;
                margin-bottom: 6px;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2b3a4f;
                border-color: #5ba4e5;
            }
            QPushButton:disabled {
                background-color: #1a202c;
                color: #7b8ba3;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3c4f66;
                height: 6px;
                background: #1c2838;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #68b6ff;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
                background: #111827;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background-color: #ef4444;
            }
            """
        )

    def on_load_patient(self) -> None:
        patient_dir = QFileDialog.getExistingDirectory(self, "Select BraTS Patient Folder")
        if not patient_dir:
            return

        try:
            self.patient_data = load_patient_data(patient_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load patient", str(exc))
            return

        self.prediction_dhw = None
        self.prediction_prob_dhw = None
        self.prediction_path = None
        self.inference_time_sec = None
        self.last_confidence = None

        modalities = dict(self.patient_data.modalities_dhw)
        if self.patient_data.ground_truth_dhw is not None:
            modalities["Ground Truth"] = self.patient_data.ground_truth_dhw.astype(np.float32)

        self.viewer2d.set_data(modalities, self.patient_data.ground_truth_dhw)
        total_slices = self.viewer2d.total_slices
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, total_slices - 1))
        self.slice_slider.setValue(min(total_slices // 2, max(0, total_slices - 1)))
        self.slice_slider.blockSignals(False)
        self._sync_slice_ui(self.viewer2d.current_slice)

        self.patient_id_label.setText(f"Patient ID: {self.patient_data.patient_id}")
        self.patient_dir_label.setText(f"Folder: {self.patient_data.patient_dir}")
        self.inference_time_label.setText("Inference time: -")
        self.dice_label.setText("Dice: -")
        self._update_stats()
        self.status.showMessage("Patient loaded", 3500)

    def on_load_model(self) -> None:
        checkpoint, _ = QFileDialog.getOpenFileName(self, "Select model checkpoint", filter="PyTorch checkpoint (*.pth *.pt)")
        if not checkpoint:
            return

        try:
            self.loaded_model = load_model(checkpoint)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load model", str(exc))
            return

        self.model_checkpoint_label.setText(os.path.basename(self.loaded_model.checkpoint_path))
        self.model_epoch_label.setText(str(self.loaded_model.epoch) if self.loaded_model.epoch is not None else "?")
        self.model_base_label.setText(str(self.loaded_model.base_channels))
        self.model_device_label.setText(str(self.loaded_model.device))
        self.model_status_label.setText("Loaded Successfully")
        self.status.showMessage("Model loaded", 3000)

    def on_run_prediction(self) -> None:
        if self.patient_data is None:
            QMessageBox.warning(self, "Missing patient", "Load a patient folder first.")
            return
        if self.loaded_model is None:
            QMessageBox.warning(self, "Missing model", "Load model checkpoint first.")
            return
        if self.prediction_thread is not None and self.prediction_thread.isRunning():
            QMessageBox.information(self, "Inference in progress", "Please wait for the current inference to finish.")
            return

        output_name = f"pred_{self.patient_data.patient_id}.nii.gz"
        output_path = str(Path("outputs") / output_name)

        self.prediction_thread = PredictionThread(
            model=self.loaded_model.model,
            device=self.loaded_model.device,
            patient_dir=self.patient_data.patient_dir,
            output_path=output_path,
        )

        self.prediction_thread.progress_changed.connect(self._on_prediction_progress)
        self.prediction_thread.prediction_ready.connect(self._on_prediction_ready)
        self.prediction_thread.error.connect(self._on_prediction_error)

        self.progress_bar.setValue(0)
        self.status.showMessage("Running sliding-window inference...")
        self.prediction_thread.start()

    def _on_prediction_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status.showMessage(message)

    def _on_prediction_ready(self, prediction_dhw: np.ndarray, prediction_prob_dhw: np.ndarray, output_path: str, elapsed_sec: float, confidence: float) -> None:
        self.prediction_dhw = prediction_dhw.astype(np.uint8)
        self.prediction_prob_dhw = prediction_prob_dhw.astype(np.float32)
        self.prediction_path = output_path
        self.inference_time_sec = elapsed_sec
        self.last_confidence = confidence

        all_modalities = dict(self.patient_data.modalities_dhw)
        all_modalities["Prediction"] = self.prediction_dhw.astype(np.float32)
        if self.patient_data.ground_truth_dhw is not None:
            all_modalities["Ground Truth"] = self.patient_data.ground_truth_dhw.astype(np.float32)

        self.viewer2d.set_data(all_modalities, self.patient_data.ground_truth_dhw)
        self.viewer2d.set_prediction(self.prediction_dhw)

        self.inference_time_label.setText(f"Inference time: {elapsed_sec:.2f} s")

        metrics = compute_comparison_metrics(self.prediction_dhw, self.patient_data.ground_truth_dhw)
        if metrics["dice"] is not None:
            self.dice_label.setText(f"Dice: {metrics['dice']:.4f}")
        else:
            self.dice_label.setText("Dice: N/A (no ground truth)")

        self._update_stats()
        self.progress_bar.setValue(100)

        pred_voxels = int(self.prediction_dhw.sum())
        if pred_voxels > 0:
            stats = compute_tumor_stats(self.prediction_dhw, self.patient_data.spacing_dhw)
            volume_ml = stats["tumor_volume_ml"]
            status_msg = f"Prediction complete: Tumor detected ({volume_ml:.2f} ml). Saved to {output_path}"
        else:
            status_msg = f"Prediction complete: No tumor detected. Saved to {output_path}"
        self.status.showMessage(status_msg, 5000)

    def _on_prediction_error(self, message: str) -> None:
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Prediction failed", message)
        self.status.showMessage("Prediction failed", 3000)

    def on_save_prediction(self) -> None:
        if self.prediction_dhw is None or self.patient_data is None:
            QMessageBox.warning(self, "No prediction", "Run prediction first.")
            return

        default_mask = str(self._patient_export_dir() / f"pred_{self.patient_data.patient_id}.nii.gz")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save prediction as", default_mask, filter="NIfTI (*.nii.gz *.nii)"
        )
        if not file_path:
            return

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        pred_hwd = np.transpose(self.prediction_dhw, (1, 2, 0)).astype(np.uint8)
        pred_nii = nib.Nifti1Image(pred_hwd, affine=self.patient_data.ref_nii.affine, header=self.patient_data.ref_nii.header)
        nib.save(pred_nii, file_path)
        self.prediction_path = file_path
        self.status.showMessage(f"Saved prediction: {file_path}", 3500)

    def on_open_prediction(self) -> None:
        if self.patient_data is None:
            QMessageBox.warning(self, "Missing patient", "Load patient first before opening prediction.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Open prediction mask", filter="NIfTI (*.nii.gz *.nii)")
        if not file_path:
            return

        pred_hwd = nib.load(file_path).get_fdata(dtype=np.float32)
        pred_dhw = np.transpose((pred_hwd > 0).astype(np.uint8), (2, 0, 1))

        self.prediction_dhw = pred_dhw
        self.prediction_prob_dhw = None
        self.prediction_path = file_path
        self.viewer2d.set_prediction(pred_dhw)

        modalities = dict(self.patient_data.modalities_dhw)
        modalities["Prediction"] = pred_dhw.astype(np.float32)
        if self.patient_data.ground_truth_dhw is not None:
            modalities["Ground Truth"] = self.patient_data.ground_truth_dhw.astype(np.float32)
        self.viewer2d.set_data(modalities, self.patient_data.ground_truth_dhw)

        self._update_stats()
        self.status.showMessage(f"Loaded prediction: {file_path}", 3500)

    def on_generate_3d(self) -> None:
        if self.patient_data is None:
            QMessageBox.warning(self, "Missing patient", "Load patient first.")
            return

        backend = os.environ.get("BTS_3D_BACKEND")
        if backend is None:
            backend = "fallback" if sys.platform.startswith("linux") else "vtk"
        backend = backend.strip().lower()

        if backend == "fallback":
            from .viewer3d_fallback import FallbackViewer3DDialog

            self.viewer3d_dialog = FallbackViewer3DDialog(
                pred_dhw=self.prediction_dhw,
                gt_dhw=self.patient_data.ground_truth_dhw,
                parent=self,
            )
            self.viewer3d_dialog.show()
            self.status.showMessage("Opened fallback 3D viewer", 3000)
            return

        try:
            from .viewer3d import Viewer3DDialog
        except Exception as exc:
            QMessageBox.critical(
                self,
                "3D viewer unavailable",
                "Failed to initialize VTK/PyVista 3D viewer.\n"
                "On Linux, try starting with: QT_QPA_PLATFORM=xcb python main_gui.py\n\n"
                f"Details: {exc}",
            )
            return

        brain = self.patient_data.modalities_dhw["FLAIR"]
        try:
            self.viewer3d_dialog = Viewer3DDialog(
                brain_dhw=brain,
                pred_dhw=self.prediction_dhw,
                gt_dhw=self.patient_data.ground_truth_dhw,
                parent=self,
            )
            self.viewer3d_dialog.show()
        except Exception as exc:
            try:
                from .viewer3d_fallback import FallbackViewer3DDialog

                self.viewer3d_dialog = FallbackViewer3DDialog(
                    pred_dhw=self.prediction_dhw,
                    gt_dhw=self.patient_data.ground_truth_dhw,
                    parent=self,
                )
                self.viewer3d_dialog.show()
                QMessageBox.warning(
                    self,
                    "VTK unavailable",
                    "Opened fallback 3D viewer (Matplotlib) because VTK crashed on this system.\n\n"
                    f"VTK error: {exc}",
                )
            except Exception as fb_exc:
                QMessageBox.critical(
                    self,
                    "3D viewer failed",
                    "Both VTK and fallback 3D viewers failed.\n\n"
                    f"VTK error: {exc}\n"
                    f"Fallback error: {fb_exc}",
                )

    def on_export_png(self) -> None:
        default_png = ""
        if self.patient_data is not None:
            default_png = str(self._patient_export_dir() / f"slice_{self.patient_data.patient_id}.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export current slice as PNG", default_png, filter="PNG (*.png)"
        )
        if not file_path:
            return
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self.viewer2d.export_png(file_path)
        self.status.showMessage(f"PNG exported: {file_path}", 3000)

    def _report_prerequisites_ok(self, context: str) -> bool:
        if self.patient_data is None:
            QMessageBox.warning(self, "Missing patient", "Load patient first.")
            return False
        if self.prediction_dhw is None:
            QMessageBox.warning(
                self,
                "Missing prediction",
                f"Run prediction first so {context} can include real predicted outputs.",
            )
            return False
        if self.loaded_model is None:
            QMessageBox.warning(self, "Missing model", "Load model checkpoint first.")
            return False
        return True

    def _prepare_report_assets(self) -> dict:
        """Compute every real-patient array needed for the on-screen results and
        the PDF report: raw + normalized modalities, tumor-focused patch, encoder
        activations, and the sigmoid probability map. Runs one forward pass."""
        raw_modalities = load_raw_modalities(self.patient_data.patient_dir)
        normalized_modalities = zscore_modalities(raw_modalities)
        focus_slice = select_focus_slice(
            self.prediction_dhw, self.patient_data.ground_truth_dhw, self.viewer2d.total_slices
        )
        feature_patch, _ = build_feature_patch(
            normalized_modalities,
            self.prediction_dhw,
            self.patient_data.ground_truth_dhw,
        )
        activations, _, patch_probabilities = capture_encoder_feature_maps(
            self.loaded_model.model,
            feature_patch,
            self.loaded_model.device,
        )
        return {
            "raw_modalities": raw_modalities,
            "normalized_modalities": normalized_modalities,
            "focus_slice": focus_slice,
            "feature_patch": feature_patch,
            "activations": activations,
            "patch_probabilities": patch_probabilities,
        }

    def _build_result_sections(self, assets: dict) -> list:
        """Build fresh (heading, explanation, Figure) tuples for the five outputs
        plus methodology. Fresh figures each call so the caller owns them."""
        classification_map = (
            self.prediction_prob_dhw if self.prediction_prob_dhw is not None else assets["patch_probabilities"]
        )
        return [
            (
                "Input 3D image",
                RESULT_EXPLANATIONS["input_3d"],
                make_input_3d_figure(
                    assets["raw_modalities"], self.prediction_dhw, self.patient_data.ground_truth_dhw
                ),
            ),
            (
                "Preprocessing output image",
                RESULT_EXPLANATIONS["preprocessing"],
                make_preprocessing_figure(
                    assets["raw_modalities"], assets["normalized_modalities"], assets["focus_slice"]
                ),
            ),
            (
                "Feature extracted output image",
                RESULT_EXPLANATIONS["features"],
                make_feature_figure(assets["activations"]),
            ),
            (
                "Segmentation output",
                RESULT_EXPLANATIONS["segmentation"],
                make_segmentation_figure(
                    assets["normalized_modalities"],
                    self.prediction_dhw,
                    self.patient_data.ground_truth_dhw,
                    assets["focus_slice"],
                ),
            ),
            (
                "Classification layer output",
                RESULT_EXPLANATIONS["classification"],
                make_classification_figure(
                    assets["feature_patch"], classification_map, assets["focus_slice"]
                ),
            ),
            (
                "Methodology",
                RESULT_EXPLANATIONS["methodology"],
                make_methodology_figure(),
            ),
        ]

    def on_view_results(self) -> None:
        if not self._report_prerequisites_ok("the results view"):
            return

        self.status.showMessage("Building result outputs from real patient prediction...")
        try:
            assets = self._prepare_report_assets()
            sections = self._build_result_sections(assets)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to build results", str(exc))
            self.status.showMessage("Failed to build results", 3000)
            return

        dialog = ResultsDialog(self.patient_data.patient_id, sections, parent=self)
        dialog.show()
        self.results_dialog = dialog
        self.status.showMessage("Results ready", 3000)

    def _patient_export_dir(self) -> Path:
        """Absolute inference_app/export/<patient_id>/ folder, created if needed."""
        export_dir = APP_DIR / "export" / self.patient_data.patient_id
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def on_export_report(self) -> None:
        if not self._report_prerequisites_ok("the report"):
            return

        pid = self.patient_data.patient_id

        # Let the user choose WHERE the bundle goes; it is always organized into a
        # <chosen base>/<patient_id>/ subfolder. Defaults to inference_app/export.
        default_base = APP_DIR / "export"
        default_base.mkdir(parents=True, exist_ok=True)
        base_dir = QFileDialog.getExistingDirectory(
            self, "Select base folder (a <patient_id> subfolder is created inside it)", str(default_base)
        )
        if not base_dir:
            return

        export_dir = Path(base_dir) / pid
        export_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = export_dir / f"report_{pid}.pdf"
        mask_path = export_dir / f"pred_{pid}.nii.gz"

        self.status.showMessage(f"Exporting organized report bundle to {export_dir} ...")

        # Save the real predicted mask alongside the report.
        pred_hwd = np.transpose(self.prediction_dhw, (1, 2, 0)).astype(np.uint8)
        pred_nii = nib.Nifti1Image(pred_hwd, affine=self.patient_data.ref_nii.affine, header=self.patient_data.ref_nii.header)
        nib.save(pred_nii, str(mask_path))

        assets = self._prepare_report_assets()
        sections = self._build_result_sections(assets)

        stats = self._compute_stats_payload()
        metrics = compute_comparison_metrics(self.prediction_dhw, self.patient_data.ground_truth_dhw)
        summary_lines = [
            f"Patient ID: {pid}",
            "Prediction source: real BraTS patient data",
            f"Prediction path: {mask_path}",
            f"Inference time (s): {self.inference_time_sec:.2f}" if self.inference_time_sec is not None else "Inference time (s): -",
        ]

        saved_files = [pdf_path.name, mask_path.name]
        with PdfPages(str(pdf_path)) as pdf:
            cover = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            cover_ax = cover.add_axes([0, 0, 1, 1])
            cover_ax.axis("off")
            cover_text = [
                "Brain Tumor Segmentation Report",
                "",
                *summary_lines,
                "",
                f"Tumor volume (ml): {stats.get('tumor_volume_ml', 0.0):.3f}",
                f"Voxel count: {stats.get('voxel_count', 0)}",
                f"Max diameter (mm): {stats.get('max_diameter_mm', 0.0):.2f}",
                f"Bounding box: {stats.get('bounding_box', '-')}",
                f"Centroid: {stats.get('centroid', '-')}",
                "",
                f"Dice: {metrics.get('dice') if metrics.get('dice') is not None else 'N/A'}",
                f"Jaccard: {metrics.get('jaccard') if metrics.get('jaccard') is not None else 'N/A'}",
                f"Sensitivity: {metrics.get('sensitivity') if metrics.get('sensitivity') is not None else 'N/A'}",
                f"Precision: {metrics.get('precision') if metrics.get('precision') is not None else 'N/A'}",
            ]
            cover.text(0.05, 0.95, "\n".join(cover_text), va="top", fontsize=10, family="monospace")
            pdf.savefig(cover, bbox_inches="tight")
            plt.close(cover)

            # Each output: save its own image in the patient folder AND add it to the PDF.
            for title, _, figure in sections:
                image_name = SECTION_FILENAMES.get(title, f"{title}.png")
                figure.savefig(str(export_dir / image_name), dpi=140, bbox_inches="tight")
                saved_files.append(image_name)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)

        self.status.showMessage(f"Report bundle exported to {export_dir}", 5000)
        QMessageBox.information(
            self,
            "Report exported",
            f"Everything for patient {pid} was organized into:\n\n{export_dir}\n\n"
            + "\n".join(f"  - {name}" for name in saved_files),
        )

    def on_modality_changed(self, text: str) -> None:
        if text in {"Prediction", "Ground Truth"}:
            if text == "Prediction" and self.prediction_dhw is None:
                return
            if text == "Ground Truth" and (self.patient_data is None or self.patient_data.ground_truth_dhw is None):
                return

        self.viewer2d.set_modality(text)
        self._sync_slice_ui(self.viewer2d.current_slice)

    def on_slice_changed(self, idx: int) -> None:
        self.viewer2d.set_slice(idx)

    def _sync_slice_ui(self, idx: int) -> None:
        total = self.viewer2d.total_slices
        self.slice_label.setText(f"Slice: {idx + 1}/{total}")
        self.stat_slice.setText(f"Slice: {idx + 1}")
        self.stat_total_slices.setText(f"Total slices: {total}")
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(idx)
        self.slice_slider.blockSignals(False)

    def _on_cursor_changed(self, text: str) -> None:
        self.status.showMessage(text)

    def _update_overlays(self) -> None:
        self.viewer2d.set_overlay_options(
            show_pred=self.show_pred_checkbox.isChecked(),
            show_gt=self.show_gt_checkbox.isChecked(),
            opacity=self.overlay_opacity_slider.value() / 100.0,
        )

    def _update_brightness_contrast(self) -> None:
        self.viewer2d.set_brightness_contrast(
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
        )

    def _compute_stats_payload(self) -> dict:
        if self.prediction_dhw is None or self.patient_data is None:
            return {
                "voxel_count": 0,
                "tumor_volume_ml": 0.0,
                "bounding_box": None,
                "centroid": None,
                "max_diameter_mm": 0.0,
            }
        return compute_tumor_stats(self.prediction_dhw, self.patient_data.spacing_dhw)

    def _volume_band(self, volume_ml: float) -> str:
        if volume_ml <= 0:
            return "none"
        if volume_ml < 5:
            return "small"
        if volume_ml < 30:
            return "moderate"
        if volume_ml < 80:
            return "large"
        return "very large"

    def _quality_note(self, metrics: dict) -> str:
        dice = metrics.get("dice")
        if dice is None:
            return "Ground truth not loaded, so overlap quality cannot be scored."
        if dice >= 0.9:
            return "Excellent overlap with ground truth."
        if dice >= 0.8:
            return "Good overlap with ground truth."
        if dice >= 0.7:
            return "Moderate overlap; inspect boundaries in 2D/3D view."
        return "Low overlap; review input quality and checkpoint selection."

    def _update_stats(self) -> None:
        stats = self._compute_stats_payload()
        metrics = {}
        if self.patient_data is not None and self.prediction_dhw is not None:
            metrics = compute_comparison_metrics(self.prediction_dhw, self.patient_data.ground_truth_dhw)

        if self.prediction_dhw is None:
            self.pred_status_label.setText("Decision: Prediction not run")
            self.pred_summary_label.setText("Summary: Run prediction to generate a whole-tumor mask.")
            self.pred_quality_label.setText("Quality note: -")
        else:
            voxels = int(stats["voxel_count"])
            volume_ml = float(stats["tumor_volume_ml"])
            confidence_text = f"{self.last_confidence:.3f}" if self.last_confidence is not None else "-"
            if voxels > 0:
                band = self._volume_band(volume_ml)
                self.pred_status_label.setText("Decision: Tumor detected")
                self.pred_summary_label.setText(
                    f"Summary: Predicted whole-tumor region is present ({band} burden), "
                    f"volume={volume_ml:.2f} ml, confidence={confidence_text}."
                )
            else:
                self.pred_status_label.setText("Decision: No tumor detected")
                self.pred_summary_label.setText(
                    f"Summary: No tumor voxels crossed the segmentation threshold (confidence={confidence_text})."
                )
            self.pred_quality_label.setText(f"Quality note: {self._quality_note(metrics)}")

        self.stat_volume.setText(f"Tumor volume (ml): {stats['tumor_volume_ml']:.3f}")
        self.stat_voxels.setText(f"Voxel count: {stats['voxel_count']}")
        self.stat_bbox.setText(f"Bounding box: {stats['bounding_box']}")
        self.stat_centroid.setText(f"Centroid: {stats['centroid']}")
        self.stat_diameter.setText(f"Max diameter (mm): {stats['max_diameter_mm']:.2f}")

        confidence_text = f"{self.last_confidence:.3f}" if self.last_confidence is not None else "-"
        self.stat_confidence.setText(f"Confidence: {confidence_text}")

        if self.patient_data is not None and self.prediction_dhw is not None:
            self.stat_jaccard.setText(
                f"Jaccard: {metrics['jaccard']:.4f}" if metrics["jaccard"] is not None else "Jaccard: N/A"
            )
            self.stat_sensitivity.setText(
                f"Sensitivity: {metrics['sensitivity']:.4f}" if metrics["sensitivity"] is not None else "Sensitivity: N/A"
            )
            self.stat_precision.setText(
                f"Precision: {metrics['precision']:.4f}" if metrics["precision"] is not None else "Precision: N/A"
            )
        else:
            self.stat_jaccard.setText("Jaccard: -")
            self.stat_sensitivity.setText("Sensitivity: -")
            self.stat_precision.setText("Precision: -")
