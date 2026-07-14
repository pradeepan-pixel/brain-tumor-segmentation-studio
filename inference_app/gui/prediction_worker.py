from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .utils import load_patient_data


class PredictionThread(QThread):
    progress_changed = pyqtSignal(int, str)
    prediction_ready = pyqtSignal(object, object, str, float, float)
    error = pyqtSignal(str)

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        patient_dir: str,
        output_path: str,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        stride: Tuple[int, int, int] = (64, 64, 64),
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.patient_dir = patient_dir
        self.output_path = output_path
        self.patch_size = patch_size
        self.stride = stride
        self.threshold = threshold

    @torch.no_grad()
    def _sliding_window(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, depth, height, width = image.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        pad_d = max(0, pd - depth)
        pad_h = max(0, ph - height)
        pad_w = max(0, pw - width)

        image_padded = np.pad(
            image,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
        )
        _, depth_p, height_p, width_p = image_padded.shape

        prob_sum = np.zeros((depth_p, height_p, width_p), dtype=np.float32)
        count_map = np.zeros((depth_p, height_p, width_p), dtype=np.float32)

        z_starts = list(range(0, max(depth_p - pd, 1), sd)) or [0]
        y_starts = list(range(0, max(height_p - ph, 1), sh)) or [0]
        x_starts = list(range(0, max(width_p - pw, 1), sw)) or [0]

        if z_starts[-1] + pd < depth_p:
            z_starts.append(depth_p - pd)
        if y_starts[-1] + ph < height_p:
            y_starts.append(height_p - ph)
        if x_starts[-1] + pw < width_p:
            x_starts.append(width_p - pw)

        total_windows = len(z_starts) * len(y_starts) * len(x_starts)
        done = 0

        self.model.eval()
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = image_padded[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)

                    logits = self.model(patch_t)
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

                    prob_sum[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += probs
                    count_map[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] += 1.0

                    done += 1
                    progress = int((done / total_windows) * 100)
                    self.progress_changed.emit(progress, f"Sliding window {done}/{total_windows}")

        count_map[count_map == 0] = 1.0
        prob_avg = prob_sum / count_map
        prob_avg = prob_avg[:depth, :height, :width]

        pred = (prob_avg > self.threshold).astype(np.uint8)
        return pred, prob_avg

    def run(self) -> None:
        try:
            start_time = time.perf_counter()
            self.progress_changed.emit(5, "Loading patient modalities")
            patient_data = load_patient_data(self.patient_dir)
            image = np.stack(
                [
                    patient_data.modalities_dhw["FLAIR"],
                    patient_data.modalities_dhw["T1"],
                    patient_data.modalities_dhw["T1CE"],
                    patient_data.modalities_dhw["T2"],
                ],
                axis=0,
            )
            ref_nii = patient_data.ref_nii

            self.progress_changed.emit(15, "Running model inference")
            pred_dhw, prob_dhw = self._sliding_window(image)

            pred_hwd = np.transpose(pred_dhw, (1, 2, 0))
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            pred_nii = nib.Nifti1Image(pred_hwd.astype(np.uint8), affine=ref_nii.affine, header=ref_nii.header)
            nib.save(pred_nii, self.output_path)

            elapsed = time.perf_counter() - start_time
            tumor_probs = prob_dhw[pred_dhw > 0]
            confidence = float(tumor_probs.mean()) if tumor_probs.size else 0.0

            self.progress_changed.emit(100, "Prediction completed")
            self.prediction_ready.emit(pred_dhw, prob_dhw, self.output_path, elapsed, confidence)
        except Exception as exc:
            self.error.emit(str(exc))
