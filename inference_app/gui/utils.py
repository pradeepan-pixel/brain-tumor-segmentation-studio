from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import BraTSDataset, _find_modality, _resolve_to_file


@dataclass
class PatientData:
    patient_id: str
    patient_dir: str
    modality_paths: Dict[str, str]
    modalities_dhw: Dict[str, np.ndarray]
    ref_nii: nib.Nifti1Image
    spacing_dhw: Tuple[float, float, float]
    ground_truth_dhw: Optional[np.ndarray]


def discover_patient_files(patient_dir: str) -> Dict[str, Optional[str]]:
    flair_path = _find_modality(patient_dir, None, "flair")
    t1ce_path = _find_modality(patient_dir, None, "t1ce")
    t2_path = _find_modality(patient_dir, None, "t2")
    seg_path = _find_modality(patient_dir, None, "seg")

    t1_candidates = sorted(glob.glob(os.path.join(patient_dir, "*t1*")))
    t1_path = None
    for candidate in t1_candidates:
        if "t1ce" in os.path.basename(candidate).lower():
            continue
        resolved = _resolve_to_file(candidate)
        if resolved:
            t1_path = resolved
            break

    return {
        "flair": flair_path,
        "t1": t1_path,
        "t1ce": t1ce_path,
        "t2": t2_path,
        "seg": seg_path,
    }


def _has_required_modalities(patient_dir: str) -> bool:
    paths = discover_patient_files(patient_dir)
    return all([paths["flair"], paths["t1"], paths["t1ce"], paths["t2"]])


def resolve_patient_dir(selected_path: str) -> str:
    """
    Resolve user selection to a BraTS patient root directory.

    Supports selecting either:
      - patient folder (preferred)
      - a modality file/folder such as *_seg.nii or *_flair.nii
    """
    if not selected_path:
        raise FileNotFoundError("No folder selected")

    candidate = selected_path
    if os.path.isfile(candidate):
        candidate = os.path.dirname(candidate)

    candidate = os.path.normpath(candidate)

    # Fast path: user selected the patient folder directly.
    if os.path.isdir(candidate) and _has_required_modalities(candidate):
        return candidate

    base = os.path.basename(candidate)
    parent = os.path.dirname(candidate)
    match = re.match(r"(.+?)_(flair|t1ce|t1|t2|seg)(\.nii(\.gz)?)?$", base, re.IGNORECASE)
    if match:
        patient_id = match.group(1)
        patient_root = os.path.join(parent, patient_id)
        if os.path.isdir(patient_root) and _has_required_modalities(patient_root):
            return patient_root

    # Fallback: if one level above has all modalities, accept it.
    if os.path.isdir(parent) and _has_required_modalities(parent):
        return parent

    raise FileNotFoundError(
        "Selected path is not a patient root and could not be auto-resolved. "
        "Please select the patient folder (e.g., BraTS2021_00000)."
    )


def load_patient_data(patient_dir: str) -> PatientData:
    patient_dir = resolve_patient_dir(patient_dir)
    paths = discover_patient_files(patient_dir)
    required = [paths["flair"], paths["t1"], paths["t1ce"], paths["t2"]]
    if not all(required):
        raise FileNotFoundError(f"Could not find all required modalities in {patient_dir}")

    flair_nii = nib.load(paths["flair"])

    def _load_and_normalize(path: str) -> np.ndarray:
        volume_hwd = nib.load(path).get_fdata(dtype=np.float32)
        volume_hwd = BraTSDataset._zscore_normalize(volume_hwd)
        return np.transpose(volume_hwd, (2, 0, 1))

    modalities_dhw = {
        "FLAIR": _load_and_normalize(paths["flair"]),
        "T1": _load_and_normalize(paths["t1"]),
        "T1CE": _load_and_normalize(paths["t1ce"]),
        "T2": _load_and_normalize(paths["t2"]),
    }

    gt_dhw = None
    if paths["seg"]:
        gt_hwd = nib.load(paths["seg"]).get_fdata(dtype=np.float32)
        gt_dhw = np.transpose((gt_hwd > 0).astype(np.uint8), (2, 0, 1))

    zooms = flair_nii.header.get_zooms()[:3]
    spacing_dhw = (float(zooms[2]), float(zooms[0]), float(zooms[1]))

    return PatientData(
        patient_id=os.path.basename(os.path.normpath(patient_dir)),
        patient_dir=patient_dir,
        modality_paths={k: v for k, v in paths.items() if v},
        modalities_dhw=modalities_dhw,
        ref_nii=flair_nii,
        spacing_dhw=spacing_dhw,
        ground_truth_dhw=gt_dhw,
    )


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    return 1.0 if total == 0 else (2.0 * inter) / total


def jaccard_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return 1.0 if union == 0 else inter / union


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    return 1.0 if (tp + fn) == 0 else tp / (tp + fn)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    return 1.0 if (tp + fp) == 0 else tp / (tp + fp)


def mask_bounding_box(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    return (int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))


def compute_tumor_stats(mask_dhw: np.ndarray, spacing_dhw: Tuple[float, float, float]) -> Dict[str, object]:
    voxel_count = int(mask_dhw.sum())
    voxel_volume_mm3 = float(spacing_dhw[0] * spacing_dhw[1] * spacing_dhw[2])
    tumor_volume_mm3 = voxel_count * voxel_volume_mm3
    tumor_volume_ml = tumor_volume_mm3 / 1000.0

    bbox = mask_bounding_box(mask_dhw)
    centroid = None
    max_diameter_mm = 0.0
    if voxel_count > 0:
        coords = np.argwhere(mask_dhw > 0)
        centroid_arr = coords.mean(axis=0)
        centroid = (float(centroid_arr[0]), float(centroid_arr[1]), float(centroid_arr[2]))

    if bbox:
        z, y, x = bbox
        dz = (z[1] - z[0] + 1) * spacing_dhw[0]
        dy = (y[1] - y[0] + 1) * spacing_dhw[1]
        dx = (x[1] - x[0] + 1) * spacing_dhw[2]
        max_diameter_mm = float(max(dz, dy, dx))

    return {
        "voxel_count": voxel_count,
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_volume_ml": tumor_volume_ml,
        "bounding_box": bbox,
        "centroid": centroid,
        "max_diameter_mm": max_diameter_mm,
    }


def compute_comparison_metrics(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if gt is None:
        return {
            "dice": None,
            "jaccard": None,
            "sensitivity": None,
            "precision": None,
        }
    return {
        "dice": float(dice_score(pred, gt)),
        "jaccard": float(jaccard_score(pred, gt)),
        "sensitivity": float(sensitivity(pred, gt)),
        "precision": float(precision(pred, gt)),
    }
