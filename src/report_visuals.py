from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import BraTSDataset, _find_modality, _resolve_to_file


def load_raw_modalities(patient_dir: str) -> Dict[str, np.ndarray]:
    flair_path = _find_modality(patient_dir, None, "flair")
    t1ce_path = _find_modality(patient_dir, None, "t1ce")
    t2_path = _find_modality(patient_dir, None, "t2")

    t1_candidates = sorted(glob.glob(os.path.join(patient_dir, "*t1*")))
    t1_path = None
    for candidate in t1_candidates:
        if "t1ce" in os.path.basename(candidate).lower():
            continue
        resolved = _resolve_to_file(candidate)
        if resolved:
            t1_path = resolved
            break

    paths = {"FLAIR": flair_path, "T1": t1_path, "T1CE": t1ce_path, "T2": t2_path}
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise FileNotFoundError(f"Missing modalities for report export: {', '.join(missing)}")

    raw_modalities: Dict[str, np.ndarray] = {}
    for name, path in paths.items():
        volume_hwd = nib.load(path).get_fdata(dtype=np.float32)
        raw_modalities[name] = np.transpose(volume_hwd, (2, 0, 1))
    return raw_modalities


def zscore_modalities(raw_modalities: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: BraTSDataset._zscore_normalize(volume) for name, volume in raw_modalities.items()}


def select_focus_slice(prediction_dhw: Optional[np.ndarray], ground_truth_dhw: Optional[np.ndarray], total_slices: int) -> int:
    if prediction_dhw is not None and np.any(prediction_dhw):
        slice_scores = prediction_dhw.sum(axis=(1, 2))
        return int(np.argmax(slice_scores))
    if ground_truth_dhw is not None and np.any(ground_truth_dhw):
        slice_scores = ground_truth_dhw.sum(axis=(1, 2))
        return int(np.argmax(slice_scores))
    return max(0, total_slices // 2)


def select_focus_center(
    prediction_dhw: Optional[np.ndarray],
    ground_truth_dhw: Optional[np.ndarray],
    fallback_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    mask = None
    if prediction_dhw is not None and np.any(prediction_dhw):
        mask = prediction_dhw > 0
    elif ground_truth_dhw is not None and np.any(ground_truth_dhw):
        mask = ground_truth_dhw > 0

    if mask is None:
        return fallback_shape[0] // 2, fallback_shape[1] // 2, fallback_shape[2] // 2

    coords = np.argwhere(mask)
    center = coords.mean(axis=0)
    return tuple(int(round(value)) for value in center)


def _extract_centered_patch(volume: np.ndarray, center: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    slices = []
    pads = []
    for dim, c, size in zip(volume.shape[1:], center, patch_size):
        start = int(c) - size // 2
        end = start + size
        pad_before = max(0, -start)
        pad_after = max(0, end - dim)
        start = max(0, start)
        end = min(dim, end)
        slices.append(slice(start, end))
        pads.append((pad_before, pad_after))

    patch = volume[:, slices[0], slices[1], slices[2]]
    if any(before or after for before, after in pads):
        patch = np.pad(patch, [(0, 0), *pads], mode="constant")
    return patch


def build_feature_patch(
    normalized_modalities_dhw: Dict[str, np.ndarray],
    prediction_dhw: Optional[np.ndarray],
    ground_truth_dhw: Optional[np.ndarray],
    patch_size: Tuple[int, int, int] = (96, 96, 96),
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    image = np.stack(
        [
            normalized_modalities_dhw["FLAIR"],
            normalized_modalities_dhw["T1"],
            normalized_modalities_dhw["T1CE"],
            normalized_modalities_dhw["T2"],
        ],
        axis=0,
    )
    center = select_focus_center(prediction_dhw, ground_truth_dhw, image.shape[1:])
    patch = _extract_centered_patch(image, center, patch_size)
    return patch, center


def capture_encoder_feature_maps(
    model: torch.nn.Module,
    patch: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    activations: Dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def hook(module, inputs, output):
            activations[name] = output.detach().cpu().numpy()

        return hook

    hooks = [
        model.enc1.register_forward_hook(make_hook("enc1")),
        model.enc2.register_forward_hook(make_hook("enc2")),
        model.enc3.register_forward_hook(make_hook("enc3")),
    ]

    try:
        model.eval()
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(patch_tensor)
            probs = torch.sigmoid(logits)
        return (
            activations,
            logits.squeeze(0).squeeze(0).detach().cpu().numpy(),
            probs.squeeze(0).squeeze(0).detach().cpu().numpy(),
        )
    finally:
        for hook in hooks:
            hook.remove()


def _set_dark_axes(ax) -> None:
    ax.set_facecolor("#0f1218")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color("white")


def make_input_3d_figure(
    raw_modalities_dhw: Dict[str, np.ndarray],
    prediction_dhw: Optional[np.ndarray],
    ground_truth_dhw: Optional[np.ndarray],
    max_points: int = 14000,
) -> Figure:
    flair = raw_modalities_dhw["FLAIR"]
    positive = flair[flair > 0]
    threshold = float(np.percentile(positive, 65)) if positive.size else float(np.percentile(flair, 80))
    brain_mask = flair > threshold

    def _sample_coords(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return coords
        if coords.shape[0] > max_points:
            step = max(1, coords.shape[0] // max_points)
            coords = coords[::step]
        return coords

    brain_coords = _sample_coords(brain_mask)
    pred_coords = _sample_coords(prediction_dhw > 0) if prediction_dhw is not None else np.empty((0, 3), dtype=int)
    gt_coords = _sample_coords(ground_truth_dhw > 0) if ground_truth_dhw is not None else np.empty((0, 3), dtype=int)

    fig = plt.figure(figsize=(11, 8), facecolor="#0f1218")
    ax = fig.add_subplot(111, projection="3d")
    _set_dark_axes(ax)
    ax.set_facecolor("#0f1218")

    if brain_coords.size:
        ax.scatter(brain_coords[:, 2], brain_coords[:, 1], brain_coords[:, 0], s=1, c="#8da2b5", alpha=0.03, linewidths=0)
    if gt_coords.size:
        ax.scatter(gt_coords[:, 2], gt_coords[:, 1], gt_coords[:, 0], s=4, c="#2ecc71", alpha=0.35, linewidths=0)
    if pred_coords.size:
        ax.scatter(pred_coords[:, 2], pred_coords[:, 1], pred_coords[:, 0], s=5, c="#ff3b30", alpha=0.60, linewidths=0)

    ax.set_title("Input 3D image from the real patient volume", color="white", pad=18, fontsize=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.text2D(
        0.02,
        0.02,
        "Blue cloud = actual FLAIR structure | Red = predicted tumor | Green = ground truth",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
    )
    return fig


def make_preprocessing_figure(
    raw_modalities_dhw: Dict[str, np.ndarray],
    normalized_modalities_dhw: Dict[str, np.ndarray],
    slice_index: int,
) -> Figure:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")
    for ax in axes.flat:
        ax.set_axis_off()

    for col, name in enumerate(["FLAIR", "T1", "T1CE", "T2"]):
        axes[0, col].imshow(raw_modalities_dhw[name][slice_index], cmap="gray")
        axes[0, col].set_title(f"{name} raw", fontsize=11)
        axes[1, col].imshow(normalized_modalities_dhw[name][slice_index], cmap="gray")
        axes[1, col].set_title(f"{name} normalized", fontsize=11)

    fig.suptitle("Preprocessing output on a real patient slice", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Each modality is z-score normalized before it enters the model so intensities are comparable across patients.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return fig


def make_feature_figure(activations: Dict[str, np.ndarray]) -> Figure:
    layers = ["enc1", "enc2", "enc3"]
    fig, axes = plt.subplots(len(layers), 4, figsize=(16, 11), facecolor="white")
    for row, layer_name in enumerate(layers):
        feature_maps = activations[layer_name][0]
        channels = feature_maps.shape[0]
        selected_channels = np.linspace(0, channels - 1, 4).astype(int)
        slice_index = feature_maps.shape[1] // 2
        for col, channel_index in enumerate(selected_channels):
            axes[row, col].imshow(feature_maps[channel_index, slice_index], cmap="viridis")
            axes[row, col].set_title(f"{layer_name} ch{channel_index}", fontsize=10)
            axes[row, col].set_axis_off()

    fig.suptitle("Feature extracted output from encoder layers", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "These feature maps are captured from encoder blocks of the actual model during inference on the current patient patch.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return fig


def make_segmentation_figure(
    normalized_modalities_dhw: Dict[str, np.ndarray],
    prediction_dhw: Optional[np.ndarray],
    ground_truth_dhw: Optional[np.ndarray],
    slice_index: int,
) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    flair = normalized_modalities_dhw["FLAIR"]
    axes[0].imshow(flair[slice_index], cmap="gray")
    axes[0].set_title("Input FLAIR", fontsize=12)
    axes[0].axis("off")

    if prediction_dhw is not None:
        axes[1].imshow(flair[slice_index], cmap="gray")
        axes[1].imshow(np.ma.masked_where(prediction_dhw[slice_index] <= 0, prediction_dhw[slice_index]), cmap="Reds", alpha=0.55)
    axes[1].set_title("Predicted segmentation", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(flair[slice_index], cmap="gray")
    if prediction_dhw is not None:
        axes[2].imshow(np.ma.masked_where(prediction_dhw[slice_index] <= 0, prediction_dhw[slice_index]), cmap="Reds", alpha=0.45)
    if ground_truth_dhw is not None:
        axes[2].imshow(np.ma.masked_where(ground_truth_dhw[slice_index] <= 0, ground_truth_dhw[slice_index]), cmap="Greens", alpha=0.35)
    axes[2].set_title("Prediction vs. ground truth", fontsize=12)
    axes[2].axis("off")

    fig.suptitle("Segmentation output for the selected patient slice", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "The red overlay is the model prediction generated from the real patient scan used in this session.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return fig


def make_classification_figure(
    patch: np.ndarray,
    probability_map: np.ndarray,
    slice_index: Optional[int] = None,
    threshold: float = 0.5,
) -> Figure:
    if probability_map.ndim == 3:
        slice_idx = probability_map.shape[0] // 2 if slice_index is None else int(np.clip(slice_index, 0, probability_map.shape[0] - 1))
        probability_slice = probability_map[slice_idx]
    else:
        probability_slice = probability_map

    patch_slice = patch[0, patch.shape[1] // 2] if patch.ndim == 4 else patch[patch.shape[0] // 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    axes[0].imshow(patch_slice, cmap="gray")
    axes[0].set_title("Input patch (FLAIR)", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(probability_slice, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Classification layer output", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow((probability_slice > threshold).astype(np.uint8), cmap="gray")
    axes[2].set_title(f"Thresholded mask ({threshold:.2f})", fontsize=12)
    axes[2].axis("off")

    fig.suptitle("Classification layer output for the patient-focused patch", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "This is the voxel-wise sigmoid probability map produced directly by the model before binarization.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return fig


def make_prediction_comparison_figure(
    raw_modalities_dhw: Dict[str, np.ndarray],
    ground_truth_dhw: Optional[np.ndarray],
    prediction_dhw: Optional[np.ndarray],
    slice_index: int,
) -> Figure:
    panels = [
        ("FLAIR", raw_modalities_dhw["FLAIR"][slice_index]),
        ("T1", raw_modalities_dhw["T1"][slice_index]),
        ("T1CE", raw_modalities_dhw["T1CE"][slice_index]),
        ("T2", raw_modalities_dhw["T2"][slice_index]),
        ("Ground Truth", ground_truth_dhw[slice_index] if ground_truth_dhw is not None else None),
        ("Prediction", prediction_dhw[slice_index] if prediction_dhw is not None else None),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), facecolor="white")
    for ax, (title, image) in zip(axes, panels):
        if image is not None:
            ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Prediction comparison on a real patient slice", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_methodology_figure() -> Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "Methodology",
        "",
        "Input. A 4-channel 3D MRI volume (FLAIR, T1, T1CE, T2) from the BraTS",
        "patient folder is aligned into a single stack.",
        "",
        "Preprocessing. Each modality is z-score normalized independently so the",
        "network sees a standardized intensity distribution across patients and",
        "scanners (MRI intensities are not standardized like CT Hounsfield units).",
        "",
        "Feature extraction. A 3D U-Net encoder (enc1-enc3) progressively extracts",
        "spatial features: shallow layers respond to edges and tissue boundaries,",
        "deeper layers to coarser, tumor-correlated structures. Skip connections",
        "carry fine detail to the decoder to preserve tumor boundaries.",
        "",
        "Inference. Because a full 240 x 240 x 155 volume does not fit in GPU",
        "memory, the model runs on overlapping 96 x 96 x 96 patches and the full",
        "prediction is reconstructed with sliding-window stitching.",
        "",
        "Classification layer. The final 1x1x1 convolution + sigmoid performs",
        "voxel-wise binary classification: every voxel gets a tumor probability,",
        "thresholded at 0.5 for the final mask. Trained with combined Dice + BCE",
        "loss to handle tumor/background class imbalance.",
        "",
        "All figures in this report are generated from the real patient scan and",
        "the loaded model checkpoint, not from synthetic placeholders.",
    ]

    ax.text(0.05, 0.97, "\n".join(lines), va="top", fontsize=10, family="monospace")
    fig.text(
        0.5,
        0.03,
        "Report explanation: every panel in this PDF is built from the actual patient prediction session.",
        ha="center",
        fontsize=10,
    )
    return fig
