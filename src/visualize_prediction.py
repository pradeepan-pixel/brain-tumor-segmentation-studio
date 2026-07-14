"""
visualize_prediction.py — Plot FLAIR / T1 / T1CE / T2 / Ground Truth / Prediction
side by side for one slice.

This is a thin CLI over the SAME engine the GUI app uses (src/report_visuals.py),
so the comparison produced here matches what the app shows under the
"Segmentation output" heading and in the exported PDF report.

This does NOT re-run the model — it loads the .nii.gz files already on disk
(the 4 modalities + the prediction saved by predict.py) and picks a slice.

Run:
    python src/visualize_prediction.py \
        --patient_dir dataset/archive/BraTS2021/BraTS2021_00000 \
        --prediction_path outputs/pred_00000.nii.gz \
        --output_image outputs/comparison_00000.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_loader import _find_modality
from src.report_visuals import (
    load_raw_modalities,
    make_prediction_comparison_figure,
    select_focus_slice,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", type=str, required=True,
                         help="Folder containing that patient's flair/t1/t1ce/t2/seg files")
    parser.add_argument("--prediction_path", type=str, required=True,
                         help="Path to the .nii.gz saved by predict.py")
    parser.add_argument("--output_image", type=str, default="outputs/comparison.png")
    parser.add_argument("--slice_index", type=int, default=None,
                         help="Which axial slice (D axis) to show. If omitted, "
                              "auto-picks the slice with the most tumor.")
    return parser.parse_args()


def _load_dhw(path):
    """Load a NIfTI as (D, H, W) to match the report engine's orientation."""
    volume_hwd = nib.load(path).get_fdata(dtype=np.float32)
    return np.transpose(volume_hwd, (2, 0, 1))


def main():
    args = parse_args()

    raw_modalities = load_raw_modalities(args.patient_dir)

    seg_path = _find_modality(args.patient_dir, None, "seg")
    gt_dhw = None
    if seg_path:
        gt_dhw = (_load_dhw(seg_path) > 0).astype(np.uint8)

    pred_dhw = (_load_dhw(args.prediction_path) > 0).astype(np.uint8)

    total_slices = raw_modalities["FLAIR"].shape[0]
    if args.slice_index is not None:
        slice_idx = args.slice_index
    else:
        slice_idx = select_focus_slice(pred_dhw, gt_dhw, total_slices)
        print(f"[visualize] Auto-picked slice {slice_idx} (most tumor voxels)")

    fig = make_prediction_comparison_figure(raw_modalities, gt_dhw, pred_dhw, slice_idx)
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    fig.savefig(args.output_image, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved comparison image to {args.output_image}")

    if gt_dhw is not None:
        intersection = (pred_dhw[slice_idx] * gt_dhw[slice_idx]).sum()
        union = pred_dhw[slice_idx].sum() + gt_dhw[slice_idx].sum()
        slice_dice = (2 * intersection) / (union + 1e-8)
        print(f"[visualize] Slice-level Dice (informal, single slice only): {slice_dice:.4f}")


if __name__ == "__main__":
    main()
