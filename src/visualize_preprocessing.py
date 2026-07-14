"""
visualize_preprocessing.py — Before/after normalization comparison for your
"Preprocessing Output" results section.

This is a thin CLI over the SAME engine the GUI app uses (src/report_visuals.py),
so the image produced here matches exactly what the app shows under the
"Preprocessing output image" heading and in the exported PDF report.

Run:
    python src/visualize_preprocessing.py \
        --patient_dir dataset/archive/BraTS2021/BraTS2021_00000 \
        --output_image outputs/preprocessing_comparison.png
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.report_visuals import (
    load_raw_modalities,
    make_preprocessing_figure,
    zscore_modalities,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", type=str, required=True)
    parser.add_argument("--output_image", type=str, default="outputs/preprocessing_comparison.png")
    parser.add_argument("--slice_index", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    raw_modalities = load_raw_modalities(args.patient_dir)
    normalized_modalities = zscore_modalities(raw_modalities)

    total_slices = raw_modalities["FLAIR"].shape[0]
    slice_index = args.slice_index if args.slice_index is not None else total_slices // 2

    fig = make_preprocessing_figure(raw_modalities, normalized_modalities, slice_index)

    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    fig.savefig(args.output_image, dpi=150)
    plt.close(fig)
    print(f"[visualize_preprocessing] Saved {args.output_image}")


if __name__ == "__main__":
    main()
