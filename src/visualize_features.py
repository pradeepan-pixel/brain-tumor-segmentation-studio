import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model import UNet3D
from predict import load_patient_volumes
from report_visuals import build_feature_patch, capture_encoder_feature_maps, make_classification_figure, make_feature_figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patient_dir", type=str, required=True)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=4, out_channels=1, base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image, ref_nii = load_patient_volumes(args.patient_dir)

    normalized_modalities = {
        "FLAIR": image[0],
        "T1": image[1],
        "T1CE": image[2],
        "T2": image[3],
    }
    patch, _ = build_feature_patch(normalized_modalities, None, None, tuple(args.patch_size))
    activations, _, probs = capture_encoder_feature_maps(model, patch, device)

    os.makedirs(args.output_dir, exist_ok=True)

    fig = make_feature_figure(activations)
    feat_path = os.path.join(args.output_dir, "feature_maps.png")
    fig.savefig(feat_path, dpi=150)
    plt.close(fig)
    print(f"[visualize_features] Saved {feat_path}")

    fig = make_classification_figure(patch, probs)
    prob_path = os.path.join(args.output_dir, "probability_map.png")
    fig.savefig(prob_path, dpi=150)
    plt.close(fig)
    print(f"[visualize_features] Saved {prob_path}")


if __name__ == "__main__":
    main()
