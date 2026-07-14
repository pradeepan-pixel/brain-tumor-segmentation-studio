"""
main.py — Quick sanity check that the model runs on a PATCH-sized input
(not a full volume — see the memory explanation in src/model.py and
src/data_loader.py for why).

For actual training/inference, use:
    python src/train.py --dataset_path <path_to_BraTS_folder>
    python src/predict.py --checkpoint checkpoints/best.pth --patient_dir <patient_folder>
"""

import torch
from src.model import UNet3D

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(in_channels=4, out_channels=1, base_channels=16).to(device)

    # Patch-sized input — matches what train.py actually feeds the model.
    # Do NOT test with a full [1,4,155,240,240] volume here; that's what
    # was OOMing your machine before.
    x = torch.randn(1, 4, 96, 96, 96).to(device)

    y = model(x)
    print("Input Shape:", x.shape)
    print("Output Shape:", y.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
