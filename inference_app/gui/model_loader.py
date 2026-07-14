from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.model import UNet3D


@dataclass
class LoadedModel:
    model: UNet3D
    device: torch.device
    checkpoint_path: str
    epoch: Optional[int]
    base_channels: int


def load_model(checkpoint_path: str, base_channels: Optional[int] = None, force_cpu: bool = False) -> LoadedModel:
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    inferred_channels = checkpoint.get("base_channels", None)
    if inferred_channels is None:
        inferred_channels = base_channels if base_channels is not None else 16

    model = UNet3D(in_channels=4, out_channels=1, base_channels=int(inferred_channels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return LoadedModel(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        epoch=checkpoint.get("epoch", None),
        base_channels=int(inferred_channels),
    )
