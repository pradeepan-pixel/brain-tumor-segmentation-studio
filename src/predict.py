"""
predict.py — Run inference on a FULL MRI volume using sliding-window patches.

Why sliding window: the model was trained on small patches (e.g. 96^3) because
a full volume doesn't fit in GPU memory. At inference time we still can't feed
the whole [4, 155, 240, 240] volume at once, so we slide the same patch size
across the volume with overlap, run the model on each patch, and stitch the
results back together (averaging overlapping regions for smoother boundaries).

Run:
    python src/predict.py --checkpoint checkpoints/best.pth \
        --patient_dir dataset/archive/BraTS2021/BraTS2021_00001 \
        --output_path outputs/BraTS2021_00001_pred.nii.gz
"""

import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from model import UNet3D
from data_loader import _find_modality, _resolve_to_file, BraTSDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patient_dir", type=str, required=True,
                         help="Folder containing that patient's flair/t1/t1ce/t2 files")
    parser.add_argument("--output_path", type=str, default="outputs/prediction.nii.gz")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64],
                         help="Smaller than patch_size = overlapping windows (smoother output, slower)")
    parser.add_argument("--base_channels", type=int, default=16,
                         help="MUST match the base_channels used during training")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_patient_volumes(patient_dir):
    flair_path = _find_modality(patient_dir, None, "flair")
    t1ce_path = _find_modality(patient_dir, None, "t1ce")
    t2_path = _find_modality(patient_dir, None, "t2")

    import glob
    t1_candidates = sorted(glob.glob(os.path.join(patient_dir, "*t1*")))
    t1_path = None
    for c in t1_candidates:
        if "t1ce" in os.path.basename(c).lower():
            continue
        resolved = _resolve_to_file(c)
        if resolved:
            t1_path = resolved
            break

    if not all([flair_path, t1_path, t1ce_path, t2_path]):
        raise FileNotFoundError(f"Could not find all 4 modalities in {patient_dir}")

    ref_nii = nib.load(flair_path)  # keep affine/header for saving prediction later
    flair = nib.load(flair_path).get_fdata(dtype=np.float32)
    t1 = nib.load(t1_path).get_fdata(dtype=np.float32)
    t1ce = nib.load(t1ce_path).get_fdata(dtype=np.float32)
    t2 = nib.load(t2_path).get_fdata(dtype=np.float32)

    flair = BraTSDataset._zscore_normalize(flair)
    t1 = BraTSDataset._zscore_normalize(t1)
    t1ce = BraTSDataset._zscore_normalize(t1ce)
    t2 = BraTSDataset._zscore_normalize(t2)

    # H, W, D -> D, H, W
    volumes = [np.transpose(v, (2, 0, 1)) for v in (flair, t1, t1ce, t2)]
    image = np.stack(volumes, axis=0)  # [4, D, H, W]

    return image, ref_nii


@torch.no_grad()
def sliding_window_inference(model, image, patch_size, stride, device, threshold=0.5):
    """
    image: numpy array [4, D, H, W]
    Returns: binary prediction numpy array [D, H, W]
    """
    _, D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Pad volume so windows fit evenly
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    image_padded = np.pad(
        image, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode="constant"
    )
    _, Dp, Hp, Wp = image_padded.shape

    prob_sum = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    count_map = np.zeros((Dp, Hp, Wp), dtype=np.float32)

    z_starts = list(range(0, max(Dp - pd, 1), sd)) or [0]
    y_starts = list(range(0, max(Hp - ph, 1), sh)) or [0]
    x_starts = list(range(0, max(Wp - pw, 1), sw)) or [0]
    if z_starts[-1] + pd < Dp:
        z_starts.append(Dp - pd)
    if y_starts[-1] + ph < Hp:
        y_starts.append(Hp - ph)
    if x_starts[-1] + pw < Wp:
        x_starts.append(Wp - pw)

    total_windows = len(z_starts) * len(y_starts) * len(x_starts)
    done = 0

    model.eval()
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = image_padded[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                patch_t = torch.from_numpy(patch).float().unsqueeze(0).to(device)

                logits = model(patch_t)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

                prob_sum[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += probs
                count_map[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += 1.0

                done += 1
                if done % 5 == 0 or done == total_windows:
                    print(f"  window {done}/{total_windows}")

    count_map[count_map == 0] = 1.0
    prob_avg = prob_sum / count_map
    prob_avg = prob_avg[:D, :H, :W]  # remove padding

    return (prob_avg > threshold).astype(np.uint8)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Using device: {device}")

    model = UNet3D(in_channels=4, out_channels=1, base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[predict] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    image, ref_nii = load_patient_volumes(args.patient_dir)
    print(f"[predict] Volume shape (C, D, H, W): {image.shape}")

    pred = sliding_window_inference(
        model, image,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=device,
        threshold=args.threshold,
    )

    # D,H,W -> H,W,D to match original NIfTI orientation before saving
    pred_hwd = np.transpose(pred, (1, 2, 0))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pred_nii = nib.Nifti1Image(pred_hwd.astype(np.uint8), affine=ref_nii.affine, header=ref_nii.header)
    nib.save(pred_nii, args.output_path)
    print(f"[predict] Saved prediction to {args.output_path}")
    print(f"[predict] Predicted tumor voxel count: {pred_hwd.sum()}")


if __name__ == "__main__":
    main()