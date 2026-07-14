"""
data_loader.py — BraTS dataset with PATCH-based sampling.

Why patches instead of full volumes:
A full BraTS volume is [4, 155, 240, 240]. Feeding that straight into a 3D UNet
blows past consumer GPU VRAM (RTX 3050 has 4-6GB) almost instantly once you
account for every intermediate activation the encoder/decoder has to keep
alive for backprop. Real nnU-Net-style pipelines never train on full volumes —
they always crop small 3D patches (commonly 96-128 per side) and slide a
window over the full volume only at inference time (see predict.py).

This loader:
  1. Finds each patient's 4 modalities + seg mask (flat BraTS2021 naming:
     BraTS2021_XXXXX_flair.nii.gz, _t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, _seg.nii.gz)
  2. Normalizes each modality independently (z-score over nonzero brain voxels)
  3. Crops a random patch per __getitem__ call, biased toward tumor-containing
     regions so the model doesn't spend most of its time on empty background
"""

import os
import glob
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def _resolve_to_file(path):
    """
    Given a path that might be a real .nii/.nii.gz FILE or a FOLDER whose name
    happens to end in .nii (this Kaggle BraTS2021 release does this — e.g.
    'BraTS2021_00000_flair.nii/' is a directory containing the real file
    '00000057_brain_flair.nii' inside it), return the actual readable file path.
    """
    if os.path.isdir(path):
        inner = glob.glob(os.path.join(path, "*.nii.gz")) + glob.glob(os.path.join(path, "*.nii"))
        return inner[0] if inner else None
    if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
        return path
    return None


def _find_modality(patient_path, patient_id, suffix):
    """
    Find a modality file for a patient, handling all naming styles seen in
    BraTS releases:
      - flat file:       BraTS2021_00000_flair.nii.gz
      - folder-as-file:  BraTS2021_00000_flair.nii/00000057_brain_flair.nii
      - nested folder:   */flair*/*.nii*
    `suffix` should already disambiguate t1 vs t1ce (e.g. pass a strict pattern).
    """
    candidates = sorted(glob.glob(os.path.join(patient_path, f"*{suffix}*")))
    for c in candidates:
        resolved = _resolve_to_file(c)
        if resolved:
            return resolved

    # fallback: nested folder structure one level deeper than expected
    nested = glob.glob(os.path.join(patient_path, f"*{suffix}*", "*", "*.nii*"))
    if nested:
        return nested[0]

    return None


class BraTSDataset(Dataset):

    def __init__(self, dataset_path, patch_size=(96, 96, 96), mode="train",
                 tumor_bias_prob=0.8, samples_per_volume=1):
        """
        dataset_path: root folder containing one subfolder per patient
        patch_size: (D, H, W) size of the random crop
        mode: "train" enables random crop augmentation; "val"/"test" also
              random-crops for now (use predict.py's sliding window for
              full-volume inference, not this loader)
        tumor_bias_prob: probability of centering the crop on a tumor voxel
                         instead of a fully random location. Keeping this high
                         (e.g. 0.8) matters a lot for BraTS: tumors are a small
                         fraction of brain volume, so pure random cropping
                         wastes most training on empty background.
        samples_per_volume: how many patches to yield per patient per epoch
                             (increases effective dataset size)
        """
        self.dataset_path = dataset_path
        self.patch_size = patch_size
        self.mode = mode
        self.tumor_bias_prob = tumor_bias_prob
        self.samples_per_volume = samples_per_volume

        self.patients = []

        for patient in sorted(os.listdir(dataset_path)):
            patient_path = os.path.join(dataset_path, patient)
            if not os.path.isdir(patient_path):
                continue

            flair = _find_modality(patient_path, patient, "flair")
            t1ce = _find_modality(patient_path, patient, "t1ce")
            t2 = _find_modality(patient_path, patient, "t2")
            seg = _find_modality(patient_path, patient, "seg")

            # t1 must not accidentally match t1ce files
            t1_candidates = sorted(glob.glob(os.path.join(patient_path, "*t1*")))
            t1 = None
            for c in t1_candidates:
                if "t1ce" in os.path.basename(c).lower():
                    continue
                resolved = _resolve_to_file(c)
                if resolved:
                    t1 = resolved
                    break

            if flair and t1 and t1ce and t2 and seg:
                paths = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2, "seg": seg}

                # Validate every file is actually readable before committing to
                # this patient. Corrupted/truncated downloads (common with large
                # Kaggle archives) will raise here instead of crashing training
                # mid-epoch later, when it's much more annoying to diagnose.
                corrupted = False
                for modality, path in paths.items():
                    try:
                        img = nib.load(path)
                        img.header  # forces header parse without loading full array
                    except Exception as e:
                        print(f"[BraTSDataset] Skipping {patient}: corrupted/unreadable "
                              f"{modality} file ({path}) — {e}")
                        corrupted = True
                        break

                if not corrupted:
                    self.patients.append(paths)
            else:
                print(f"[BraTSDataset] Skipping incomplete patient: {patient}")

        if len(self.patients) == 0:
            raise RuntimeError(
                f"No complete patients found under {dataset_path}. "
                f"Check folder structure / file naming."
            )

        print(f"[BraTSDataset] Loaded {len(self.patients)} patients ({mode} mode).")

    def __len__(self):
        return len(self.patients) * self.samples_per_volume

    @staticmethod
    def _zscore_normalize(volume):
        """Normalize using only nonzero (brain tissue) voxels, keep background at 0."""
        mask = volume > 0
        if mask.sum() == 0:
            return volume
        mean = volume[mask].mean()
        std = volume[mask].std() + 1e-8
        volume = volume.copy()
        volume[mask] = (volume[mask] - mean) / std
        return volume

    def _random_patch_coords(self, seg, patch_size):
        d, h, w = seg.shape
        pd, ph, pw = patch_size

        pd, ph, pw = min(pd, d), min(ph, h), min(pw, w)

        use_tumor_center = (
            random.random() < self.tumor_bias_prob and seg.sum() > 0
        )

        if use_tumor_center:
            tumor_voxels = np.argwhere(seg > 0)
            cz, cy, cx = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]
        else:
            cz = random.randint(0, d - 1)
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)

        z0 = int(np.clip(cz - pd // 2, 0, d - pd))
        y0 = int(np.clip(cy - ph // 2, 0, h - ph))
        x0 = int(np.clip(cx - pw // 2, 0, w - pw))

        return z0, y0, x0, pd, ph, pw

    def __getitem__(self, index):
        try:
            return self._load_patch(index)
        except Exception as e:
            # A file passed the header check at startup but still failed to
            # fully read (rare — e.g. disk hiccup, partial corruption deeper
            # in the file). Don't let one bad sample kill hours of training:
            # log it and substitute a different random patient instead.
            bad_patient = self.patients[index % len(self.patients)]
            print(f"[BraTSDataset] Runtime read error on {bad_patient['flair']}: {e}. "
                  f"Substituting a different sample.")
            fallback_idx = random.randint(0, len(self.patients) - 1)
            return self._load_patch(fallback_idx)

    def _load_patch(self, index):
        patient_idx = index % len(self.patients)
        patient = self.patients[patient_idx]

        flair = nib.load(patient["flair"]).get_fdata(dtype=np.float32)
        t1 = nib.load(patient["t1"]).get_fdata(dtype=np.float32)
        t1ce = nib.load(patient["t1ce"]).get_fdata(dtype=np.float32)
        t2 = nib.load(patient["t2"]).get_fdata(dtype=np.float32)
        seg = nib.load(patient["seg"]).get_fdata(dtype=np.float32)

        # BraTS labels are 0 (background), 1, 2, 4 (tumor sub-regions).
        # Binarize to whole-tumor for this baseline (any label > 0 = tumor).
        seg = (seg > 0).astype(np.float32)

        flair = self._zscore_normalize(flair)
        t1 = self._zscore_normalize(t1)
        t1ce = self._zscore_normalize(t1ce)
        t2 = self._zscore_normalize(t2)

        # H, W, D -> D, H, W to match patch_size convention
        flair = np.transpose(flair, (2, 0, 1))
        t1 = np.transpose(t1, (2, 0, 1))
        t1ce = np.transpose(t1ce, (2, 0, 1))
        t2 = np.transpose(t2, (2, 0, 1))
        seg = np.transpose(seg, (2, 0, 1))

        z0, y0, x0, pd, ph, pw = self._random_patch_coords(seg, self.patch_size)

        def crop(vol):
            return vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

        image = np.stack([crop(flair), crop(t1), crop(t1ce), crop(t2)], axis=0)
        mask = crop(seg)[np.newaxis, ...]

        # Pad if the volume was smaller than patch_size in any dimension
        pad_d = self.patch_size[0] - image.shape[1]
        pad_h = self.patch_size[1] - image.shape[2]
        pad_w = self.patch_size[2] - image.shape[3]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, 0), (0, max(pad_d, 0)), (0, max(pad_h, 0)), (0, max(pad_w, 0))),
                mode="constant",
            )
            mask = np.pad(
                mask,
                ((0, 0), (0, max(pad_d, 0)), (0, max(pad_h, 0)), (0, max(pad_w, 0))),
                mode="constant",
            )

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()

        return image, mask