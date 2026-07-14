import nibabel as nib
import matplotlib.pyplot as plt

flair_path = "dataset/archive/BraTS2021/BraTS2021_00714/BraTS2021_00714_flair.nii/00000377_brain_flair.nii"
t1_path = "dataset/archive/BraTS2021/BraTS2021_00714/BraTS2021_00714_t1.nii/00000377_brain_t1.nii"
t1ce_path = "dataset/archive/BraTS2021/BraTS2021_00714/BraTS2021_00714_t1ce.nii/00000377_brain_t1ce.nii"
t2_path = "dataset/archive/BraTS2021/BraTS2021_00714/BraTS2021_00714_t2.nii/00000377_brain_t2.nii"
seg_path = "dataset/archive/BraTS2021/BraTS2021_00714/BraTS2021_00714_seg.nii/00000377_final_seg.nii"

flair = nib.load(flair_path).get_fdata()
t1 = nib.load(t1_path).get_fdata()
t1ce = nib.load(t1ce_path).get_fdata()
t2 = nib.load(t2_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

print("FLAIR:", flair.shape)
print("T1:", t1.shape)
print("T1CE:", t1ce.shape)
print("T2:", t2.shape)
print("SEG:", seg.shape)

slice_no = flair.shape[2] // 2

plt.figure(figsize=(15,5))

plt.subplot(1,5,1)
plt.imshow(flair[:,:,slice_no], cmap="gray")
plt.title("FLAIR")

plt.subplot(1,5,2)
plt.imshow(t1[:,:,slice_no], cmap="gray")
plt.title("T1")

plt.subplot(1,5,3)
plt.imshow(t1ce[:,:,slice_no], cmap="gray")
plt.title("T1CE")

plt.subplot(1,5,4)
plt.imshow(t2[:,:,slice_no], cmap="gray")
plt.title("T2")

plt.subplot(1,5,5)
plt.imshow(seg[:,:,slice_no], cmap="gray")
plt.title("Mask")

plt.tight_layout()
plt.show()