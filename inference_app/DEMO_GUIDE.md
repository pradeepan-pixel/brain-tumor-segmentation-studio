# Brain Tumor Segmentation Studio - Demo Learning Guide

## 1. What this project is

This is a desktop inference and visualization system for brain tumor segmentation on BraTS MRI data.

It uses:
- A trained 3D U-Net checkpoint (`checkpoints/best.pth`)
- Multi-modal MRI input (FLAIR, T1, T1CE, T2)
- Sliding-window full-volume inference
- 2D and 3D visualization tools
- Quantitative tumor statistics and overlap metrics

Important:
- This app is for academic demonstration and research workflow.
- It is not a certified medical diagnosis tool.

## 2. End-to-end pipeline

1. Load patient folder
- Select one patient root, for example `dataset/archive/BraTS2021/BraTS2021_00011`
- App auto-detects modalities: FLAIR, T1, T1CE, T2
- Ground-truth segmentation is optional

2. Load model checkpoint
- Select `checkpoints/best.pth`
- App shows checkpoint metadata (epoch, base channels, device)

3. Run prediction
- Inference runs in a background thread (UI does not freeze)
- Sliding-window predicts entire MRI volume
- Output mask is auto-saved in `outputs/`

4. Visual inspection
- 2D viewer with slice scrolling and overlay controls
- 3D viewer with tumor surface/volume exploration

5. Quantitative review
- Tumor volume
- Voxel count
- Bounding box
- Centroid
- Max diameter
- Dice/Jaccard/Sensitivity/Precision (if GT exists)

6. Export
- PNG of current view
- NIfTI segmentation mask
- PDF report for presentation

## 3. What exactly is being predicted

This baseline predicts a binary Whole Tumor (WT) mask:
- `0` = background
- `1` = tumor

It does not split tumor into sub-regions (for example ET, TC, edema) in this version.

## 4. GUI panel explanation (what to say in demo)

### Left panel - Model and patient context
- Confirms loaded checkpoint and hardware (CPU/GPU)
- Shows inference time and Dice (if GT exists)
- Helps audience trust that inference used your trained model

### Center panel - 2D diagnostic view
- Shows one slice at a time
- Modality switch helps demonstrate MRI contrast differences
- Red overlay = prediction
- Green overlay = ground truth
- Opacity/brightness/contrast controls help boundary inspection

### Right panel - Prediction status and metrics
- Decision: Tumor detected or No tumor detected
- Summary: brief interpretation including volume and confidence
- Quality note: overlap interpretation when GT is available
- Metrics quantify segmentation behavior

### Bottom panel - workflow actions
- Load Patient -> Load Model -> Run Prediction -> Visualize -> Export

## 5. Meaning of each score

## Dice score
- Measures overlap between predicted and ground-truth masks.
- Range: 0 to 1 (higher is better)
- Formula: `Dice = 2TP / (2TP + FP + FN)`

## Jaccard (IoU)
- Another overlap measure, usually lower than Dice.
- Range: 0 to 1 (higher is better)
- Formula: `Jaccard = TP / (TP + FP + FN)`

## Sensitivity (Recall)
- How much real tumor the model captured.
- Formula: `TP / (TP + FN)`

## Precision
- How much predicted tumor is truly tumor.
- Formula: `TP / (TP + FP)`

## Confidence in this app
- Mean probability over voxels classified as tumor.
- Higher means stronger model confidence in positive regions.

## 6. Interpreting prediction status text

Tumor detected means:
- Predicted mask has one or more tumor voxels after thresholding.

No tumor detected means:
- No voxel crossed threshold for tumor class.
- Could mean true negative or missed lesion; always inspect MRI context.

Volume band used in summary:
- small: < 5 ml
- moderate: 5 to < 30 ml
- large: 30 to < 80 ml
- very large: >= 80 ml

## 7. Demo script (2 to 3 minute version)

1. "This is my Brain Tumor Segmentation Studio built around my trained 3D U-Net model."
2. "I load a BraTS patient folder and the trained checkpoint."
3. "When I run prediction, the app performs sliding-window inference on the full MRI volume in background."
4. "Red is model prediction and green is ground truth for comparison."
5. "On the right, the app gives a clear decision and quantitative statistics including volume and overlap metrics."
6. "I can open 3D rendering and export PNG/PDF report for documentation."

## 8. Strong points to highlight in viva/demo

- Full pipeline from trained model to deployable desktop application
- Non-blocking inference with progress tracking
- Multi-modal MRI support
- Combined qualitative (2D/3D) and quantitative evaluation
- Reproducible outputs and reporting for project documentation

## 9. Limitations you should mention honestly

- Single binary whole-tumor output (not multi-class tumor sub-region segmentation)
- Metrics depend on quality and availability of ground truth
- No clinical validation or regulatory approval
- Performance can vary across scanners and domain shifts

## 10. Detailed run steps (copy-paste)

### A) One-time system setup (Linux)

Install required OS libraries for Qt/VTK:

```bash
sudo apt update
sudo apt install -y \
	libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 libglu1-mesa \
	libxcb-icccm4 libxcb-keysyms1
```

### B) One-time Python setup

From project root:

```bash
cd /home/pradeepan/brain_tumor_segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### C) Start app (recommended stable mode)

This mode uses fallback 3D backend on Linux for crash-safe demo.

From project root:

```bash
cd /home/pradeepan/brain_tumor_segmentation
source .venv/bin/activate
python inference_app/main_gui.py
```

From inference_app folder:

```bash
cd /home/pradeepan/brain_tumor_segmentation/inference_app
source .venv/bin/activate
python main_gui.py
```

### D) Start app with original VTK 3D mode (optional)

Use this only if you want to test native VTK viewer explicitly.

From inference_app folder:

```bash
cd /home/pradeepan/brain_tumor_segmentation/inference_app
source .venv/bin/activate
QT_QPA_PLATFORM=xcb BTS_3D_BACKEND=vtk python main_gui.py
```

If VTK still crashes, try software OpenGL:

```bash
QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 BTS_3D_BACKEND=vtk python main_gui.py
```

### E) Prevent mixed environments (important)

Use only one environment at a time.

If your shell prompt shows both conda and venv (example: `(brain_seg) (.venv)`), run:

```bash
conda deactivate
source /home/pradeepan/brain_tumor_segmentation/.venv/bin/activate
```

Check active Python:

```bash
which python
python -c "import sys; print(sys.executable)"
```

Expected path should point to your project venv under:
`/home/pradeepan/brain_tumor_segmentation/.venv/`

### E2) One-click desktop launcher (recommended for demo)

A ready-made launcher is installed so you do not need the terminal:

- Startup script: `inference_app/start_app.sh`
  (enters the venv, sets `QT_QPA_PLATFORM=xcb` + `BTS_3D_BACKEND=vtk`, then runs the app)
- Desktop icon: `~/Desktop/BrainTumorSegmentation Studio`

Just double-click the desktop icon. It boots into the venv, opens the app in
VTK 3D mode, and when you close the window the whole process exits cleanly.

To run the same launcher from a terminal:

```bash
/home/pradeepan/brain_tumor_segmentation/inference_app/start_app.sh
```

If the desktop icon shows "Untrusted", right-click it once and choose
"Allow Launching" (or it is already marked trusted via `gio`).

### F) Quick demo workflow after launch

1. Load Patient: select one patient root folder, for example `dataset/archive/BraTS2021/BraTS2021_00011`
2. Load Model: select `checkpoints/best.pth`
3. Run Prediction
4. Review 2D overlays and metrics
5. Generate 3D
6. Export PNG and Export Report
