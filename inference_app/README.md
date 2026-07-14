# Brain Tumor Segmentation Studio

Standalone inference and visualization desktop app for your trained BraTS model.

## Run

From project root:

```bash
pip install -r requirements.txt
python inference_app/main_gui.py
```

## Workflow

1. Click **Load Patient** and select a BraTS patient folder.
2. Click **Load Model** and select `checkpoints/best.pth`.
3. Click **Run Prediction** to run sliding-window inference in background.
4. Review 2D overlays and statistics.
5. Click **Generate 3D** for interactive tumor rendering.
6. Export PNG and PDF report from bottom actions.

## Notes

- Training code and architecture are unchanged.
- Existing model and prediction logic are reused from `src/model.py` and `src/predict.py`.
- Detailed demo study guide: `inference_app/DEMO_GUIDE.md`

## Linux troubleshooting (Qt plugin xcb)

If you see this error:

`Could not load the Qt platform plugin "xcb"`

Install missing system libraries (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y \
	libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 libglu1-mesa \
	libxcb-icccm4 libxcb-keysyms1
```

Then run again:

```bash
python inference_app/main_gui.py
```

If your desktop session is Wayland and xcb still fails, force Wayland:

```bash
QT_QPA_PLATFORM=wayland python inference_app/main_gui.py
```

## 3D viewer behavior on Linux

- Primary 3D mode uses VTK/PyVista for full interactive rendering.
- If VTK fails on your graphics stack, the app now automatically opens a fallback 3D viewer using Matplotlib voxels.
- Fallback still supports rotation, zoom, tumor visibility selection, and screenshot export for reports.

By default on Linux, the app opens fallback 3D mode to avoid fatal VTK window crashes.

To force VTK mode manually:

```bash
BTS_3D_BACKEND=vtk python inference_app/main_gui.py
```
