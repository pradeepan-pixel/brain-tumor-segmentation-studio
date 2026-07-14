#!/usr/bin/env bash
#
# start_app.sh — one-click launcher for Brain Tumor Segmentation Studio.
#
# Goes into the project venv first, then runs the GUI with the native VTK 3D
# backend enabled. When you close the app window, this script exits fully too.
#
set -e

APP_DIR="/home/pradeepan/brain_tumor_segmentation/inference_app"
cd "$APP_DIR"

# 1) Enter the virtual environment.
# shellcheck disable=SC1091
source "$APP_DIR/venv/bin/activate"

# 2) 3D view settings (VTK backend on X11).
export QT_QPA_PLATFORM=xcb
export BTS_3D_BACKEND=vtk

# 3) Run the app. `exec` replaces this shell with python so that when the
#    GUI closes, the process tree exits cleanly with no leftover shell.
exec python main_gui.py
