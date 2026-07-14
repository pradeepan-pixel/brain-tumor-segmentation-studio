<p align="center">
  <img src="screenshots/banner.png" alt="Brain Tumor Segmentation Studio Banner" width="100%">
</p>

<h1 align="center">🧠 Brain Tumor Segmentation Studio</h1>

<p align="center">
A Deep Learning Desktop Application for Brain Tumor Segmentation from Multi-Modal MRI using a Custom 3D U-Net
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![PyQt6](https://img.shields.io/badge/PyQt6-Desktop%20GUI-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</p>

---

# 📌 Project Overview

Brain Tumor Segmentation Studio is a desktop application that automatically detects and segments brain tumors from multi-modal MRI scans using a custom-built **3D U-Net** deep learning model trained on the **BraTS2021 dataset**.

The application provides an end-to-end workflow—from loading MRI scans and performing segmentation to visualizing results in both **2D and 3D**, computing tumor statistics, and exporting reports.

---

# ✨ Key Features

- 🧠 Custom 3D U-Net implementation in PyTorch
- 🩻 Multi-modal MRI support (FLAIR, T1, T1CE, T2)
- 🖥️ Interactive PyQt6 desktop application
- 📊 Automatic tumor segmentation
- 🌍 Interactive 3D tumor visualization
- 🖼️ 2D MRI slice viewer
- 📈 Tumor volume and prediction statistics
- 📄 PDF report generation
- 💾 Model checkpoint loading
- ⚡ GPU acceleration using CUDA

---

# 📷 Application Preview

## 🏠 Home Screen

![Home](screenshots/homepage.png)

---

## 📂 Loading Patient MRI

![Patient](screenshots/patients_load.png)

---

## 🤖 Loading Trained Model

![Model](screenshots/model.png)

---

## 🎯 Brain Tumor Prediction

![Prediction](screenshots/prediction_074.png)

---

## 🌍 3D Tumor Visualization

![3D](screenshots/3d_generated.png)

---

## 🔷 3D Surface Rendering

![Surface](screenshots/3d_volume.png)

---

## 🕸️ 3D Wireframe

![Wireframe](screenshots/3d_wireframe.png)

---

# 🏗️ System Architecture

```text
               BraTS2021 Dataset
                      │
                      ▼
         MRI Preprocessing Pipeline
                      │
                      ▼
              Custom 3D U-Net
                      │
                      ▼
            Trained Model (.pth)
                      │
                      ▼
          PyQt6 Desktop Application
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
 2D MRI Viewer                 3D Tumor Viewer
      │                               │
      └───────────────┬───────────────┘
                      ▼
          Prediction & PDF Report
```

---

# 🔬 Methodology

## 1️⃣ Input MRI

- Multi-modal MRI volumes
- FLAIR
- T1
- T1CE
- T2

![Input](screenshots/1_input_3d.png)

---

## 2️⃣ MRI Preprocessing

- Intensity normalization
- Volume resizing
- Tensor conversion

![Preprocessing](screenshots/2_preprocessing.png)

---

## 3️⃣ Feature Extraction

The encoder extracts hierarchical spatial features using 3D convolutions.

![Features](screenshots/3_features.png)

---

## 4️⃣ Segmentation Output

The decoder reconstructs a voxel-wise tumor mask.

![Segmentation](screenshots/4_segmentation.png)

---

## 5️⃣ Tumor Classification

The predicted segmentation is used to determine whether a tumor is present and to compute tumor statistics.

![Classification](screenshots/5_classification.png)

---

## 6️⃣ Complete Pipeline

![Pipeline](screenshots/6_methodology.png)

---
# 📊 Results

The model was evaluated on real BraTS2021 patient volumes using the trained checkpoint.

| Metric | Value |
|---------|------:|
| Dataset | BraTS2021 |
| Model | Custom 3D U-Net |
| Output | Binary Whole Tumor Segmentation |
| Epoch | 36 |
| Inference Time | **2.76 s** |
| Dice Score | **0.9461** |
| Jaccard (IoU) | **0.8977** |
| Sensitivity | **0.9978** |
| Precision | **0.8995** |
| Tumor Volume | **109.504 ml** |
| Confidence | **0.946** |

> Example prediction from **BraTS2021_00043** using the trained model.

# 📂 Dataset

**Dataset Used**

**BraTS2021 (Brain Tumor Segmentation Challenge)**

MRI Modalities

- FLAIR
- T1
- T1CE
- T2

Ground-truth segmentation masks are provided for supervised training.

> **Note:**  
> The dataset is **not included** in this repository because of its size and licensing requirements.

---

# 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Deep Learning | PyTorch |
| GUI | PyQt6 |
| Medical Imaging | Nibabel |
| Visualization | Matplotlib, PyVista, VTK |
| Dataset | BraTS2021 |
| Version Control | Git & GitHub |

---

# 📁 Project Structure

```text
brain_tumor_segmentation
│
├── checkpoints/
├── inference_app/
├── screenshots/
├── src/
├── README.md
├── COMPLETEOVERVIEW.md
├── requirements.txt
└── main.py
```

---

# ⚙️ Installation

```bash
git clone https://github.com/pradeepan-pixel/brain-tumor-segmentation-studio.git

cd brain-tumor-segmentation-studio

pip install -r requirements.txt
```

Run the application

```bash
python inference_app/main_gui.py
```

---

# 🚀 Usage

1. Launch the application.
2. Load a BraTS2021 patient folder.
3. Load the trained model (`best.pth`).
4. Run prediction.
5. View segmentation in 2D.
6. Generate a 3D visualization.
7. Export the report.

---

# 🔮 Future Improvements

- Multi-class tumor segmentation
- MONAI integration
- nnU-Net support
- SwinUNETR implementation
- Docker deployment
- Web application version
- Real-time inference optimization

---

# 🙏 Acknowledgements

- BraTS2021 Challenge
- PyTorch
- PyQt6
- Nibabel
- PyVista
- VTK

---

# 👨‍💻 Author

**Pradeep Kamalanathan**

Computer Science Engineering Student

GitHub:

https://github.com/pradeepan-pixel

---

<p align="center">

⭐ If you found this project interesting, consider giving it a star!

</p>
