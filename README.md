# Computer Vision Portfolio

Advanced computer vision implementations in Python, covering classical image processing, deep learning, and modern CV techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👨‍🎓 About

**Xiaojun Zhang, PhD**  
Materials Science + Machine Learning + Computer Vision

Portfolio of coursework from **ENGG5104 (Image Processing and Computer Vision)** at the Chinese University of Hong Kong, demonstrating:
- Classical image processing algorithms
- Deep learning for vision tasks  
- Feature detection and matching
- Motion estimation and optical flow
- Semantic segmentation

## 🎯 Projects Overview

### 1. 🖼️ Classical Image Processing
**[View Project](01-image-processing/)** | **[Demo Notebook](demos/01_image_processing_showcase.ipynb)**

Implementation of fundamental CV algorithms from scratch:
- **Affine Transformations** - Rotation, scaling, translation with sub-pixel accuracy
- **Histogram Equalization** - Global and adaptive methods for contrast enhancement
- **Line Detection** - Hough transform for robust geometric feature detection
- **Bilateral Filtering** - Edge-preserving smoothing and joint bilateral filtering
- **Fourier Filtering** - Frequency domain high-pass filtering for edge enhancement

**Key Results:** Sub-pixel transformations, enhanced low-light images, robust line detection

**Tech Stack:** Python, NumPy, OpenCV (minimal - mostly from scratch), Matplotlib

---

### 2. 🧠 Deep Learning for Image Classification
**[View Project](02-image-classification/)** | **[Demo Notebook](demos/02_classification_demo.ipynb)**

Custom VGGNet implementation with advanced training techniques:
- **Architecture:** 11-layer CNN with <200M FLOPs constraint
- **Custom Loss:** Cross-entropy implementation from scratch (no PyTorch loss functions)
- **Data Augmentation:** Random flip, crop, Cutout regularization (no torchvision.transforms)
- **Dataset:** CIFAR-10

**Results:** 67%+ accuracy on CIFAR-10

**Tech Stack:** PyTorch, NumPy, custom implementations

---

### 3. 🎯 Local Feature Matching  
**[View Project](03-feature-matching/)** | **[Demo Notebook](demos/03_feature_matching_demo.ipynb)**

SIFT-like feature detection and matching pipeline:
- **Harris Corner Detection** - Interest point localization
- **SIFT-like Descriptors** - 128-dimensional gradient orientation histograms
- **Ratio Test Matching** - Robust correspondence finding with nearest neighbor distance ratio

**Results:** 90% matching accuracy on challenging viewpoint changes (Notre Dame, Mount Rushmore datasets)

**Tech Stack:** Python, NumPy, OpenCV (for basic operations), custom descriptor implementation

---

### 4. 🎬 Optical Flow Estimation
**[View Project](04-optical-flow/)** | **[Demo Notebook](demos/04_optical_flow_demo.ipynb)**

FlowNet-based architecture for dense motion estimation:
- **FlowNet Encoder** - Convolutional encoder for feature extraction
- **Refinement Module** - Multi-scale feature aggregation with deconvolution
- **Multi-scale Optimization** - Coarse-to-fine supervision for improved accuracy
- **Loss Functions:** EPE (End Point Error) with multi-scale weighting

**Results:** <6.0 pixels EPE on MPI Sintel dataset

**Constraints:** <2300M FLOPs, <5M parameters

**Tech Stack:** PyTorch, custom architecture implementation

---

### 5. 🏙️ Semantic Segmentation ⭐ **FEATURED PROJECT**
**[View Project](05-semantic-segmentation/)** | **[Demo Notebook](demos/05_segmentation_demo.ipynb)**

State-of-the-art semantic segmentation with efficiency focus:

**Models Implemented:**
- **PSPNet** - Pyramid Scene Parsing Network with pyramid pooling module
- **ASPP** - Atrous Spatial Pyramid Pooling for multi-scale context
- **OCR** - Object-Contextual Representations with attention mechanism
- **MobileNetV2** - Efficient depthwise separable convolutions for mobile deployment

**Dataset:** Cityscapes urban street scenes (19 classes)

**Results:**
| Backbone | Module | mIoU | Speed | Notes |
|----------|--------|------|-------|-------|
| ResNet-50 | PSPNet | 68.0% | Baseline | Best accuracy |
| ResNet-18 | PSPNet | 61.6% | 1.8x faster | Good trade-off |
| MobileNetV2 | PSPNet | 59.0% | 3.0x faster | Mobile-ready |

**Key Achievements:**
- Multiple context aggregation methods (PPM, ASPP, OCR)
- Efficiency optimization for mobile deployment
- Real-world urban scene understanding

**Tech Stack:** PyTorch, dilated convolutions, depthwise separable convolutions

---

## 🚀 Quick Start

### Installation

**Windows:**
```powershell
# Clone repository
git clone https://github.com/xzhang2365/cv-computer-vision-portfolio.git
cd cv-computer-vision-portfolio

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Clone repository
git clone https://github.com/xzhang2365/cv-computer-vision-portfolio.git
cd cv-computer-vision-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demos
```powershell
# Launch Jupyter notebooks
jupyter notebook demos/

# Or run specific projects
cd 01-image-processing
python affine_transformation/affine_transform.py --input data/sample.jpg --rotation 45
```

---

## 📊 Technical Highlights

### ✅ Custom Implementations
- Cross-entropy loss (no PyTorch built-ins)
- Data augmentation (no torchvision.transforms)
- Harris corner detector (no OpenCV keypoint functions)
- SIFT-like descriptors (custom gradient histograms)
- FlowNet architecture (custom encoder-decoder)
- PSPNet, ASPP, OCR modules

### ⚡ Computational Efficiency
- VGGNet: <200M FLOPs
- FlowNet: <2300M FLOPs, <5M parameters  
- MobileNetV2: 3x speedup vs ResNet-50

### 🎯 Advanced Techniques
- Multi-scale supervision
- Pyramid pooling modules
- Dilated/atrous convolutions
- Depthwise separable convolutions
- Attention mechanisms (OCR)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **Deep Learning** | PyTorch 2.0+, custom architectures |
| **Computer Vision** | OpenCV 4.5+, custom algorithms |
| **Scientific Computing** | NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Jupyter |
| **Datasets** | CIFAR-10, MPI Sintel, Cityscapes |

---

## 📈 Results Summary

| Task | Method | Metric | Result | Notes |
|------|--------|--------|--------|-------|
| **Classification** | VGGNet + Cutout | Accuracy | 67%+ | CIFAR-10 |
| **Feature Matching** | SIFT + Ratio Test | Accuracy | 90% | Notre Dame |
| **Optical Flow** | FlowNetERM | EPE | <6.0 px | MPI Sintel |
| **Segmentation** | PSPNet-R50 | mIoU | 68.0% | Cityscapes |
| **Segmentation** | PSPNet-MobileNetV2 | mIoU | 59.0% | 3x faster |

---

## 📁 Repository Structure
```
cv-computer-vision-portfolio/
│
├── 01-image-processing/          # Classical algorithms
│   ├── affine_transformation/
│   ├── histogram_equalization/
│   ├── line_detection/
│   ├── bilateral_filter/
│   └── fourier_filtering/
│
├── 02-image-classification/      # VGGNet on CIFAR-10
│   ├── model.py
│   ├── loss.py
│   ├── transform.py
│   └── results/
│
├── 03-feature-matching/          # SIFT-like matching
│   ├── match_functions.py
│   └── results/
│
├── 04-optical-flow/              # FlowNet
│   ├── networks/
│   ├── losses.py
│   └── results/
│
├── 05-semantic-segmentation/     # PSPNet, MobileNetV2
│   ├── models.py
│   └── results/
│
├── demos/                        # Interactive notebooks
├── docs/                         # Documentation
├── data/                         # Sample data
└── utils/                        # Shared utilities
```

---

## 🎓 Course Information

**Course:** ENGG5104 - Image Processing and Computer Vision  
**Institution:** The Chinese University of Hong Kong  
**Semester:** Spring 2022

---

## 📧 Contact

**Xiaojun Zhang, PhD**
- 📧 Email: xzhang2365@gmail.com
- 🌐 Website: [xjzhang2365.github.io](https://xjzhang2365.github.io)
- 💼 LinkedIn: [xiaojun-zhang](https://linkedin.com/in/xiaojun-zhang-9b849532)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ENGG5104 course materials, The Chinese University of Hong Kong
- CIFAR-10 dataset (Krizhevsky & Hinton)
- MPI Sintel optical flow dataset
- Cityscapes dataset for semantic segmentation

---

**⭐ Star this repo if you find it useful!**

---

*This repository showcases educational projects combining classical computer vision with modern deep learning techniques.*