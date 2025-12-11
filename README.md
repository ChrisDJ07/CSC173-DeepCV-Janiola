# Deep Learning–Based Visual Recognition and Value Computation of Philippine Coins
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Christian Dave J. Janiola, 2022-0137  
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
[150-250 words: Summarize problem (e.g., "Urban waste sorting in Mindanao"), dataset, deep CV method (e.g., YOLOv8 fine-tuned on custom trash images), key results (e.g., 92% mAP), and contributions.][web:25][web:41]

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
Computer vision has enabled machines to perceive and interpret visual information with increasing accuracy. Tasks such as image classification, object detection, and segmentation are now widely used in industries ranging from manufacturing to finance. This project proposes a deep learning–based system capable of identifying and counting Philippine coins from an image. The system will detect individual coins, classify their denominations, and compute the total monetary value present in the image.

This application is relevant for automated payment systems, coin-sorting machines, kiosk terminals, and assistive technologies for visually impaired individuals. The project is also aligned with the objectives of the “Intelligent Systems” course by providing a hands-on implementation of convolutional neural networks (CNNs) for real-world visual understanding.


### Problem Statement
Philippine coins come in several denominations with overlapping physical characteristics (e.g., size, color, reflectiveness). Currently, automated recognition systems for Philippine currency are limited. Most existing models are trained on foreign currency datasets, and readily available tools (e.g., pre-trained YOLO detectors) cannot reliably perform coin recognition without re-training

### Objectives
- To gather Philippine‑coin dataset using publicly available sources.
- To apply transfer learning on pre‑trained CNN or object detection models (e.g.YOLOv8‑n/s) for improved performance despite limited data.
- To train and validate the model using a complete pipeline including preprocessing, augmentation, hyperparameter tuning, and evaluation.
- To detect and classify coin denominations from images containing multiple coins.
- To implement a post‑processing module that counts detected units per denomination and computes the total monetary value.
- To evaluate model performance using accuracy, confusion matrices, precision/recall, and mAP.

## Related Work
- [Paper 1: YOLOv8 for real-time detection [1]]
- [Paper 2: Transfer learning on custom datasets [2]]
- [Gap: Your unique approach, e.g., Mindanao-specific waste classes] [web:25]

## Methodology
### Dataset
- Source: World Coins Dataset (Kaggle)
- Split: 70/15/15 train/val/test
- Preprocessing: Augmentation, resizing to 640x640 [web:41]

### Architecture
![Model Diagram](images/architecture.png)
- Backbone: [e.g., CSPDarknet53]
- Head: [e.g., YOLO detection layers]
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | SGD |

### Training Code Snippet
train.py excerpt
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion
- Strengths: [e.g., Handles occluded trash well]
- Limitations: [e.g., Low-light performance]
- Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/ChrisDJ07/CSC173-DeepCV-Janiola`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
ultralytics
opencv-python
albumentations

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

## GitHub Pages
View this project site: [https://github.com/ChrisDJ07/CSC173-DeepCV-Janiola](https://github.com/ChrisDJ07/CSC173-DeepCV-Janiola)

