
# CSC173 Deep Computer Vision Project Proposal
**Student:** Christian Dave J. Janiola, 2022-0137  
**Date:** December 11, 2025

## 1. Project Title 
Deep Learning–Based Visual Recognition and Value Computation of Philippine Coins

## 2. Problem Statement
Philippine coins come in several denominations with overlapping physical characteristics (e.g., size, color, reflectiveness). Currently, automated recognition systems for Philippine currency are limited. Most existing models are trained on foreign currency datasets, and readily available tools (e.g., pre-trained YOLO detectors) cannot reliably perform coin recognition without re-training.


## 3. Objectives
- To gather Philippine‑coin dataset manually using a phone camera and spare Philippine peso coins.
- To apply transfer learning on pre‑trained CNN or object detection models (e.g.YOLOv8‑n/s) for improved performance despite limited data.
- To train and validate the model using a complete pipeline including preprocessing, augmentation, hyperparameter tuning, and evaluation.
- To detect and classify coin denominations from images containing multiple coins.
- To implement a post‑processing module that counts detected units per denomination and computes the total monetary value.
- To evaluate model performance using accuracy, confusion matrices, precision/recall, and mAP.

## 4. Dataset Plan
- Source: Manually collected data taking phone pictures with old Philippine coins up to the ₱10 denomination and the new ₱20 coin denomination
- Classes:
    - Philippine ₱1 coin
    - Philippine ₱5 coin
    - Philippine ₱10 coin
    - Philippine ₱20 coin
- Acquisition: Manual collection from spare coins on hand.

## 5. Technical Approach
- Architecture sketch:
    - YOLOv8‑n pre-trained on COCO
    - Fine‑tuning on Kaggle dataset
    - Output: bounding box + coin class
- Model: YOLOv8n
- Framework: PyTorch
- Hardware: Google Colab

## 6. Expected Challenges & Mitigations
### Challenge 1: Limited Philippine coin dataset
- Mitigation: Use transfer learning from large pre-trained models; augment heavily (rotation, brightness, blur).

### Challenge 2: Overlapping or touching coins
- Mitigation: Choose an object detector over pure classification; use YOLO or segmentation‑based approaches.

### Challenge 3: High reflectiveness and lighting variability
- Mitigation: Include lighting augmentations and collect images under varied conditions.
