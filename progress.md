# CSC173 Deep Computer Vision Project Progress Report
**Student:** Christian Dave J. Janiola, 2022-0137  
**Date:** December 13, 2025  
**Repository:** [https://github.com/ChrisDJ07/CSC173-DeepCV-Janiola](https://github.com/ChrisDJ07/CSC173-DeepCV-Janiola)


## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 263 images manually collected, annotated in Roboflow, 70/20/10 split |
| Model Training | ✅ Completed | 23 epochs (early stopped), mAP50=99.4%, mAP50-95=97.7% |
| Test Evaluation | ✅ Completed | Test set: 26 images, 101 coins, all metrics > 95% |
| Webcam Demo | ✅ Working | Real-time inference with coin counting & value computation |
| Final README | ✅ In Progress | README + demo video recording |

## 1. Dataset Summary
- **Total images:** 263 manually collected
- **Train/Val/Test split:** 70% (185) / 20% (53) / 10% (26)
- **Total coins in dataset:** 954
- **Classes implemented:** 4 denominations
  - ₱1 peso (1_peso)
  - ₱5 peso (5_peso)
  - ₱10 peso (10_peso)
  - ₱20 peso (20_peso)
- **Preprocessing applied:** 
  - Resize: 640×640
  - Rotation: ±12°
  - Brightness: ±15%
  - Blur: up to 0.6px

**Class Distribution (Training Set):**
```
1_peso:   344 instances (18.6%)
5_peso:   250 instances (37.3%)
10_peso:  172 instances (20.4%)
20_peso:  188 instances (27.1%)
─────────────────────────
Total:    954 instances (balanced enough, slight 1_peso bias)
```

**Sample data preview:**
![Dataset Sample](images/dataset_sample.jpg)

## 2. Training Progress

### Training Summary
- **Model:** YOLOv8 Nano (Transfer Learning from COCO)
- **Framework:** PyTorch + Ultralytics
- **Hardware:** Google Colab (Tesla T4 GPU)
- **Training Time:** ~3 minutes for 23 epochs
- **Stopping Criteria:** Early stopping triggered at epoch 23 (patience=10, no improvement in val_mAP)

**Training Curves (so far)**
![Training Curves](coin_yolo_training\yolov8n_coins2\results.png)

### Test Set Results (26 images, 101 coins)
| Metric | Value | Status |
|--------|-------|--------|
| **mAP50** | 98.8% | Excellent ✅ |
| **mAP50-95** | 97.1% | Excellent ✅ |
| **Precision** | 96.6% | Excellent ✅ |
| **Recall** | 95.1% | Excellent ✅ |

### Per-Class Performance (Test Set)
| Denomination | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | Status |
|--------------|--------|-----------|-----------|--------|-------|----------|--------|
| 1_peso | 11 | 33 | 100% | 90.8% | 98.7% | 95.1% | ✅ Good |
| 5_peso | 15 | 34 | 96.8% | 89.6% | 98.2% | 96.9% | ✅ Good |
| 10_peso | 12 | 21 | 90.8% | 100% | 98.9% | 98.0% | ✅ Excellent |
| 20_peso | 8 | 13 |  98.8% | 100% | 99.5% | 98.6% | ✅ Good |

**Observation:** 5_peso has slightly lower recall (89.6%); may be slightly harder to distinguish visually. Consider adding more 5_peso images for future improvements.

## 3. Issues Encountered & Resolutions

| Issue | Root Cause | Resolution | Status |
|-------|-----------|-----------|--------|
| Class names mislabeled | Roboflow export order != expected order | Fixed data.yaml with correct class order | ✅ Fixed |
| Webcam showed wrong denominations | Values dict didn't match class order | Rebuilt values dict from model.names dynamically | ✅ Fixed |
| Model too sensitive (false positives) | Default conf=0.25 too low | Increased to conf=0.6-0.7 | ✅ Mitigated |
| Training data shuffled order | Roboflow randomized class indices | Verified all classes present in training | ✅ Verified |
| Slow laptop CPU training | Ryzen 5 5500U CPU-only inference | Switched to Colab GPU for training | ✅ Solved |

## 4. Next Steps (Before Final Submission)
- [ ] **Compare to baseline**
- [ ] **Record 5-minute demo video**
- [ ] **Update README.md with final results**