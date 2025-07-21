# 🦷 Dental Disease Detection with YOLOv8  
  **Graduation Project**

An AI-powered dental diagnosis system for identifying **oral diseases** such as caries, gingivitis, ulcers, and more, using real-time object detection with **YOLOv8**. The system is integrated into a **mobile Flutter application**, with **FastAPI backend** for inference.

---

## 🎯 Objective

To create a lightweight and accessible tool that enables **early detection of dental abnormalities** from intraoral images captured via smartphone, assisting patients and dentists in preliminary screening.

---

## 📱 Project Components

| Component        | Description                                           |
|------------------|-------------------------------------------------------|
| 🔍 YOLOv8n       | Ultralytics model fine-tuned for 6 dental classes     |
| 🐍 FastAPI       | Python backend for serving YOLOv8 inference           |
| 📱 Flutter App   | Android mobile app (camera + result display)         |
| 📦 ONNX + TF     | Exported model for broader deployment compatibility  |

---

## 📊 Detected Dental Conditions

| Class ID | Condition              |
|----------|------------------------|
| 0        | Calculus               |
| 1        | Caries (Tooth Decay)   |
| 2        | Gingivitis             |
| 3        | Hypodontia             |
| 4        | Tooth Discoloration    |
| 5        | Ulcer                  |

---

## 🧠 Model Training

- **Base model**: `yolov8n.pt`  
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- **Dataset**: 6-class oral disease dataset, manually labeled in YOLO format
- **Image size**: `320x320`
- **Augmentation**: Extensive class-based augmentation using `albumentations`
- **Split**: 75% train, 15% valid, 10% test

---

### 📦 Dataset Preparation

- Balanced dataset using under/over-sampling strategies
- Class-specific augmentations applied for minority classes (e.g., `hypodontia`, `ulcer`)
- Final dataset distribution:

| Class             | Instances |
|------------------|-----------|
| Caries           | 10000     |
| Hypodontia       | 5000      |
| Ulcer            | 5000      |
| Calculus         | 6000      |
| Gingivitis       | 6000      |
| Discoloration    | 4000      |

> 📈 Class distribution is visualized in `balanced_distribution.png`

---

### 🧪 Evaluation Results

- **Test Accuracy**: ~92.6%
- **Mean Precision / Recall / F1-score**: ~0.92
- **YOLO Validation Metrics**: Precision, Recall, mAP@50, Confusion Matrix

![Confusion Matrix](runs/detect/val19/confusion_matrix.png)

---

### 📸 Sample Predictions

![Detection Example](runs/detect/val19/val_batch0.jpg)

---

## 🛠️ Inference & Deployment

- The best YOLOv8 model was exported to:
  - ✅ ONNX: `best.onnx`
  - ✅ TensorFlow: `tf_model/`
- Ready for integration with **FastAPI**, **Flutter**, or **TensorFlow Lite**

```python
# FastAPI endpoint receives an image and returns YOLOv8 predictions
results = model.predict(image_path)
results[0].show()  # Show prediction
