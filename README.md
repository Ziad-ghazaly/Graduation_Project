# 🦷 Dental Disease Detection with YOLOv8 and FastAPI

This project presents a real-time mobile dental diagnosis system for detecting six types of oral diseases using YOLOv8. It is integrated with a Flutter-based mobile frontend and a FastAPI backend, and supports lightweight deployment for mobile use cases.

--- 

## App Overview

The app enables dental disease detection directly from the camera or gallery, returning bounding boxes and disease class labels. YOLOv8n was chosen for its balance of speed and accuracy, making it suitable for edge devices.

 [Watch the app demo on LinkedIn](https://www.linkedin.com/posts/ziad-ghazaly-a6828b283_artificialintelligence-deeplearning-yolov8-activity-7345865207572848640-FmqN?utm_source=share&utm_medium=member_desktop&rcm=ACoAAETrIDAB2VqJsrYbntfFRs1iXmofFvHGC1U)

 ### Model detection Screenshots

<p align="center">
  <img src="screenshots/sample_model_predictions.jpeg" width="250"/>
  <img src="screenshots/sample_model_predictions (2).jpeg" width="250"/>
  <img src="screenshots/sample_model_predictions (3).jpeg" width="250"/>
</p>


---

##  Detected Dental Conditions

| Class ID | Disease Name            |
|----------|-------------------------|
| 0        | Calculus                |
| 1        | Caries                  |
| 2        | Gingivitis              |
| 3        | Hypodontia              |
| 4        | Tooth Discoloration     |
| 5        | Ulcer                   |

---

##  Model Architecture

- **Base Model:** YOLOv8n (Ultralytics)
- **Training Backend:** PyTorch via Ultralytics wrapper
- **Deployment:** Exported to ONNX and TensorFlow for further compatibility

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

---

## 📁 Dataset

- **Dataset Size:** ~30K labeled images
- **Classes:** 6 dental disease types
- **Label Format:** YOLO format (.txt)
- **Augmentation:** Custom augmentation pipeline per class (Albumentations)

---

##  Training Configuration

| Parameter        | Value           |
|------------------|-----------------|
| Image Size       | 512x512         |
| Epochs           | 30              |
| Batch Size       | 16              |
| Optimizer        | Adam            |
| Augmentations    | Class-specific  |
| Hardware         | CUDA (GPU)      |

---

##  Evaluation

- Evaluation was performed using validation and test splits (15% + 10%)
- Confusion matrix and precision-recall curves were generated

---

##  Export and Deployment

Model was exported to:
- **ONNX:** For future inference with hardware-optimized runtimes
- **TensorFlow SavedModel:** For mobile TensorFlow Lite conversion (optional)

```python
# Export to TensorFlow
onnx_model = onnx.load("best.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")
```

---

##  Backend API

- **Framework:** FastAPI
- **Purpose:** Host trained model and expose endpoints for mobile requests
- **Endpoints:**
  - `/predict` – Accepts image input and returns predictions
  - `/health` – Basic API health check

---

##  Features

- Real-time detection on mobile
- Lightweight model (YOLOv8n)
- Balanced dataset through augmentation
- Training visualization (loss curves, class distribution)
- Exportable for ONNX/TensorFlow

---

##  Future Improvements

- Integrate Firebase or Supabase for storing cases
- Add segmentation for detailed dental mapping
- Improve model performance via more balanced data

---

##  How to Run

1. Clone the repo
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   yolo task=detect mode=train data=data.yaml model=yolov8n.pt epochs=30 imgsz=320
   ```
4. Start FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```
5. Use your Flutter app to make requests to the backend.

---

##  Related Files

- `data.yaml`: Class definitions
- `runs/`: Training results and metrics
- `weights/`: Trained models (best.pt, ONNX, etc.)
- `images/`: Visualizations and screenshots

---

## License

This project is for academic and research purposes only.
