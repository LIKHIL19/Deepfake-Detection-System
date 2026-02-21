# 🎭 Deepfake Detection using ResNet-18

A complete deep learning pipeline to detect whether an image or video is **REAL** or **FAKE** using face-level analysis and a ResNet-18 classifier.

---

## 🚀Project Overview

This project builds an end-to-end deepfake detection system:

- 🎥 Extract frames from videos  
- 😀 Detect & crop faces  
- 🧠 Train a deep CNN (ResNet-18)  
- 📊 Evaluate with multiple performance metrics  
- 🌐 Deploy with Streamlit UI  

The model performs both **frame-level** and **video-level** classification.

---

## 🛠 Tech Stack

- **Python**
- **PyTorch**
- **ResNet-18 (Transfer Learning)**
- **OpenCV**
- **Pillow**
- **Pandas**
- **Streamlit**
- **Matplotlib**

**GPU Used:** NVIDIA RTX 3050 (4GB)

---

## 📂 Project Structure

```text
Deepfake-project/
│
├── scripts/
│   ├── extract_frames.py
│   ├── precrop_faces_v2.py
│   ├── train_resnet18.py
│   ├── eval_test.py
│   ├── predict.py
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│
├── preprocessed/
│   ├── faces/
│   ├── videos/
│
├── best_model.pth
├── checkpoint.pth
└── app.py
```

---

## 🔄 Workflow

### 1️⃣ Frame Extraction
Videos are converted into frames using OpenCV.

### 2️⃣ Face Cropping
Each frame is processed to detect and crop only the facial region.

### 3️⃣ Dataset Preparation
CSV files define:
- Image path  
- Label (0 = REAL, 1 = FAKE)

### 4️⃣ Model Training
- Pretrained **ResNet-18**
- Final classification head (binary)
- Weighted sampling for class balance
- Cosine learning rate schedule
- Best model saved automatically

### 5️⃣ Evaluation
Model evaluated on unseen test data using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### 6️⃣ Inference
- Predict on single image
- Predict on full video (frame averaging)
- Streamlit UI for drag-and-drop testing

---

## 📊 Final Performance

### 🎯 Frame-Level
- Accuracy: **82%**
- F1 Score: **0.83**
- ROC-AUC: **0.87**

### 🎬 Video-Level
- Accuracy: **84%**
- F1 Score: **0.84**
- ROC-AUC: **0.89**
- PR-AUC: **0.90**

Video-level performance improves reliability by averaging frame predictions.
