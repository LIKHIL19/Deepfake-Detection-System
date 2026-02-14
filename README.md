Deepfake Detection using ResNet-18 (Face-Level Classification)
Overview

This project is a complete deep learning–based deepfake detection system built using ResNet-18 and PyTorch.

The system detects whether a given image or video is REAL (authentic) or FAKE (manipulated) by analyzing facial features extracted from video frames.

Unlike traditional machine learning projects, this system performs:

Video processing

Frame extraction

Face detection & cropping

Deep CNN-based feature extraction

Binary classification

Frame-level & Video-level evaluation

Web-based inference interface (Streamlit)

The project was built step-by-step from raw video data to final deployment-ready prediction UI.

Project Objective

To build a robust deepfake detection system that:

Works on both images and videos

Uses face-level analysis for better accuracy

Achieves strong generalization performance

Provides clear evaluation metrics

Includes a usable frontend interface

Key Features
Data Processing Pipeline

Video → Frame extraction

Face cropping using OpenCV

Dataset verification and validation

Train / Validation / Test split generation

Model Training

ResNet-18 (pretrained on ImageNet)

Logistic Regression classification head (final linear layer + Softmax)

WeightedRandomSampler for class balancing

Cosine Annealing Learning Rate Scheduler

Label Smoothing Cross Entropy

Mixed Precision Training (AMP)

Checkpoint saving & resume support

Evaluation

Frame-level evaluation

Video-level evaluation (average logits across frames)

Confusion Matrix

Accuracy, Precision, Recall, F1 Score

ROC-AUC & PR-AUC

Log Loss & Brier Score

Inference

Single image prediction

Single video prediction

Automatic frame sampling

Face detection during inference

Streamlit-based interactive UI

Tech Stack
Component	Technology
Programming Language	Python
Deep Learning Framework	PyTorch
Model Architecture	ResNet-18
Computer Vision	OpenCV
Image Handling	Pillow (PIL)
Data Handling	Pandas
Visualization	Matplotlib
UI	Streamlit
Hardware	NVIDIA RTX 3050 4GB GPU
