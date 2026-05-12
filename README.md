# Behavioral detection-based home security using AI

## Overview

This is a smart-home security system with AI, capable of multiple **facial recognition**, **behaviour recognition**, and **suspicious activity** alarm monitoring in real-time, and integrating it into a comprehensive real-time surveillance system. The system should be able to detect known persons, unknown persons, and suspicious activity – all within a web interface created using **streamlit**.

## Project Structure
 
```
streamlit_app/
├── app.py                  # Main application entry point
├── behavior_app.py         # Suspicious activity detection module
├── inference.py            # Behaviour model inference logic
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── yolov8n.pt              # YOLOv8 model weights
├── models/                 # Behaviour detection models
├── known_faces/            # Stored face embeddings (local only)
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## Dataset Used

| Dataset | Author | Year | Link |
|---|---|---|---|
| Pins Face Recognition | Buraк | n.d. | [Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) |

The Pins Face Recognition dataset was used to train and evaluate the face recognition component of this system.

## Installation
 
### Prerequisites
- Python 3.12
- Git
- Microsoft C++ Build Tools (for InsightFace)

## Requirements

Full list of dependencies in `requirements.txt`.:
 
```
streamlit==1.57.0
streamlit-webrtc==0.64.6
streamlit-autorefresh==1.0.1
av==16.1.0
tornado>=6.2
opencv-python-headless==4.11.0.86
numpy==1.26.4
pillow==12.2.0
insightface==0.7.3
onnxruntime==1.25.1
torch==2.5.1
transformers==4.46.3
ultralytics==8.3.40
mediapipe==0.10.21
```
 
---
