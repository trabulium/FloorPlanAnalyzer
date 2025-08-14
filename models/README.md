# Model Files

This directory contains the pre-trained models used for floor plan analysis.

## Directory Structure

```
models/
├── cubicasa/     # CubiCasa5K neural network model
│   └── model/    # Place model.pkl file here
└── yolo/         # YOLOv8 floor plan detection model
    └── best.pt   # Pre-trained YOLO weights
```

## Obtaining Models

### CubiCasa5K Model
Download from the [CubiCasa5K repository](https://github.com/CubiCasa/CubiCasa5k)

### YOLOv8 Model
The YOLO model should be trained on floor plan datasets. You can:
1. Use a pre-trained model if available
2. Train your own using the Ultralytics framework

## Note
Model files are not included in the repository due to their size. Please download them separately.