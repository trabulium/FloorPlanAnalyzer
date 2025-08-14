# FloorPlanAnalyzer

A comprehensive floor plan analysis tool with multiple detection methods including deep learning models. Automatically detects rooms, doors, windows, and other architectural elements from floor plan images.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Multiple Detection Methods:**
  - Traditional computer vision (watershed segmentation)
  - YOLOv8 AI-based detection
  - CubiCasa5K neural network (state-of-the-art)
  - Hybrid approach combining YOLO and OCR

- **Advanced Capabilities:**
  - Automatic room type classification
  - OCR-based text detection for room labels
  - Scale calibration for accurate measurements
  - Room area calculation in square meters
  - Interactive room editing and resizing
  - SVG export for vector graphics

- **User-Friendly GUI:**
  - Dynamic controls based on selected method
  - Real-time visualization with color-coded rooms
  - Editable room list with type modification
  - Zoom functionality for detailed inspection

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for text detection)

#### Install Tesseract:

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FloorPlanAnalyzer.git
cd FloorPlanAnalyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the GUI application:
```bash
python src/main.py
```

### Detection Methods

#### 1. CubiCasa5K (Recommended)
Best overall accuracy using a neural network trained on 5000+ floor plans.
- Adjust minimum room area (100-10000 px²)
- Control smoothing iterations

#### 2. Traditional (Watershed)
Classic computer vision approach using watershed segmentation.
- Configure gap closing for doors
- Adjust wall thickness detection

#### 3. YOLOv8
AI-based detection for architectural elements.
- Detects walls, doors, windows, stairs, columns
- Adjustable confidence threshold (10-80%)

#### 4. Hybrid (YOLO + OCR)
Combines YOLO's element detection with OCR text recognition.
- Best for floor plans with clear text labels

### Workflow

1. **Load Image**: Click "Load Floor Plan" to select your image
2. **Calibrate Scale** (optional): Use known dimensions for accurate measurements
3. **Select Method**: Choose detection method based on your floor plan
4. **Detect**: Click "Detect Rooms & Doors" to analyze
5. **Edit** (optional): 
   - Select rooms from the list
   - Edit room types or resize dimensions
   - Remove unwanted detections
6. **Export**: Save as SVG or JSON

## Project Structure

```
FloorPlanAnalyzer/
├── src/
│   ├── main.py           # Main GUI application
│   ├── detector.py        # Core detection algorithms
│   └── yolo_detector.py   # YOLO integration
├── models/
│   ├── cubicasa/          # CubiCasa5K model files
│   └── yolo/              # YOLOv8 model weights
├── examples/              # Sample floor plans
├── docs/                  # Documentation
└── tests/                 # Unit tests
```

## Acknowledgments

This project builds upon several excellent open-source projects:

### CubiCasa5K
- **Repository**: [CubiCasa/floortrans](https://github.com/CubiCasa/CubiCasa5k)
- **Paper**: "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis"
- **License**: MIT License
- The CubiCasa5K model provides state-of-the-art floor plan segmentation

### YOLOv8 Floor Plan Detection
- **Base Model**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Training Dataset**: Custom floor plan dataset
- **License**: AGPL-3.0 License for Ultralytics

### Dependencies
- OpenCV for image processing
- PyTorch for deep learning
- Tesseract for OCR functionality
- PyQt5 for the graphical interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **CubiCasa5K Model**: MIT License
- **Ultralytics YOLOv8**: AGPL-3.0 License
- **PyQt5**: GPL/Commercial License

Please ensure compliance with all third-party licenses when using this software.

## Citation

If you use this tool in your research, please consider citing:

```bibtex
@software{floorplananalyzer2024,
  title = {FloorPlanAnalyzer: Multi-Method Floor Plan Analysis Tool},
  year = {2024},
  url = {https://github.com/yourusername/FloorPlanAnalyzer}
}
```

For the CubiCasa5K model:
```bibtex
@article{kalervo2019cubicasa5k,
  title={CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis},
  author={Kalervo, Ahti and Ylioinas, Juha and Häikiö, Markku and Karhu, Antti and Kannala, Juho},
  journal={arXiv preprint arXiv:1904.01920},
  year={2019}
}
```

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Roadmap

- [ ] Add support for more file formats (DWG, DXF)
- [ ] Implement automatic dimension extraction
- [ ] Add 3D visualization capabilities
- [ ] Support for multi-story buildings
- [ ] Batch processing for multiple floor plans
- [ ] API for integration with other tools