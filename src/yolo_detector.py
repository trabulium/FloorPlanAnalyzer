#!/usr/bin/env python3
"""
YOLOv8 Floor Plan Detector
Uses pre-trained YOLOv8 model to detect walls, doors, windows, etc.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents a detected object in the floor plan"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: float


class YOLOFloorPlanDetector:
    """
    YOLOv8-based floor plan object detection
    Detects walls, doors, windows, columns, stairs, etc.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize with YOLOv8 model
        
        Args:
            model_path: Path to model weights. If None, uses default path.
        """
        if model_path is None:
            # Use model from models directory
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, 'models', 'yolo', 'best.pt')
        
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"YOLOv8 model loaded from {self.model_path}")
                
                # Get class names from model
                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                    print(f"Available classes: {list(self.class_names.values())}")
                else:
                    self.class_names = {}
            else:
                print(f"Model file not found: {self.model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None
    
    def detect(self, image_path: str, confidence: float = 0.4, 
               filter_classes: List[str] = None) -> Dict:
        """
        Detect objects in floor plan image
        
        Args:
            image_path: Path to floor plan image
            confidence: Minimum confidence threshold (0-1)
            filter_classes: List of class names to detect (None = all)
        
        Returns:
            Dictionary with detected objects by category
        """
        if self.model is None:
            print("Model not loaded")
            return {}
        
        # Run inference
        results = self.model(image_path, conf=confidence)
        
        # Process results
        detected_objects = {
            'walls': [],
            'doors': [],
            'windows': [],
            'columns': [],
            'stairs': [],
            'other': []
        }
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get class label
                class_id = int(box.cls[0])
                label = self.class_names.get(class_id, f"Class_{class_id}")
                
                # Filter by class if specified
                if filter_classes and label not in filter_classes:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and area
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                # Create detected object
                obj = DetectedObject(
                    label=label,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    area=area
                )
                
                # Categorize by type
                label_lower = label.lower()
                if 'wall' in label_lower:
                    detected_objects['walls'].append(obj)
                elif 'door' in label_lower:
                    detected_objects['doors'].append(obj)
                elif 'window' in label_lower:
                    detected_objects['windows'].append(obj)
                elif 'column' in label_lower:
                    detected_objects['columns'].append(obj)
                elif 'stair' in label_lower:
                    detected_objects['stairs'].append(obj)
                else:
                    detected_objects['other'].append(obj)
        
        # Print summary
        print(f"\nDetection Summary:")
        for category, objects in detected_objects.items():
            if objects:
                print(f"  {category}: {len(objects)}")
        
        return detected_objects
    
    def visualize_detections(self, image_path: str, detections: Dict, 
                             output_path: str = "yolo_detection_result.png"):
        """
        Visualize detected objects on the image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Color scheme for different object types
        colors = {
            'walls': (0, 0, 255),      # Red
            'doors': (0, 255, 0),      # Green
            'windows': (255, 128, 0),   # Blue
            'columns': (128, 0, 128),  # Purple
            'stairs': (255, 255, 0),   # Cyan
            'other': (128, 128, 128)   # Gray
        }
        
        # Draw detections
        for category, objects in detections.items():
            color = colors.get(category, (128, 128, 128))
            
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{obj.label} ({obj.confidence:.2f})"
                cv2.putText(img, label_text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save result
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to {output_path}")
        
        return img
    
    def extract_room_boundaries(self, detections: Dict) -> List[np.ndarray]:
        """
        Extract room boundaries from detected walls
        This creates polygons representing rooms
        """
        walls = detections.get('walls', [])
        
        if not walls:
            return []
        
        # TODO: Implement algorithm to connect walls into room polygons
        # This would involve:
        # 1. Finding wall intersections
        # 2. Creating a graph of connected walls
        # 3. Finding closed loops that represent rooms
        
        room_boundaries = []
        
        return room_boundaries


def main():
    """Test the YOLOv8 detector"""
    detector = YOLOFloorPlanDetector()
    
    # Test on sample image
    test_image = "granny_flat_eucalypt.png"
    if Path(test_image).exists():
        print(f"\nProcessing {test_image}...")
        
        # Detect objects
        detections = detector.detect(test_image, confidence=0.3)
        
        # Visualize results
        detector.visualize_detections(test_image, detections)
        
        # Print detailed results
        print("\nDetailed Results:")
        for category, objects in detections.items():
            if objects:
                print(f"\n{category.upper()}:")
                for obj in objects[:5]:  # Show first 5 of each type
                    print(f"  - {obj.label} at {obj.center}, conf: {obj.confidence:.2f}")
    else:
        print(f"Test image {test_image} not found")


if __name__ == "__main__":
    main()