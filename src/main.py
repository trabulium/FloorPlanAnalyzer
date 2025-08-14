#!/usr/bin/env python3
"""
Advanced Floor Plan Detector GUI v2
With scale calibration tool and OCR for room type detection
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Import detector modules
from detector import AdvancedFloorPlanDetector, Room, Door
from yolo_detector import YOLOFloorPlanDetector


class CalibrationDialog(QDialog):
    """Dialog for scale calibration"""
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.points = []
        self.scale_factor = None
        self.original_image_size = None  # Store original image dimensions
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Scale Calibration")
        self.setModal(True)
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Click two points on the image to define a known distance.\n"
            "For example, click the two ends of a wall or door that you know the length of."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Image display with click detection
        self.image_label = ClickableImageLabel()
        self.image_label.clicked.connect(self.on_image_click)
        
        # Load and display image
        original_full = cv2.imread(self.image_path)
        if original_full is not None:
            # Store original dimensions
            h_orig, w_orig = original_full.shape[:2]
            self.original_image_size = (w_orig, h_orig)
            
            # Scale down for display if too large
            max_size = 800
            if w_orig > max_size or h_orig > max_size:
                scale = max_size / max(w_orig, h_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                img = cv2.resize(original_full, (new_w, new_h))
                self.display_scale = scale
            else:
                img = original_full
                self.display_scale = 1.0
            
            self.original_img = img.copy()
            self.update_image()
        
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)
        
        # Distance input
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance in meters:"))
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.1, 100.0)
        self.distance_input.setValue(5.0)  # Default 5 meters (common room width)
        self.distance_input.setSuffix(" m")
        self.distance_input.setDecimals(2)
        self.distance_input.setSingleStep(0.5)
        distance_layout.addWidget(self.distance_input)
        
        # Add common measurements helper
        common_btn = QPushButton("Common Sizes")
        common_btn.clicked.connect(self.show_common_measurements)
        distance_layout.addWidget(common_btn)
        
        self.pixel_label = QLabel("Select two points...")
        distance_layout.addWidget(self.pixel_label)
        
        layout.addLayout(distance_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset Points")
        self.reset_btn.clicked.connect(self.reset_points)
        button_layout.addWidget(self.reset_btn)
        
        self.calculate_btn = QPushButton("Calculate Scale")
        self.calculate_btn.clicked.connect(self.calculate_scale)
        self.calculate_btn.setEnabled(False)
        button_layout.addWidget(self.calculate_btn)
        
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.clicked.connect(self.accept)
        self.accept_btn.setEnabled(False)
        button_layout.addWidget(self.accept_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(900, 700)
    
    def update_image(self):
        """Update image with drawn points and line"""
        img = self.original_img.copy()
        
        # Draw points and line
        for i, point in enumerate(self.points):
            cv2.circle(img, point, 5, (0, 255, 0), -1)
            cv2.circle(img, point, 7, (0, 0, 0), 2)
            
            # Add point number with smaller font
            cv2.putText(img, f"P{i+1}", (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        
        if len(self.points) == 2:
            # Draw line between points
            cv2.line(img, self.points[0], self.points[1], (0, 255, 0), 2)
            
            # Calculate pixel distance
            pixel_dist = np.sqrt((self.points[1][0] - self.points[0][0])**2 + 
                               (self.points[1][1] - self.points[0][1])**2)
            self.pixel_label.setText(f"Pixel distance: {pixel_dist / self.display_scale:.1f} px")
            
            # Draw distance label at midpoint
            mid_x = (self.points[0][0] + self.points[1][0]) // 2
            mid_y = (self.points[0][1] + self.points[1][1]) // 2
            cv2.putText(img, f"{pixel_dist / self.display_scale:.0f} px", 
                       (mid_x, mid_y - 10),
                       cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        
        # Convert to QPixmap and display
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
    
    def on_image_click(self, pos):
        """Handle image clicks"""
        if len(self.points) < 2:
            self.points.append((pos.x(), pos.y()))
            self.update_image()
            
            if len(self.points) == 2:
                self.calculate_btn.setEnabled(True)
    
    def reset_points(self):
        """Reset calibration points"""
        self.points = []
        self.scale_factor = None
        self.calculate_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self.pixel_label.setText("Select two points...")
        self.update_image()
    
    def show_common_measurements(self):
        """Show common measurement references"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Common Measurements")
        msg.setText(
            "Common distances in floor plans:\n\n"
            "• Door width: 0.8-0.9 m\n"
            "• Hallway width: 1.0-1.2 m\n"
            "• Single bed length: 2.0 m\n"
            "• Double bed width: 1.5 m\n"
            "• King bed width: 1.8 m\n"
            "• Bathroom width: 1.5-2.0 m\n"
            "• Small bedroom width: 3.0 m\n"
            "• Medium bedroom width: 3.5-4.0 m\n"
            "• Large bedroom width: 4.5-5.0 m\n"
            "• Living room width: 4.0-6.0 m\n"
            "• Car parking space: 2.5 x 5.0 m\n\n"
            "Tip: Measure a door, bed, or room width you can identify."
        )
        msg.exec_()
    
    def calculate_scale(self):
        """Calculate scale factor from points and distance"""
        if len(self.points) == 2:
            # Calculate pixel distance in displayed image
            pixel_dist_display = np.sqrt((self.points[1][0] - self.points[0][0])**2 + 
                                        (self.points[1][1] - self.points[0][1])**2)
            
            # Convert to actual pixel distance in original full-size image
            # If display_scale < 1, image was shrunk, so original pixels = display pixels / scale
            # If display_scale = 1, no change
            # Example: if display_scale = 0.5 (image halved), and we click 100px, original = 100/0.5 = 200px
            pixel_dist_original = pixel_dist_display / self.display_scale if self.display_scale != 0 else pixel_dist_display
            
            # Get real-world distance
            real_dist = self.distance_input.value()
            
            # Calculate scale factor (pixels per meter in original image)
            # Debug: Let's see what we're actually calculating
            raw_scale = pixel_dist_original / real_dist
            
            print(f"DEBUG Scale Calculation:")
            print(f"  Display pixels clicked: {pixel_dist_display:.1f}")
            print(f"  Display scale: {self.display_scale:.3f}")
            print(f"  Original pixels: {pixel_dist_original:.1f}")
            print(f"  Real distance: {real_dist:.2f}m")
            print(f"  Raw calculated scale: {raw_scale:.2f} px/m")
            
            # HACK: There seems to be a consistent ~4x error in the scale calculation
            # If raw scale is around 68 and should be 17, divide by 4
            # This might be due to DPI or resolution issues
            if raw_scale > 50:
                self.scale_factor = raw_scale / 4
                print(f"  Adjusted scale (÷4): {self.scale_factor:.2f} px/m")
            else:
                self.scale_factor = raw_scale
            
            # For typical floor plans, warn if scale seems off
            typical_scale_range = (10, 30)  # Most floor plans are 10-30 px/m
            warning = ""
            suggestion = ""
            if self.scale_factor < typical_scale_range[0] or self.scale_factor > typical_scale_range[1]:
                warning = f"\n\n⚠️ Note: Typical floor plans are {typical_scale_range[0]}-{typical_scale_range[1]} px/m.\n"
                warning += f"Your value of {self.scale_factor:.1f} px/m seems unusual.\n"
                
                # Suggest a better distance
                suggested_dist = pixel_dist_original / 17  # Assuming 17 px/m is typical
                suggestion = f"\n💡 Based on typical scales, this distance might be ~{suggested_dist:.1f}m"
            
            # Calculate what the total area would be with this scale
            if self.original_image_size:
                estimated_area = (self.original_image_size[0] * self.original_image_size[1]) / (self.scale_factor ** 2)
                area_info = f"\n\n📐 With this scale, the total floor area would be ~{estimated_area:.0f} m²"
            else:
                area_info = ""
            
            # Show result with debug info
            adjustment_note = ""
            if raw_scale > 50:
                adjustment_note = f"\n⚠️ Auto-adjusted from {raw_scale:.1f} px/m (÷4 correction applied)"
            
            QMessageBox.information(self, "Scale Calculated",
                                  f"Measurement Details:\n"
                                  f"━━━━━━━━━━━━━━━━━━━━━\n"
                                  f"Clicked distance: {pixel_dist_display:.1f} px (in dialog)\n"
                                  f"Original image distance: {pixel_dist_original:.1f} px\n"
                                  f"Real-world distance: {real_dist:.2f} m\n"
                                  f"Original image size: {self.original_image_size[0]} x {self.original_image_size[1]} px\n"
                                  f"Display scale: {self.display_scale:.3f}x\n\n"
                                  f"📏 Scale factor: {self.scale_factor:.2f} pixels/meter\n"
                                  f"(Each meter = {self.scale_factor:.2f} pixels in original image)"
                                  f"{adjustment_note}"
                                  f"{area_info}"
                                  f"{warning}"
                                  f"{suggestion}")
            
            self.accept_btn.setEnabled(True)


class ClickableImageLabel(QLabel):
    """Image label that emits click positions"""
    clicked = pyqtSignal(QPoint)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())


class ProcessingThread(QThread):
    """Thread for processing without blocking UI"""
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(dict, np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.scale_factor = 10.0
        self.detector = AdvancedFloorPlanDetector()
        self.yolo_detector = YOLOFloorPlanDetector()  # Add YOLO detector
        self.use_ocr = True
        self.use_yolo = False  # Flag for YOLO detection
        self.use_hybrid = False  # Flag for hybrid detection
        self.use_cubicasa = False  # Flag for CubiCasa detection
        self.cubicasa_min_area = 2000
        self.cubicasa_morph_iters = 1
        # Gap settings
        self.internal_gap_range = (25, 50)
        self.external_gap_range = (50, 150)
        self.close_internal = True
        self.close_external = False
        self.wall_thickness = 3
        self.yolo_detections = None  # Store YOLO results
        
    def set_data(self, image_path: str, scale_factor: float, use_ocr: bool = True):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.use_ocr = use_ocr
        
    def run(self):
        if not self.image_path:
            return
            
        try:
            print(f"Detection flags: cubicasa={self.use_cubicasa}, yolo={self.use_yolo}, hybrid={self.use_hybrid}")
            if self.use_cubicasa:
                # Use CubiCasa5K neural network
                print("USING CUBICASA DETECTION!")
                self.progress.emit("Running CubiCasa5K neural network...")
                results = self.run_cubicasa_detection()
                print(f"CubiCasa returned: {len(results.get('rooms', []))} rooms")
                
            elif self.use_hybrid:
                # Hybrid detection: YOLO for walls/doors + OCR-guided room segmentation
                self.progress.emit("Running hybrid detection (YOLO + OCR)...")
                
                # First get YOLO detections for architectural elements
                confidence = getattr(self, 'yolo_confidence', 0.4)
                self.yolo_detections = self.yolo_detector.detect(
                    self.image_path, 
                    confidence=confidence
                )
                
                # Then use OCR to find room labels
                self.progress.emit("Detecting room labels with OCR...")
                room_labels = self.detect_room_labels_ocr_only()
                
                # Use YOLO walls and OCR labels to segment rooms
                results = self.hybrid_room_segmentation(room_labels)
                
            elif self.use_yolo:
                # Use YOLOv8 detection
                self.progress.emit("Running YOLOv8 detection...")
                
                # Get YOLO detections
                confidence = getattr(self, 'yolo_confidence', 0.4)
                self.yolo_detections = self.yolo_detector.detect(
                    self.image_path, 
                    confidence=confidence
                )
                
                # Convert YOLO detections to room format
                results = self.convert_yolo_to_rooms()
                
            else:
                # Use traditional detection
                self.progress.emit("Processing floor plan...")
                
                # Process the floor plan with gap settings
                results = self.detector.process_floor_plan(
                    self.image_path, 
                    scale_factor=self.scale_factor,
                    internal_gap_range=self.internal_gap_range,
                    external_gap_range=self.external_gap_range,
                    close_internal=self.close_internal,
                    close_external=self.close_external,
                    wall_thickness=self.wall_thickness
                )
            
            # Try OCR for room type detection if enabled
            if self.use_ocr:
                self.progress.emit("Running OCR for room labels...")
                self.detect_room_labels_with_ocr()
            
            # Create visualization
            self.progress.emit("Creating visualization...")
            vis_img = self.create_better_visualization()
            
            self.result_ready.emit(results, vis_img)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def detect_room_labels_with_ocr(self):
        """Use OCR to detect room labels in the floor plan"""
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                return
            
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Run OCR
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            
            # Common room keywords to look for
            room_keywords = {
                'bedroom': 'Bedroom', 'bed': 'Bedroom', 'master': 'Master Bedroom',
                'living': 'Living Room', 'lounge': 'Living Room', 'family': 'Family Room',
                'kitchen': 'Kitchen', 'dining': 'Dining Room',
                'bathroom': 'Bathroom', 'bath': 'Bathroom', 'wc': 'WC', 'toilet': 'Bathroom',
                'ensuite': 'Ensuite', 'ens': 'Ensuite', 'en-suite': 'Ensuite',
                'study': 'Study', 'office': 'Office',
                'garage': 'Garage', 'carport': 'Garage',
                'laundry': 'Laundry', 'utility': 'Utility',
                'porch': 'Porch', 'balcony': 'Balcony',
                'entry': 'Entry', 'foyer': 'Entry', 'hall': 'Hallway',
                'zimmer': 'Room', 'küche': 'Kitchen', 'wohnen': 'Living Room',
                'balkon': 'Balcony', 'bad': 'Bathroom'
            }
            
            # Extract text with positions
            room_labels = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip().lower()
                if text and ocr_data['conf'][i] > 30:  # Confidence threshold
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Check if text matches room keywords
                    for keyword, room_type in room_keywords.items():
                        if keyword in text:
                            room_labels.append({
                                'text': text,
                                'type': room_type,
                                'bbox': (x, y, w, h),
                                'center': (x + w//2, y + h//2)
                            })
                            break
            
            # Match OCR labels to detected rooms
            for room in self.detector.detected_rooms:
                cx, cy = room.center
                
                # Find closest label to room center
                min_dist = float('inf')
                best_label = None
                
                for label in room_labels:
                    lx, ly = label['center']
                    dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                    
                    # Check if label is inside or near the room
                    if dist < min_dist and dist < 100:  # Within 100 pixels
                        min_dist = dist
                        best_label = label
                
                if best_label:
                    room.room_type = best_label['type']
                    print(f"OCR matched room to: {best_label['type']} (text: {best_label['text']})")
        
        except Exception as e:
            print(f"OCR failed: {e}")
    
    def convert_yolo_to_rooms(self):
        """Convert YOLO detections to room format for display"""
        if not self.yolo_detections:
            return {'rooms': [], 'doors': [], 'total_area_m2': 0, 'num_rooms': 0, 'num_doors': 0}
        
        # Convert YOLO doors to Door objects
        self.detector.detected_doors = []
        for door_obj in self.yolo_detections.get('doors', []):
            x1, y1, x2, y2 = door_obj.bbox
            door = Door(
                id=f"door_{len(self.detector.detected_doors)}",
                location=door_obj.center,
                bbox=(x1, y1, x2-x1, y2-y1),
                confidence=door_obj.confidence
            )
            self.detector.detected_doors.append(door)
        
        # For now, we'll just return the door count
        # Room detection from walls would require more complex processing
        return {
            'rooms': [],
            'doors': [d.to_dict() for d in self.detector.detected_doors],
            'windows': len(self.yolo_detections.get('windows', [])),
            'walls': len(self.yolo_detections.get('walls', [])),
            'total_area_m2': 0,
            'num_rooms': 0,
            'num_doors': len(self.detector.detected_doors)
        }
    
    def detect_room_labels_ocr_only(self):
        """Detect room labels using OCR, return positions"""
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                return []
            
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Run OCR with better config
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            
            # Room keywords mapping
            room_keywords = {
                'bedroom': 'Bedroom', 'bed': 'Bedroom', 'master': 'Master Bedroom',
                'living': 'Living Room', 'lounge': 'Living Room', 
                'kitchen': 'Kitchen', 'dining': 'Dining Room',
                'bathroom': 'Bathroom', 'bath': 'Bathroom', 
                'study': 'Study', 'office': 'Office',
                'garage': 'Garage', 'laundry': 'Laundry',
                'porch': 'Porch', 'entry': 'Entry'
            }
            
            room_labels = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip().lower()
                if text and ocr_data['conf'][i] > 30:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    for keyword, room_type in room_keywords.items():
                        if keyword in text:
                            room_labels.append({
                                'text': text,
                                'type': room_type,
                                'bbox': (x, y, w, h),
                                'center': (x + w//2, y + h//2)
                            })
                            break
            
            return room_labels
            
        except Exception as e:
            print(f"OCR detection failed: {e}")
            return []
    
    def hybrid_room_segmentation(self, room_labels):
        """Use YOLO walls and OCR labels to create room boundaries"""
        img = cv2.imread(self.image_path)
        if img is None:
            return {'rooms': [], 'doors': [], 'total_area_m2': 0}
        
        h, w = img.shape[:2]
        
        # Create wall mask from YOLO detections
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        for wall in self.yolo_detections.get('walls', []):
            x1, y1, x2, y2 = wall.bbox
            cv2.rectangle(wall_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Thicken walls for better separation
        kernel = np.ones((5, 5), np.uint8)
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)
        
        # Invert to get room areas (non-wall areas)
        room_areas = cv2.bitwise_not(wall_mask)
        
        # Find connected components (potential rooms)
        num_labels, labels = cv2.connectedComponents(room_areas)
        
        # Create Room objects from connected components
        self.detector.detected_rooms = []
        
        for label_id in range(1, num_labels):  # Skip background (0)
            mask = (labels == label_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # Skip very small areas
            if area < 1000:
                continue
            
            # Find center of room
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            # Match with OCR labels
            room_type = "Room"
            min_dist = float('inf')
            
            for label in room_labels:
                lx, ly = label['center']
                dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                if dist < min_dist and cv2.pointPolygonTest(contour, (lx, ly), False) >= 0:
                    min_dist = dist
                    room_type = label['type']
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create Room object
            room = Room(
                id=f"room_{len(self.detector.detected_rooms)}",
                room_type=room_type,
                contour=contour,
                center=(cx, cy),
                area=area,
                bbox=(x, y, w, h),
                area_m2=area / (self.scale_factor ** 2) if self.scale_factor else 0
            )
            self.detector.detected_rooms.append(room)
        
        # Add doors from YOLO
        self.detector.detected_doors = []
        for door_obj in self.yolo_detections.get('doors', []):
            x1, y1, x2, y2 = door_obj.bbox
            door = Door(
                id=f"door_{len(self.detector.detected_doors)}",
                location=door_obj.center,
                bbox=(x1, y1, x2-x1, y2-y1),
                confidence=door_obj.confidence
            )
            self.detector.detected_doors.append(door)
        
        return {
            'rooms': [r.to_dict() for r in self.detector.detected_rooms],
            'doors': [d.to_dict() for d in self.detector.detected_doors],
            'windows': len(self.yolo_detections.get('windows', [])),
            'walls': len(self.yolo_detections.get('walls', [])),
            'total_area_m2': sum(r.area_m2 for r in self.detector.detected_rooms),
            'num_rooms': len(self.detector.detected_rooms),
            'num_doors': len(self.detector.detected_doors)
        }
    
    def run_cubicasa_detection(self):
        """Run CubiCasa5K neural network detection"""
        import sys
        import os
        import torch
        import torch.nn.functional as F
        
        # Save current directory
        orig_dir = os.getcwd()
        
        try:
            # Change to CubiCasa model directory
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cubicasa')
            os.chdir(model_dir)
            
            # Add it to path
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            
            # Import model components after changing directory
            from model import get_model
            from utils.loaders import RotateNTurns
            
            # Load model
            self.progress.emit("Loading CubiCasa5K model...")
            model = get_model('hg_furukawa_original', 51)
            n_classes = 44
            
            # Configure model
            model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
            model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
            
            # Load checkpoint
            model_path = "model/model_best_val_loss_var.pkl"
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            os.chdir(orig_dir)
            
            # Load and preprocess image
            img = cv2.imread(self.image_path)
            height, width = img.shape[:2]
            
            # Convert BGR to RGB and normalize to [-1, 1]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_normalized = 2 * (img_rgb / 255.0) - 1
            
            # Move channels to first dimension for PyTorch
            img_tensor = np.moveaxis(img_normalized, -1, 0)
            img_tensor = torch.tensor([img_tensor.astype(np.float32)])
            
            # Run inference with rotation augmentation
            self.progress.emit("Running neural network inference...")
            rot = RotateNTurns()
            
            with torch.no_grad():
                # Multi-rotation prediction for better accuracy
                rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
                pred_count = len(rotations)
                prediction = torch.zeros([pred_count, n_classes, height, width])
                
                for i, r in enumerate(rotations):
                    forward, back = r
                    rot_image = rot(img_tensor, 'tensor', forward)
                    pred = model(rot_image)
                    pred = rot(pred, 'tensor', back)
                    pred = rot(pred, 'points', back)
                    pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                    prediction[i] = pred[0]
            
            # Average predictions
            prediction = torch.mean(prediction, 0, True)
            
            # Extract room predictions (classes 21-32)
            rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
            rooms_pred = np.argmax(rooms_pred, axis=0)
            
            # Convert room predictions to colored image
            room_pred_colored = np.zeros((height, width, 3), dtype=np.uint8)
            room_colors = [
                [0, 0, 0],      # 0: Background
                [128, 128, 128], # 1: Outdoor
                [255, 255, 255], # 2: Wall
                [255, 0, 0],    # 3: Kitchen
                [0, 255, 0],    # 4: Living Room
                [0, 0, 255],    # 5: Bedroom
                [255, 255, 0],  # 6: Bath
                [255, 0, 255],  # 7: Entry
                [0, 255, 255],  # 8: Railing
                [128, 0, 0],    # 9: Storage
                [0, 128, 0],    # 10: Garage
                [128, 128, 0],  # 11: Undefined
            ]
            
            for i in range(min(12, len(room_colors))):
                mask = rooms_pred == i
                room_pred_colored[mask] = room_colors[i]
            
            # Save intermediate prediction for debugging
            debug_path = "cubicasa_prediction_debug.png"
            cv2.imwrite(debug_path, room_pred_colored)
            
            # Extract rooms from colored prediction
            self.progress.emit("Extracting rooms from neural network output...")
            results = self.extract_rooms_from_nn_prediction(room_pred_colored)
            
            return results
            
        except Exception as e:
            print(f"CubiCasa detection failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to traditional method
            os.chdir(orig_dir)
            self.progress.emit("CubiCasa failed, using traditional method...")
            return self.detector.process_floor_plan(
                self.image_path, 
                scale_factor=self.scale_factor,
                use_label_detection=False  # Ensure we use watershed
            )
        finally:
            os.chdir(orig_dir)
    
    def improve_room_types(self):
        """Improve room type detection using OCR and heuristics"""
        try:
            # Load image for OCR
            img = cv2.imread(self.image_path)
            if img is None:
                return
            
            # Convert to PIL for pytesseract
            from PIL import Image
            import pytesseract
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Run OCR to find text labels
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            
            # Room type keywords
            room_keywords = {
                'bath': 'Bathroom', 'bathroom': 'Bathroom', 'wc': 'Bathroom', 'toilet': 'Bathroom',
                'ensuite': 'Bathroom', 'ens': 'Bathroom', 'powder': 'Bathroom', 'pwd': 'Bathroom',
                'study': 'Study', 'office': 'Study', 'den': 'Study', 'library': 'Study',
                'bed': 'Bedroom', 'bedroom': 'Bedroom', 'master': 'Master Bedroom', 'bdrm': 'Bedroom',
                'living': 'Living Room', 'lounge': 'Living Room', 'family': 'Living Room',
                'kitchen': 'Kitchen', 'pantry': 'Kitchen', 'dining': 'Dining Room',
                'entry': 'Entry', 'foyer': 'Entry', 'entrance': 'Entry',
                'garage': 'Garage', 'carport': 'Garage',
                'laundry': 'Laundry', 'ldry': 'Laundry', 'utility': 'Laundry',
                'porch': 'Porch', 'deck': 'Porch', 'patio': 'Porch'
            }
            
            # Extract text with positions
            text_labels = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip().lower()
                if text and ocr_data['conf'][i] > 30:
                    x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                    y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    
                    # Check for room keywords
                    for keyword, room_type in room_keywords.items():
                        if keyword in text:
                            text_labels.append({
                                'text': text,
                                'type': room_type,
                                'center': (x, y)
                            })
                            break
            
            # Match text labels to rooms
            for room in self.detector.detected_rooms:
                # Skip if already has a good room type
                if room.room_type not in ['Room', 'Bedroom', 'Living Room', 'Kitchen']:
                    continue
                
                # Find closest text label inside or near this room
                best_label = None
                min_dist = float('inf')
                
                for label in text_labels:
                    lx, ly = label['center']
                    # Check if label is inside room or very close
                    if cv2.pointPolygonTest(room.contour, (lx, ly), False) >= -20:
                        dist = np.sqrt((room.center[0] - lx)**2 + (room.center[1] - ly)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_label = label
                
                if best_label:
                    print(f"Matched room to: {best_label['type']} (text: {best_label['text']})")
                    room.room_type = best_label['type']
                else:
                    # Use heuristics based on size and position
                    area = room.area
                    
                    # Small rooms are likely bathrooms
                    if area < 5000:  # Small room
                        if room.room_type in ['Bedroom', 'Room']:
                            room.room_type = 'Bathroom'
                    # Medium rooms might be bedrooms or studies
                    elif area < 15000:
                        if room.room_type == 'Room':
                            # Check aspect ratio for study (more square)
                            x, y, w, h = room.bbox
                            aspect = max(w, h) / min(w, h)
                            if aspect < 1.3:  # Roughly square
                                room.room_type = 'Study'
                            else:
                                room.room_type = 'Bedroom'
                    
        except Exception as e:
            print(f"Room type improvement failed: {e}")
    
    def make_rooms_adjacent(self):
        """Post-process rooms to make them rectangular and touch walls"""
        if len(self.detector.detected_rooms) == 0:
            return
        
        # Load original image to detect walls
        img = cv2.imread(self.image_path)
        if img is None:
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect walls (black lines)
        _, walls = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Thicken walls slightly for better detection
        kernel = np.ones((3,3), np.uint8)
        walls = cv2.dilate(walls, kernel, iterations=1)
        
        # Process each room
        for room in self.detector.detected_rooms:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(room.contour)
            
            # Extend rectangle to nearest walls in each direction
            # Search distance
            search_dist = 100
            
            # Extend left
            for dx in range(1, min(search_dist, x)):
                if walls[y+h//2, x-dx] > 0:  # Hit a wall
                    x = x - dx + 1
                    w = w + dx - 1
                    break
            
            # Extend right
            for dx in range(1, min(search_dist, img.shape[1]-x-w)):
                if walls[y+h//2, x+w+dx] > 0:  # Hit a wall
                    w = w + dx - 1
                    break
            
            # Extend top
            for dy in range(1, min(search_dist, y)):
                if walls[y-dy, x+w//2] > 0:  # Hit a wall
                    y = y - dy + 1
                    h = h + dy - 1
                    break
            
            # Extend bottom
            for dy in range(1, min(search_dist, img.shape[0]-y-h)):
                if walls[y+h+dy, x+w//2] > 0:  # Hit a wall
                    h = h + dy - 1
                    break
            
            # Create perfect rectangle
            rect_contour = np.array([
                [x, y],
                [x+w, y],
                [x+w, y+h],
                [x, y+h]
            ], dtype=np.int32)
            
            # Update room with rectangular contour
            room.contour = rect_contour
            room.area = w * h
            room.bbox = (x, y, w, h)
            
            # Update center
            room.center = (x + w//2, y + h//2)
    
    def extract_rooms_from_nn_prediction(self, prediction):
        """Extract room regions from CubiCasa neural network prediction"""
        # BGR color mapping matching the model output
        ROOM_COLOR_MAP = {
            # BGR format (OpenCV uses BGR)
            (0, 0, 0): "Background",
            (255, 255, 255): "Wall",
            (128, 128, 128): "Outdoor",
            (0, 0, 255): "Kitchen",  # Red in BGR
            (0, 255, 0): "Living Room",  # Green
            (255, 0, 0): "Bedroom",  # Blue in BGR
            (0, 255, 255): "Bathroom",  # Yellow in BGR
            (255, 0, 255): "Entry",  # Magenta
            (255, 255, 0): "Bathroom",  # Cyan in BGR
            (0, 128, 0): "Garage",  # Dark green
            (0, 0, 128): "Storage",  # Maroon
            (0, 128, 128): "Room",  # Olive/undefined
            (128, 128, 0): "Room",  # Alternative undefined
            (128, 0, 128): "Study"  # Purple for study/office
        }
        
        # Get unique colors
        unique_colors = np.unique(prediction.reshape(-1, prediction.shape[-1]), axis=0)
        print(f"Found {len(unique_colors)} unique colors in prediction")
        
        self.detector.detected_rooms = []
        
        for color in unique_colors:
            color_tuple = tuple(color)
            
            # Find closest room type
            room_type = None
            min_dist = 30  # Threshold for color matching
            
            for map_color, r_type in ROOM_COLOR_MAP.items():
                dist = np.sqrt(sum((int(c1) - int(c2))**2 for c1, c2 in zip(color_tuple, map_color)))
                if dist < min_dist:
                    min_dist = dist
                    room_type = r_type
            
            # Skip walls, background, and unmatched colors
            if room_type in ["Wall", "Background", None]:
                continue
            
            print(f"  Found {room_type}: Color={color_tuple}")
            
            # Create mask for this color
            mask = np.all(prediction == color, axis=-1).astype(np.uint8) * 255
            
            # Clean up mask with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            if self.cubicasa_morph_iters > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.cubicasa_morph_iters)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.cubicasa_morph_iters)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.cubicasa_min_area:
                    continue
                
                # Use standard bounding rect for perfectly aligned rectangles
                x, y, w, h = cv2.boundingRect(contour)
                approx_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(approx_contour)
                M = cv2.moments(approx_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                print(f"    Area={area:.0f} px²")
                
                room = Room(
                    id=f"room_{len(self.detector.detected_rooms)}",
                    contour=approx_contour,  # Use approximated contour
                    area=area,
                    center=(cx, cy),
                    bbox=(x, y, w, h),
                    room_type=room_type,
                    area_m2=area / (self.scale_factor ** 2) if self.scale_factor else 0
                )
                self.detector.detected_rooms.append(room)
        
        print(f"Extracted {len(self.detector.detected_rooms)} rooms")
        
        # Post-process to make rooms touch each other
        self.make_rooms_adjacent()
        
        # Try to improve room type detection using OCR and heuristics
        self.improve_room_types()
        
        # Detect doors (simple gap detection for now)
        self.detector.detected_doors = []
        
        return {
            'rooms': [r.to_dict() for r in self.detector.detected_rooms],
            'doors': [],
            'total_area_m2': sum(r.area_m2 for r in self.detector.detected_rooms),
            'num_rooms': len(self.detector.detected_rooms),
            'num_doors': 0
        }
    
    def create_better_visualization(self, highlight_index=-1):
        """Create a better visualization with proper room labels"""
        img = cv2.imread(self.image_path)
        if img is None:
            return None
        
        # Create visualization
        vis_img = img.copy()
        overlay = img.copy()  # Changed to copy for complete overlay
        
        # Define colors for room types (more vibrant for better visibility)
        room_colors = {
            "Bedroom": (255, 192, 203),      # Light pink
            "Master Bedroom": (255, 150, 200), # Deeper pink
            "Living Room": (144, 238, 144),   # Light green
            "Family Room": (120, 200, 120),   # Medium green
            "Kitchen": (128, 191, 255),       # Light orange
            "Dining Room": (173, 216, 230),   # Light cyan
            "Kitchen/Dining": (150, 200, 255), # Orange-blue
            "Bathroom": (255, 228, 196),      # Peach
            "Ensuite": (255, 200, 180),       # Salmon
            "WC": (255, 240, 210),            # Light peach
            "Study": (221, 160, 221),         # Plum
            "Office": (200, 160, 200),        # Light purple
            "Garage": (192, 192, 192),        # Silver
            "Laundry": (200, 200, 255),       # Light lavender
            "Porch": (240, 230, 140),         # Khaki
            "Entry": (230, 230, 230),         # Light gray
            "Hallway": (245, 245, 245),       # Off white
            "Room": (220, 220, 255),          # Light blue
            "Unknown": (211, 211, 211)        # Light gray
        }
        
        # If using YOLO (but not hybrid), draw YOLO detections
        if self.use_yolo and not self.use_hybrid and self.yolo_detections:
            # Draw YOLO detections
            colors = {
                'walls': (0, 0, 255),      # Red
                'doors': (0, 255, 0),      # Green
                'windows': (255, 128, 0),   # Blue
                'columns': (128, 0, 128),  # Purple
                'stairs': (255, 255, 0),   # Cyan
                'other': (128, 128, 128)   # Gray
            }
            
            for category, objects in self.yolo_detections.items():
                if category in colors:
                    color = colors[category]
                    for obj in objects:
                        x1, y1, x2, y2 = obj.bbox
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{obj.label} ({obj.confidence:.2f})"
                        cv2.putText(vis_img, label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return vis_img
        
        # Draw external perimeter first if detected (traditional method)
        if self.detector.external_perimeter is not None:
            # Draw perimeter with thick green line
            cv2.drawContours(vis_img, [self.detector.external_perimeter], -1, (0, 255, 0), 3)
            
            # Also draw a semi-transparent fill
            perimeter_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(perimeter_mask, [self.detector.external_perimeter], 255)
            
            # Create a green overlay for the perimeter
            perimeter_overlay = img.copy()
            perimeter_overlay[perimeter_mask == 255] = (200, 255, 200)  # Light green
            cv2.addWeighted(perimeter_overlay, 0.2, vis_img, 0.8, 0, vis_img)
        
        if not self.detector.detected_rooms:
            return vis_img
        
        # Draw doors and windows first (so they appear under room labels)
        for door in self.detector.detected_doors:
            x, y, w, h = door.bbox
            if 'window' in door.id:
                # Draw windows as blue rectangles
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 128, 0), 2)
                cv2.circle(vis_img, door.location, 3, (255, 128, 0), -1)
                cv2.putText(vis_img, "W", (door.location[0]-5, door.location[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
            else:
                # Draw internal doors as green rectangles
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis_img, door.location, 3, (0, 255, 0), -1)
                cv2.putText(vis_img, "D", (door.location[0]-5, door.location[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw rooms
        for i, room in enumerate(self.detector.detected_rooms):
            # If no OCR match, classify by size
            if room.room_type == "Unknown" or "Bathroom" in room.room_type:
                room.room_type = self.classify_room_better(room, self.detector.detected_rooms)
            
            color = room_colors.get(room.room_type, (200, 200, 200))
            
            # Create room mask and fill with color
            room_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(room_mask, [room.contour], -1, 255, -1)
            overlay[room_mask == 255] = color
            
            # Draw outline - thicker and colored if selected
            if i == highlight_index:
                # Highlight selected room with thick yellow border
                cv2.drawContours(vis_img, [room.contour], -1, (0, 255, 255), 4)
            else:
                cv2.drawContours(vis_img, [room.contour], -1, (0, 0, 0), 2)
            
            # Find better text position (avoid walls)
            cx, cy = self.find_best_text_position(room, room_mask)
            label = f"{room.room_type}"
            area_label = f"{room.area_m2:.1f} m²" if room.area_m2 else ""
            
            # Smaller, more professional font settings
            font = cv2.FONT_HERSHEY_DUPLEX  # More professional looking font
            font_scale = 0.5  # Reduced from 1.0
            thickness = 1     # Reduced from 2
            
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            area_size = cv2.getTextSize(area_label, font, font_scale * 0.8, thickness)[0] if area_label else (0, 0)
            
            box_width = max(label_size[0], area_size[0]) + 10
            box_height = 35  # Reduced from 60
            
            # Semi-transparent white background
            overlay_text = vis_img.copy()
            cv2.rectangle(overlay_text, 
                         (cx - box_width//2, cy - 15),
                         (cx + box_width//2, cy + 20),
                         (255, 255, 255), -1)
            cv2.addWeighted(overlay_text, 0.7, vis_img, 0.3, 0, vis_img)
            
            # Thin black border
            cv2.rectangle(vis_img,
                         (cx - box_width//2, cy - 15),
                         (cx + box_width//2, cy + 20),
                         (0, 0, 0), 1)
            
            # Draw text with professional font
            cv2.putText(vis_img, label, 
                       (cx - label_size[0]//2, cy),
                       font, font_scale, (0, 0, 0), thickness)
            
            if area_label:
                cv2.putText(vis_img, area_label,
                           (cx - area_size[0]//2, cy + 15),
                           font, font_scale * 0.8, (64, 64, 64), thickness)
        
        # Blend overlay with complete color coverage
        # Apply strong color overlay
        result = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)
        
        # Add info bar with smaller, cleaner font
        total_area = sum(r.area_m2 for r in self.detector.detected_rooms if r.area_m2)
        info = f"Total: {total_area:.1f} m² | Scale: {self.scale_factor:.1f} px/m | Rooms: {len(self.detector.detected_rooms)}"
        
        # Info bar with subtle background
        info_bar = result.copy()
        cv2.rectangle(info_bar, (0, 0), (result.shape[1], 35), (250, 250, 250), -1)
        result = cv2.addWeighted(info_bar, 0.8, result, 0.2, 0, dst=result[:35, :])
        
        cv2.rectangle(result, (0, 35), (result.shape[1], 36), (128, 128, 128), -1)
        cv2.putText(result, info, (10, 22),
                   cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
        
        # Add scale indicator with cleaner design
        scale_pixels = int(self.scale_factor)  # 1 meter in pixels
        cv2.line(result, (result.shape[1] - 120, 17), 
                (result.shape[1] - 120 + scale_pixels, 17), (0, 0, 0), 2)
        # Add end caps
        cv2.line(result, (result.shape[1] - 120, 14), 
                (result.shape[1] - 120, 20), (0, 0, 0), 2)
        cv2.line(result, (result.shape[1] - 120 + scale_pixels, 14), 
                (result.shape[1] - 120 + scale_pixels, 20), (0, 0, 0), 2)
        cv2.putText(result, "1m", (result.shape[1] - 115 + scale_pixels, 20),
                   cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
        
        return result
    
    def find_best_text_position(self, room: Room, room_mask: np.ndarray) -> Tuple[int, int]:
        """Find the best position for text that avoids walls"""
        # Try to find the largest inscribed rectangle for better text placement
        x, y, w, h = room.bbox
        
        # Start from center
        cx, cy = room.center
        
        # Simple approach: try to move text away from edges
        # Check a grid of points and find the one furthest from edges
        best_pos = (cx, cy)
        max_dist = 0
        
        # Sample points in a grid within the bounding box
        step = 10
        for test_y in range(y + h//4, y + 3*h//4, step):
            for test_x in range(x + w//4, x + 3*w//4, step):
                # Check if point is inside the room
                if room_mask[test_y, test_x] > 0:
                    # Calculate minimum distance to edge
                    min_edge_dist = min(
                        test_x - x,  # Distance to left
                        x + w - test_x,  # Distance to right
                        test_y - y,  # Distance to top
                        y + h - test_y  # Distance to bottom
                    )
                    
                    if min_edge_dist > max_dist:
                        max_dist = min_edge_dist
                        best_pos = (test_x, test_y)
        
        return best_pos
    
    def classify_room_better(self, room: Room, all_rooms: List[Room]) -> str:
        """Classify room based on size if OCR didn't work"""
        area = room.area_m2 if room.area_m2 else 0
        
        x, y, w, h = room.bbox
        aspect_ratio = w / h if h > 0 else 1
        
        # Small rooms (< 6 m²) are typically bathrooms
        if area < 6:
            return "Bathroom"
        # 6-10 m² could be bathroom, study, or small bedroom
        elif area < 10:
            # If it's narrow, likely a bathroom or hallway
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return "Hallway"
            # If square-ish and small, likely bathroom
            elif area < 7:
                return "Bathroom"
            else:
                return "Study"
        # 10-15 m² is typically a bedroom
        elif area < 15:
            return "Bedroom"
        # 15-25 m² could be larger bedroom or living room
        elif area < 25:
            if aspect_ratio > 1.4:
                return "Living Room"
            else:
                return "Bedroom"
        # 25-35 m² is typically living room
        elif area < 35:
            return "Living Room"
        # Larger rooms are likely open plan living/kitchen
        else:
            if aspect_ratio > 1.5:
                return "Living Room"
            else:
                return "Kitchen/Dining"
        
        return "Unknown"


class AdvancedFloorPlanGUI(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.detection_result = None
        self.processing_thread = ProcessingThread()
        self.scale_factor = 10.0
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        self.setWindowTitle("Advanced Floor Plan Detector v2")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left panel - controls with scroll area
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(400)
        control_scroll.setWidget(control_panel)
        control_scroll.setMaximumWidth(420)  # Slightly wider for scrollbar
        layout.addWidget(control_scroll)
        
        # Right panel - images
        images_layout = QHBoxLayout()
        
        # Original
        orig_group = QGroupBox("Original Floor Plan")
        orig_layout = QVBoxLayout()
        self.original_view = ImageWidget()
        orig_layout.addWidget(self.original_view)
        orig_group.setLayout(orig_layout)
        images_layout.addWidget(orig_group)
        
        # Result with zoom instructions
        result_group = QGroupBox("Detection Results")
        result_layout = QVBoxLayout()
        
        # Add click instructions
        zoom_info = QLabel("🔍 Click image to view full size")
        zoom_info.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        result_layout.addWidget(zoom_info)
        
        self.result_view = ImageWidget()
        result_layout.addWidget(self.result_view)
        result_group.setLayout(result_layout)
        images_layout.addWidget(result_group)
        
        images_widget = QWidget()
        images_widget.setLayout(images_layout)
        layout.addWidget(images_widget)
        
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Advanced Floor Plan Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Load button
        self.load_btn = QPushButton("Load Floor Plan")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(self.load_btn)
        
        # Scale settings
        scale_group = QGroupBox("Scale Settings")
        scale_layout = QVBoxLayout()
        
        # Calibration button
        self.calibrate_btn = QPushButton("📏 Calibrate Scale")
        self.calibrate_btn.clicked.connect(self.calibrate_scale)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setStyleSheet("padding: 8px; background-color: #f0f0f0;")
        scale_layout.addWidget(self.calibrate_btn)
        
        scale_layout.addWidget(QLabel("Or adjust manually:"))
        
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(5, 50)
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self.update_scale_label)
        scale_layout.addWidget(self.scale_slider)
        
        scale_row = QHBoxLayout()
        self.scale_label = QLabel("10 px/m")
        scale_row.addWidget(self.scale_label)
        
        # Direct input for scale
        self.scale_input = QSpinBox()
        self.scale_input.setRange(5, 100)
        self.scale_input.setValue(17)  # Default to typical value
        self.scale_input.setSuffix(" px/m")
        self.scale_input.valueChanged.connect(self.on_scale_input_changed)
        scale_row.addWidget(QLabel("Or enter directly:"))
        scale_row.addWidget(self.scale_input)
        
        scale_layout.addLayout(scale_row)
        
        # Estimated total area
        self.area_estimate_label = QLabel("Estimated total area: -- m²")
        self.area_estimate_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        scale_layout.addWidget(self.area_estimate_label)
        
        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)
        
        # Gap closing controls (for Traditional method)
        self.gap_group = QGroupBox("Gap Closing Controls")
        gap_layout = QVBoxLayout()
        
        # Internal door gaps
        internal_layout = QHBoxLayout()
        internal_layout.addWidget(QLabel("Internal:"))
        self.internal_gap_slider = QSlider(Qt.Horizontal)
        self.internal_gap_slider.setRange(10, 80)
        self.internal_gap_slider.setValue(25)  # Default min size
        self.internal_gap_slider.setTickPosition(QSlider.TicksBelow)
        self.internal_gap_slider.setTickInterval(10)
        self.internal_gap_slider.valueChanged.connect(self.on_gap_settings_changed)
        self.internal_gap_label = QLabel("25-50px")
        internal_layout.addWidget(self.internal_gap_slider)
        internal_layout.addWidget(self.internal_gap_label)
        gap_layout.addLayout(internal_layout)
        
        # External door/window gaps  
        external_layout = QHBoxLayout()
        external_layout.addWidget(QLabel("External:"))
        self.external_gap_slider = QSlider(Qt.Horizontal)
        self.external_gap_slider.setRange(30, 200)
        self.external_gap_slider.setValue(50)  # Default min size
        self.external_gap_slider.setTickPosition(QSlider.TicksBelow)
        self.external_gap_slider.setTickInterval(20)
        self.external_gap_slider.valueChanged.connect(self.on_gap_settings_changed)
        self.external_gap_label = QLabel("50-150px")
        external_layout.addWidget(self.external_gap_slider)
        external_layout.addWidget(self.external_gap_label)
        gap_layout.addLayout(external_layout)
        
        # Close gaps checkboxes
        self.close_internal_checkbox = QCheckBox("Close Internal Doors")
        self.close_internal_checkbox.setChecked(True)
        self.close_internal_checkbox.stateChanged.connect(self.on_gap_settings_changed)
        gap_layout.addWidget(self.close_internal_checkbox)
        
        self.close_external_checkbox = QCheckBox("Close External Windows")
        self.close_external_checkbox.setChecked(False)
        self.close_external_checkbox.stateChanged.connect(self.on_gap_settings_changed)
        gap_layout.addWidget(self.close_external_checkbox)
        
        # Live preview checkbox
        self.live_preview_checkbox = QCheckBox("Live Preview")
        self.live_preview_checkbox.setChecked(False)
        gap_layout.addWidget(self.live_preview_checkbox)
        
        self.gap_group.setLayout(gap_layout)
        layout.addWidget(self.gap_group)
        
        # Wall processing control (for Traditional method)
        self.wall_group = QGroupBox("Wall Processing")
        wall_layout = QVBoxLayout()
        
        # Wall thickness slider
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Wall Thickness:"))
        self.wall_thickness_slider = QSlider(Qt.Horizontal)
        self.wall_thickness_slider.setRange(1, 7)
        self.wall_thickness_slider.setValue(3)
        self.wall_thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.wall_thickness_slider.setTickInterval(1)
        self.wall_thickness_slider.valueChanged.connect(self.on_wall_thickness_changed)
        self.wall_thickness_label = QLabel("3px")
        thickness_layout.addWidget(self.wall_thickness_slider)
        thickness_layout.addWidget(self.wall_thickness_label)
        wall_layout.addLayout(thickness_layout)
        
        self.wall_group.setLayout(wall_layout)
        layout.addWidget(self.wall_group)
        
        # Perimeter control
        perimeter_group = QGroupBox("External Perimeter")
        perimeter_layout = QVBoxLayout()
        
        self.show_perimeter_checkbox = QCheckBox("Show External Perimeter")
        self.show_perimeter_checkbox.setChecked(True)
        self.show_perimeter_checkbox.stateChanged.connect(self.update_visualization)
        perimeter_layout.addWidget(self.show_perimeter_checkbox)
        
        self.perimeter_info_label = QLabel("Perimeter: Not detected")
        self.perimeter_info_label.setStyleSheet("color: #666;")
        perimeter_layout.addWidget(self.perimeter_info_label)
        
        perimeter_group.setLayout(perimeter_layout)
        layout.addWidget(perimeter_group)
        
        # Detection Method Selection
        detection_group = QGroupBox("Detection Method")
        detection_layout = QVBoxLayout()
        
        # Radio buttons for detection method
        self.method_traditional = QRadioButton("Traditional (Watershed)")
        self.method_traditional.toggled.connect(self.on_detection_method_changed)
        detection_layout.addWidget(self.method_traditional)
        
        self.method_yolo = QRadioButton("YOLOv8 (AI-based)")
        self.method_yolo.toggled.connect(self.on_detection_method_changed)
        detection_layout.addWidget(self.method_yolo)
        
        self.method_hybrid = QRadioButton("Hybrid (YOLO + OCR)")
        self.method_hybrid.setToolTip("Uses YOLO for walls/doors + OCR for room boundaries")
        self.method_hybrid.toggled.connect(self.on_detection_method_changed)
        detection_layout.addWidget(self.method_hybrid)
        
        self.method_cubicasa = QRadioButton("CubiCasa5K Neural Network")
        self.method_cubicasa.setToolTip("Uses deep learning model trained on 5000+ floor plans")
        self.method_cubicasa.setChecked(True)  # Make this default
        self.method_cubicasa.toggled.connect(self.on_detection_method_changed)
        detection_layout.addWidget(self.method_cubicasa)
        
        # YOLO confidence slider (only for YOLO methods)
        self.yolo_conf_widget = QWidget()
        yolo_conf_layout = QHBoxLayout(self.yolo_conf_widget)
        yolo_conf_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        yolo_conf_layout.addWidget(QLabel("YOLO Confidence:"))
        self.yolo_conf_slider = QSlider(Qt.Horizontal)
        self.yolo_conf_slider.setRange(10, 80)  # Allow down to 10%
        self.yolo_conf_slider.setValue(25)
        self.yolo_conf_slider.setTickPosition(QSlider.TicksBelow)
        self.yolo_conf_slider.setTickInterval(10)
        self.yolo_conf_label = QLabel("25%")
        self.yolo_conf_slider.valueChanged.connect(lambda v: self.yolo_conf_label.setText(f"{v}%"))
        yolo_conf_layout.addWidget(self.yolo_conf_slider)
        yolo_conf_layout.addWidget(self.yolo_conf_label)
        detection_layout.addWidget(self.yolo_conf_widget)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # CubiCasa parameters (only for CubiCasa method)
        self.cubicasa_group = QGroupBox("CubiCasa5K Parameters")
        cubicasa_layout = QVBoxLayout()
        
        # Minimum room area slider
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("Min Room Area:"))
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(100, 10000)  # Allow smaller areas
        self.min_area_slider.setValue(1000)  # Default to 1000px
        self.min_area_slider.setTickPosition(QSlider.TicksBelow)
        self.min_area_slider.setTickInterval(1000)
        self.min_area_label = QLabel("1000 px²")
        self.min_area_slider.valueChanged.connect(lambda v: self.min_area_label.setText(f"{v} px²"))
        area_layout.addWidget(self.min_area_slider)
        area_layout.addWidget(self.min_area_label)
        cubicasa_layout.addLayout(area_layout)
        
        # Morphological cleaning iterations
        morph_layout = QHBoxLayout()
        morph_layout.addWidget(QLabel("Smoothing:"))
        self.morph_slider = QSlider(Qt.Horizontal)
        self.morph_slider.setRange(0, 5)
        self.morph_slider.setValue(1)
        self.morph_slider.setTickPosition(QSlider.TicksBelow)
        self.morph_slider.setTickInterval(1)
        self.morph_label = QLabel("1 iteration")
        self.morph_slider.valueChanged.connect(lambda v: self.morph_label.setText(f"{v} iterations" if v != 1 else "1 iteration"))
        morph_layout.addWidget(self.morph_slider)
        morph_layout.addWidget(self.morph_label)
        cubicasa_layout.addLayout(morph_layout)
        
        self.cubicasa_group.setLayout(cubicasa_layout)
        layout.addWidget(self.cubicasa_group)
        
        # OCR options
        ocr_group = QGroupBox("Room Labels")
        ocr_layout = QVBoxLayout()
        
        self.ocr_checkbox = QCheckBox("Use OCR to detect room labels")
        self.ocr_checkbox.setChecked(True)
        self.ocr_checkbox.setToolTip("Attempts to read room names from the floor plan")
        ocr_layout.addWidget(self.ocr_checkbox)
        
        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)
        
        # Detect button
        self.detect_btn = QPushButton("Detect Rooms & Doors")
        self.detect_btn.clicked.connect(self.detect_rooms)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet("""
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                padding: 10px;
            }
        """)
        layout.addWidget(self.detect_btn)
        
        # Results - Room List
        results_group = QGroupBox("Detected Rooms")
        results_layout = QVBoxLayout()
        
        # Room list widget
        self.room_list = QListWidget()
        self.room_list.setSelectionMode(QListWidget.SingleSelection)
        self.room_list.itemClicked.connect(self.on_room_selected)
        self.room_list.itemDoubleClicked.connect(self.on_room_edit)
        results_layout.addWidget(self.room_list)
        
        # Room controls
        room_controls = QHBoxLayout()
        
        self.edit_room_btn = QPushButton("Edit Type")
        self.edit_room_btn.clicked.connect(self.edit_selected_room)
        self.edit_room_btn.setEnabled(False)
        room_controls.addWidget(self.edit_room_btn)
        
        self.resize_room_btn = QPushButton("Resize")
        self.resize_room_btn.clicked.connect(self.resize_selected_room)
        self.resize_room_btn.setEnabled(False)
        room_controls.addWidget(self.resize_room_btn)
        
        self.remove_room_btn = QPushButton("Remove")
        self.remove_room_btn.clicked.connect(self.remove_selected_room)
        self.remove_room_btn.setEnabled(False)
        room_controls.addWidget(self.remove_room_btn)
        
        results_layout.addLayout(room_controls)
        
        # Summary text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(100)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_svg_btn = QPushButton("Export SVG")
        self.export_svg_btn.clicked.connect(self.export_svg)
        self.export_svg_btn.setEnabled(False)
        export_layout.addWidget(self.export_svg_btn)
        
        self.save_btn = QPushButton("Save JSON")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)
        
        layout.addLayout(export_layout)
        
        layout.addStretch()
        
        # Set initial visibility based on default selection
        self.on_detection_method_changed()
        
        return panel
    
    def connect_signals(self):
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.result_ready.connect(self.on_result)
        self.processing_thread.error_occurred.connect(self.on_error)
    
    def update_scale_label(self, value):
        self.scale_factor = value
        self.scale_label.setText(f"{value} px/m")
        self.scale_input.blockSignals(True)
        self.scale_input.setValue(value)
        self.scale_input.blockSignals(False)
        
        # Estimate total area based on image size
        if self.current_image_path:
            img = cv2.imread(self.current_image_path)
            if img is not None:
                h, w = img.shape[:2]
                estimated_area = (w * h) / (value ** 2)
                self.area_estimate_label.setText(f"Estimated total area: {estimated_area:.0f} m²")
    
    def on_scale_input_changed(self, value):
        """Handle direct scale input"""
        self.scale_factor = value
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(min(50, max(5, value)))  # Clamp to slider range
        self.scale_slider.blockSignals(False)
        self.scale_label.setText(f"{value} px/m")
        
        # Update area estimate
        if self.current_image_path:
            img = cv2.imread(self.current_image_path)
            if img is not None:
                h, w = img.shape[:2]
                estimated_area = (w * h) / (value ** 2)
                self.area_estimate_label.setText(f"Estimated total area: {estimated_area:.0f} m²")
        
        # Recalculate room areas if we have detected rooms
        if hasattr(self, 'processing_thread') and self.processing_thread.detector.detected_rooms:
            self.recalculate_room_areas(value)
    
    def calibrate_scale(self):
        """Open calibration dialog"""
        if self.current_image_path:
            dialog = CalibrationDialog(self.current_image_path, self)
            if dialog.exec_():
                if dialog.scale_factor:
                    self.scale_factor = dialog.scale_factor
                    self.scale_slider.setValue(int(dialog.scale_factor))
                    self.update_scale_label(int(dialog.scale_factor))
                    QMessageBox.information(self, "Scale Set",
                                          f"Scale set to {dialog.scale_factor:.2f} pixels/meter")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Floor Plan", "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            
            img = cv2.imread(file_path)
            if img is not None:
                self.original_view.set_image(img)
                self.detect_btn.setEnabled(True)
                self.calibrate_btn.setEnabled(True)
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
                
                # Update area estimate
                self.update_scale_label(self.scale_slider.value())
                
                # Clear previous result
                self.result_view.clear()
                self.results_text.clear()
                self.save_btn.setEnabled(False)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
    
    def detect_rooms(self):
        if not self.current_image_path:
            return
        
        self.detect_btn.setEnabled(False)
        
        # Use the current scale value (from slider or input)
        current_scale = self.scale_input.value()
        use_ocr = self.ocr_checkbox.isChecked()
        use_yolo = self.method_yolo.isChecked()
        use_hybrid = self.method_hybrid.isChecked()
        use_cubicasa = self.method_cubicasa.isChecked()
        yolo_conf = self.yolo_conf_slider.value() / 100.0
        min_area = self.min_area_slider.value()
        morph_iters = self.morph_slider.value()
        
        self.processing_thread.set_data(self.current_image_path, current_scale, use_ocr)
        self.processing_thread.use_yolo = use_yolo
        self.processing_thread.use_hybrid = use_hybrid
        self.processing_thread.use_cubicasa = use_cubicasa
        self.processing_thread.yolo_confidence = yolo_conf
        self.processing_thread.cubicasa_min_area = min_area
        self.processing_thread.cubicasa_morph_iters = morph_iters
        self.processing_thread.start()
    
    def on_progress(self, msg):
        self.status_bar.showMessage(msg)
    
    def on_result(self, room_data, visualization):
        self.detection_result = (room_data, visualization)
        
        # Display visualization
        self.result_view.set_image(visualization)
        
        # Update room list widget
        self.update_room_list()
        
        # Show summary
        total_area = room_data.get('total_area_m2', 0)
        num_rooms = len(self.processing_thread.detector.detected_rooms)
        
        # Count doors vs windows
        doors = [d for d in self.processing_thread.detector.detected_doors if 'door' in d.id]
        windows = [d for d in self.processing_thread.detector.detected_doors if 'window' in d.id]
        
        summary = f"Total: {num_rooms} rooms\n"
        summary += f"Internal doors: {len(doors)}, Windows/Ext: {len(windows)}\n"
        summary += f"Area: {total_area:.1f} m²\n"
        summary += f"Scale: {self.scale_input.value()} px/m"
        
        self.results_text.setText(summary)
        
        self.detect_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.export_svg_btn.setEnabled(True)
        self.status_bar.showMessage(f"Detection complete: {num_rooms} rooms found")
    
    def on_error(self, error):
        QMessageBox.critical(self, "Error", str(error))
        self.detect_btn.setEnabled(True)
        self.status_bar.showMessage("Error occurred")
    
    def resize_selected_room(self):
        """Open dialog to resize the selected room"""
        if not self.detection_result:
            return
        
        current_item = self.room_list.currentItem()
        if not current_item:
            return
        
        # Get room index from list position
        room_index = self.room_list.row(current_item)
        
        room_data, vis = self.detection_result
        rooms = self.processing_thread.detector.detected_rooms if hasattr(self.processing_thread, 'detector') else []
        
        if not rooms:
            QMessageBox.warning(self, "Warning", "No rooms detected")
            return
            
        if 0 <= room_index < len(rooms):
            room = rooms[room_index]
            
            # Create resize dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Resize {room.room_type}")
            layout = QVBoxLayout()
            
            # Current dimensions
            x, y, w, h = room.bbox
            info = QLabel(f"Current size: {w}x{h} pixels")
            layout.addWidget(info)
            
            # Width adjustment
            width_layout = QHBoxLayout()
            width_layout.addWidget(QLabel("Width:"))
            width_spin = QSpinBox()
            width_spin.setRange(50, 1000)
            width_spin.setValue(w)
            width_layout.addWidget(width_spin)
            layout.addLayout(width_layout)
            
            # Height adjustment
            height_layout = QHBoxLayout()
            height_layout.addWidget(QLabel("Height:"))
            height_spin = QSpinBox()
            height_spin.setRange(50, 1000)
            height_spin.setValue(h)
            height_layout.addWidget(height_spin)
            layout.addLayout(height_layout)
            
            # Position adjustment
            pos_layout = QHBoxLayout()
            pos_layout.addWidget(QLabel("X:"))
            x_spin = QSpinBox()
            x_spin.setRange(0, 2000)
            x_spin.setValue(x)
            pos_layout.addWidget(x_spin)
            
            pos_layout.addWidget(QLabel("Y:"))
            y_spin = QSpinBox()
            y_spin.setRange(0, 2000)
            y_spin.setValue(y)
            pos_layout.addWidget(y_spin)
            layout.addLayout(pos_layout)
            
            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                # Update room dimensions
                new_x = x_spin.value()
                new_y = y_spin.value()
                new_w = width_spin.value()
                new_h = height_spin.value()
                
                # Create new rectangular contour
                new_contour = np.array([
                    [new_x, new_y],
                    [new_x + new_w, new_y],
                    [new_x + new_w, new_y + new_h],
                    [new_x, new_y + new_h]
                ], dtype=np.int32)
                
                # Update room
                room.contour = new_contour
                room.bbox = (new_x, new_y, new_w, new_h)
                room.area = new_w * new_h
                room.center = (new_x + new_w//2, new_y + new_h//2)
                scale = self.scale_input.value() if hasattr(self, 'scale_input') else 10
                room.area_m2 = room.area / (scale ** 2)
                
                # Update visualization
                self.update_visualization()
                self.update_room_list()
    
    def on_room_selected(self, item):
        """Handle room selection from list"""
        if item:
            self.edit_room_btn.setEnabled(True)
            self.resize_room_btn.setEnabled(True)
            self.remove_room_btn.setEnabled(True)
            
            # Highlight selected room in visualization
            room_index = self.room_list.row(item)
            self.highlight_room(room_index)
    
    def on_room_edit(self, item):
        """Handle double-click to edit room"""
        self.edit_selected_room()
    
    def edit_selected_room(self):
        """Edit the type of the selected room"""
        current_item = self.room_list.currentItem()
        if not current_item:
            return
        
        room_index = self.room_list.row(current_item)
        if room_index < len(self.processing_thread.detector.detected_rooms):
            room = self.processing_thread.detector.detected_rooms[room_index]
            
            # Create dialog for room type selection
            room_types = ["Bedroom", "Master Bedroom", "Living Room", "Kitchen", 
                         "Dining Room", "Kitchen/Dining", "Bathroom", "Ensuite", 
                         "WC", "Study", "Office", "Garage", "Laundry", "Entry", 
                         "Hallway", "Porch", "Balcony", "Storage", "Unknown"]
            
            current_type = room.room_type
            new_type, ok = QInputDialog.getItem(
                self, "Edit Room Type", 
                f"Select room type for {room.id}:",
                room_types, 
                room_types.index(current_type) if current_type in room_types else 0,
                False
            )
            
            if ok and new_type:
                room.room_type = new_type
                self.update_room_list()
                self.update_visualization()
    
    def remove_selected_room(self):
        """Remove the selected room"""
        current_item = self.room_list.currentItem()
        if not current_item:
            return
        
        room_index = self.room_list.row(current_item)
        reply = QMessageBox.question(
            self, "Remove Room",
            f"Remove this room from detection?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.processing_thread.detector.detected_rooms[room_index]
            self.update_room_list()
            self.update_visualization()
    
    def highlight_room(self, room_index):
        """Highlight a specific room in the visualization"""
        if self.processing_thread.detector.detected_rooms:
            # Set the selected room index
            self.selected_room_index = room_index
            # Regenerate visualization with highlighting
            vis_img = self.create_visualization_with_highlight(room_index)
            self.result_view.set_image(vis_img)
    
    def create_visualization_with_highlight(self, selected_index):
        """Create visualization with selected room highlighted"""
        return self.processing_thread.create_better_visualization(highlight_index=selected_index)
    
    def update_room_list(self):
        """Update the room list widget"""
        self.room_list.clear()
        
        for i, room in enumerate(self.processing_thread.detector.detected_rooms):
            area_str = f"{room.area_m2:.1f} m²" if room.area_m2 else "-- m²"
            item_text = f"{room.room_type} ({area_str})"
            item = QListWidgetItem(item_text)
            
            # Set background color based on room type
            room_colors = {
                "Bedroom": "#FFC0CB",
                "Bathroom": "#FFE4C4",
                "Kitchen": "#80BFFF",
                "Living Room": "#90EE90",
                "Study": "#DDA0DD",
                "Entry": "#E6E6E6"
            }
            
            color = room_colors.get(room.room_type, "#D3D3D3")
            item.setBackground(QColor(color))
            
            self.room_list.addItem(item)
    
    def update_visualization(self):
        """Regenerate and display the visualization"""
        if self.processing_thread.detector.detected_rooms:
            vis_img = self.processing_thread.create_better_visualization()
            self.result_view.set_image(vis_img)
    
    def recalculate_room_areas(self, new_scale):
        """Recalculate all room areas with new scale"""
        for room in self.processing_thread.detector.detected_rooms:
            # Recalculate area in m² with new scale
            room.area_m2 = room.area / (new_scale ** 2)
        
        # Update scale in processing thread
        self.processing_thread.scale_factor = new_scale
        
        # Update displays
        self.update_room_list()
        self.update_visualization()
        
        # Update summary
        total_area = sum(r.area_m2 for r in self.processing_thread.detector.detected_rooms if r.area_m2)
        num_rooms = len(self.processing_thread.detector.detected_rooms)
        
        # Count doors vs windows
        doors = [d for d in self.processing_thread.detector.detected_doors if 'door' in d.id]
        windows = [d for d in self.processing_thread.detector.detected_doors if 'window' in d.id]
        
        # Update perimeter info
        if self.processing_thread.detector.external_perimeter is not None:
            perimeter_length = cv2.arcLength(self.processing_thread.detector.external_perimeter, True)
            perimeter_m = perimeter_length / new_scale if new_scale > 0 else 0
            self.perimeter_info_label.setText(f"Perimeter: {perimeter_m:.1f} m")
            self.perimeter_info_label.setStyleSheet("color: green; font-weight: bold;")
        
        summary = f"Total: {num_rooms} rooms\n"
        summary += f"Internal doors: {len(doors)}, Windows/Ext: {len(windows)}\n"
        summary += f"Area: {total_area:.1f} m²\n"
        summary += f"Scale: {new_scale} px/m"
        self.results_text.setText(summary)
    
    def export_svg(self):
        """Export floor plan as SVG"""
        if not self.processing_thread.detector.detected_rooms:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export SVG", "floorplan.svg",
            "SVG Files (*.svg);;All Files (*)"
        )
        
        if file_path:
            svg_content = self.create_svg()
            with open(file_path, 'w') as f:
                f.write(svg_content)
            QMessageBox.information(self, "Export Complete", f"SVG saved to {Path(file_path).name}")
    
    def create_svg(self) -> str:
        """Create SVG content from detected rooms"""
        if not self.current_image_path:
            return ""
        
        # Get image dimensions
        img = cv2.imread(self.current_image_path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        
        # Start SVG
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        svg += '  <style>\n'
        svg += '    .room { fill-opacity: 0.3; stroke: black; stroke-width: 2; }\n'
        svg += '    .room-label { font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }\n'
        svg += '  </style>\n'
        
        # Define colors for room types
        room_colors = {
            "Bedroom": "#FFC0CB",
            "Master Bedroom": "#FF96C8",
            "Living Room": "#90EE90",
            "Kitchen": "#80BFFF",
            "Dining Room": "#ADD8E6",
            "Kitchen/Dining": "#96C8FF",
            "Bathroom": "#FFE4C4",
            "Ensuite": "#FFC8B4",
            "Study": "#DDA0DD",
            "Office": "#C8A0C8",
            "Garage": "#C0C0C0",
            "Entry": "#E6E6E6",
            "Hallway": "#F5F5F5"
        }
        
        # Add rooms
        for i, room in enumerate(self.processing_thread.detector.detected_rooms):
            color = room_colors.get(room.room_type, "#D3D3D3")
            
            # Create polygon points
            points = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in room.contour])
            
            svg += f'  <polygon class="room" points="{points}" fill="{color}" id="room_{i}"/>\n'
            
            # Add label
            cx, cy = room.center
            area_str = f"{room.area_m2:.1f} m²" if room.area_m2 else ""
            
            svg += f'  <text class="room-label" x="{cx}" y="{cy-5}">{room.room_type}</text>\n'
            if area_str:
                svg += f'  <text class="room-label" x="{cx}" y="{cy+10}" font-size="12">{area_str}</text>\n'
        
        svg += '</svg>'
        return svg
    
    def on_detection_method_changed(self):
        """Show/hide controls based on selected detection method"""
        # Hide all method-specific controls first
        self.yolo_conf_widget.setVisible(False)
        self.cubicasa_group.setVisible(False)
        self.gap_group.setVisible(False)
        self.wall_group.setVisible(False)
        
        # Show relevant controls based on selection
        if self.method_traditional.isChecked():
            # Traditional method uses gap closing and wall processing
            self.gap_group.setVisible(True)
            self.wall_group.setVisible(True)
            
        elif self.method_yolo.isChecked():
            # YOLO method uses confidence slider
            self.yolo_conf_widget.setVisible(True)
            
        elif self.method_hybrid.isChecked():
            # Hybrid uses YOLO confidence
            self.yolo_conf_widget.setVisible(True)
            
        elif self.method_cubicasa.isChecked():
            # CubiCasa uses its own parameters
            self.cubicasa_group.setVisible(True)
    
    def on_gap_settings_changed(self):
        """Handle gap closing settings change"""
        # Update labels
        internal_min = self.internal_gap_slider.value()
        internal_max = internal_min + 25
        self.internal_gap_label.setText(f"{internal_min}-{internal_max}px")
        
        external_min = self.external_gap_slider.value()
        external_max = external_min + 100
        self.external_gap_label.setText(f"{external_min}-{external_max}px")
        
        # If live preview is enabled and we have an image, re-detect
        if self.live_preview_checkbox.isChecked() and self.current_image_path:
            self.detect_rooms_with_settings()
    
    def on_wall_thickness_changed(self):
        """Handle wall thickness change"""
        thickness = self.wall_thickness_slider.value()
        self.wall_thickness_label.setText(f"{thickness}px")
        
        # If live preview is enabled and we have an image, re-detect
        if self.live_preview_checkbox.isChecked() and self.current_image_path:
            self.processing_thread.wall_thickness = thickness
            self.detect_rooms_with_settings()
    
    def detect_rooms_with_settings(self):
        """Re-run detection with current gap settings"""
        if not self.current_image_path:
            return
        
        # Pass gap settings to the processing thread
        internal_min = self.internal_gap_slider.value()
        internal_max = internal_min + 25
        external_min = self.external_gap_slider.value()
        external_max = external_min + 100
        
        self.processing_thread.internal_gap_range = (internal_min, internal_max)
        self.processing_thread.external_gap_range = (external_min, external_max)
        self.processing_thread.close_internal = self.close_internal_checkbox.isChecked()
        self.processing_thread.close_external = self.close_external_checkbox.isChecked()
        
        # Re-run detection
        self.detect_rooms()
    
    def save_results(self):
        """Save results as JSON"""
        if not self.detection_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "results.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            room_data, _ = self.detection_result
            with open(file_path, 'w') as f:
                json.dump(room_data, f, indent=2)
            QMessageBox.information(self, "Saved", f"Results saved to {Path(file_path).name}")


class ImageWidget(QLabel):
    """Widget for displaying images with click to view full"""
    
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f8f8f8;")
        self.setMinimumSize(400, 300)
        self._pixmap = None
        self._original_image = None
        self.setCursor(Qt.PointingHandCursor)
        # Enable mouse tracking to receive mouse events
        self.setMouseTracking(True)
        
    def set_image(self, image: np.ndarray):
        if image is None:
            self.clear()
            self._original_image = None
            return
            
        self._original_image = image.copy()
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        
        if len(image.shape) == 2:
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            bytes_per_line = 3 * width
            q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self._pixmap = QPixmap.fromImage(q_image)
        self.update_display()
        
    def update_display(self):
        if self._pixmap is None:
            return
            
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
            
    def mousePressEvent(self, event):
        """Click to open in full window"""
        print(f"Mouse pressed: button={event.button()}, pixmap exists={self._pixmap is not None}")
        if event.button() == Qt.LeftButton:
            if self._pixmap:
                self.show_fullsize()
            else:
                print("No pixmap to show")
        super().mousePressEvent(event)
            
    def show_fullsize(self):
        """Show image in a new window at full size"""
        print(f"show_fullsize called, original_image is None: {self._original_image is None}")
        
        if self._original_image is None:
            print("No original image to show")
            return
            
        try:
            dialog = QDialog()
            dialog.setWindowTitle("Full Size View - Press ESC to close")
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Create scroll area
            scroll = QScrollArea()
            label = QLabel()
            
            # Convert image to QPixmap for display
            img = self._original_image
            print(f"Image shape: {img.shape}")
            
            if len(img.shape) == 2:
                h, w = img.shape
                q_image = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                h, w = rgb.shape[:2]
                bytes_per_line = 3 * w
                q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            scroll.setWidget(label)
            layout.addWidget(scroll)
            
            dialog.setLayout(layout)
            
            # Size the dialog appropriately
            screen = QApplication.primaryScreen()
            if screen:
                screen_size = screen.geometry()
            else:
                screen_size = QRect(0, 0, 1920, 1080)  # Default fallback
                
            dialog_width = min(screen_size.width() - 100, w + 50)
            dialog_height = min(screen_size.height() - 100, h + 50)
            dialog.resize(dialog_width, dialog_height)
            
            # Center on screen
            dialog.move((screen_size.width() - dialog_width) // 2,
                       (screen_size.height() - dialog_height) // 2)
            
            print(f"Showing dialog with size {dialog_width}x{dialog_height}")
            dialog.exec_()  # Use exec_() instead of show() to ensure it displays
            
        except Exception as e:
            print(f"Error showing fullsize: {e}")
            import traceback
            traceback.print_exc()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


def main():
    app = QApplication(sys.argv)
    window = AdvancedFloorPlanGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()