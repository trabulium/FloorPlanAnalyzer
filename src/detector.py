#!/usr/bin/env python3
"""
Advanced Floor Plan Detector
Implements sophisticated algorithms from the AFPARS Java application:
- Template matching for door detection
- Connected component analysis for room detection
- Gap closing algorithms
- Morphological operations for noise removal
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path


@dataclass
class Room:
    """Represents a detected room"""
    id: str
    contour: np.ndarray
    area: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    room_type: str = "Unknown"
    area_m2: Optional[float] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'area_pixels': float(self.area),
            'area_m2': float(self.area_m2) if self.area_m2 else None,
            'center': [int(self.center[0]), int(self.center[1])],
            'bbox': [int(x) for x in self.bbox],
            'room_type': self.room_type
        }


@dataclass
class Door:
    """Represents a detected door"""
    id: str
    location: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    def to_dict(self):
        return {
            'id': self.id,
            'location': [int(self.location[0]), int(self.location[1])],
            'bbox': [int(x) for x in self.bbox],
            'confidence': float(self.confidence)
        }


class AdvancedFloorPlanDetector:
    """
    Advanced floor plan detection using techniques from AFPARS:
    - Morphological cleaning
    - Template matching for doors
    - Gap closing
    - Connected component analysis
    """
    
    def __init__(self):
        self.scale_factor = None  # pixels per meter
        self.detected_rooms = []
        self.detected_doors = []
        self.detected_walls = None
        self.external_perimeter = None  # Store the external boundary
        
    def process_floor_plan(self, image_path: str, scale_factor: float = 50.0, debug: bool = False,
                          internal_gap_range: Tuple[int, int] = (25, 50),
                          external_gap_range: Tuple[int, int] = (50, 150),
                          close_internal: bool = True,
                          close_external: bool = False,
                          wall_thickness: int = 3,
                          use_label_detection: bool = True) -> Dict:
        """
        Process a floor plan image
        
        Args:
            image_path: Path to the floor plan image
            scale_factor: Pixels per meter (default 50)
            
        Returns:
            Dictionary with detected elements
        """
        self.scale_factor = scale_factor
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Step 1: Morphological cleaning (noise removal)
        cleaned = self.morphological_cleaning(img, wall_thickness)
        if debug:
            cv2.imwrite("debug_1_cleaned.png", cleaned)
        
        # Step 2: Detect doors using template matching
        self.detected_doors = self.detect_doors_template_matching(cleaned, internal_gap_range, external_gap_range)
        print(f"Detected {len(self.detected_doors)} doors")
        
        # Step 3: Detect rooms using label-based approach if enabled
        if use_label_detection:
            # First detect room labels with OCR
            room_labels = self.detect_room_labels_ocr(image_path)
            print(f"Detected {len(room_labels)} room labels via OCR")
            
            # Then find room boundaries based on labels
            self.detected_rooms = self.detect_rooms_from_labels(cleaned, room_labels, scale_factor)
            print(f"Detected {len(self.detected_rooms)} rooms from labels")
        else:
            # Try erosion-based separation first
            self.detected_rooms = self.detect_rooms_using_erosion(cleaned)
            
            # If erosion doesn't work well, fall back to watershed
            if len(self.detected_rooms) <= 1:
                segmented = self.watershed_room_segmentation(cleaned)
                if debug:
                    cv2.imwrite("debug_2_segmented.png", segmented)
                self.detected_rooms = self.extract_rooms_from_segments(segmented)
                print(f"Detected {len(self.detected_rooms)} rooms via watershed")
            else:
                print(f"Detected {len(self.detected_rooms)} rooms via erosion")
        
        # Step 5: Calculate room areas in m²
        self.calculate_room_areas()
        
        # Step 6: Detect walls
        self.detected_walls = self.detect_walls(cleaned)
        
        # Step 7: Detect external perimeter
        self.external_perimeter = self.detect_external_perimeter(cleaned)
        if debug and self.external_perimeter is not None:
            perimeter_img = img.copy()
            cv2.drawContours(perimeter_img, [self.external_perimeter], -1, (0, 255, 0), 3)
            cv2.imwrite("debug_3_perimeter.png", perimeter_img)
        
        return {
            'rooms': [room.to_dict() for room in self.detected_rooms],
            'doors': [door.to_dict() for door in self.detected_doors],
            'total_area_m2': sum(r.area_m2 for r in self.detected_rooms if r.area_m2),
            'num_rooms': len(self.detected_rooms),
            'num_doors': len(self.detected_doors)
        }
    
    def morphological_cleaning(self, image: np.ndarray, wall_thickness: int = 3) -> np.ndarray:
        """
        Apply morphological operations for noise removal
        Similar to MorphologicalTransform in AFPARS
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple threshold like AFPARS (they use simple threshold not adaptive)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Remove small noise (text, dots) but preserve walls
        kernel_small = np.ones((2, 2), np.uint8)
        
        # Opening to remove small noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Only thicken walls if requested (0 means no thickening)
        if wall_thickness > 0:
            cleaned = self.thicken_walls(cleaned, wall_thickness)
        
        return cleaned
    
    def thicken_walls(self, image: np.ndarray, thickness: int = 3) -> np.ndarray:
        """
        Thicken walls to make them more prominent for detection
        """
        # Invert so walls are white
        inverted = cv2.bitwise_not(image)
        
        # Dilate to thicken walls
        kernel = np.ones((thickness, thickness), np.uint8)
        thickened = cv2.dilate(inverted, kernel, iterations=1)
        
        # Invert back
        return cv2.bitwise_not(thickened)
    
    def detect_doors_template_matching(self, image: np.ndarray, 
                                      internal_range: Tuple[int, int] = (25, 50),
                                      external_range: Tuple[int, int] = (50, 150)) -> List[Door]:
        """
        Detect doors using improved algorithm
        Look for gaps in walls that indicate doorways
        """
        doors = []
        
        # Method 1: Detect gaps in walls (door openings)
        doors_from_gaps = self.detect_door_gaps(image, internal_range, external_range)
        
        # Deduplicate doors that are too close together
        for door in doors_from_gaps:
            if not self.is_overlapping_door(door.location, doors, min_distance=15):
                doors.append(door)
        
        # Method 2: Template matching for door symbols if templates exist
        door_templates = self.create_door_templates()
        if door_templates:
            for template_name, template in door_templates.items():
                # Apply template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where matching score > threshold
                threshold = 0.6  # Lower threshold
                locations = np.where(result >= threshold)
                
                # Convert to list of points
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    
                    # Check if this location doesn't overlap with existing doors
                    if not self.is_overlapping_door(pt, doors, min_distance=20):
                        door = Door(
                            id=f"door_{len(doors)}",
                            location=pt,
                            bbox=(pt[0], pt[1], template.shape[1], template.shape[0]),
                            confidence=float(confidence)
                        )
                        doors.append(door)
        
        return doors
    
    def detect_door_gaps(self, image: np.ndarray,
                        internal_range: Tuple[int, int] = (25, 50),
                        external_range: Tuple[int, int] = (50, 150)) -> List[Door]:
        """
        Detect doors by finding gaps in walls
        Differentiate between internal doors and external windows/doors
        """
        doors = []
        h, w = image.shape[:2]
        
        # Invert image so walls are white
        inverted = cv2.bitwise_not(image)
        
        # Find horizontal and vertical lines (walls)
        horizontal = self.detect_horizontal_lines(inverted)
        vertical = self.detect_vertical_lines(inverted)
        
        # Find gaps in horizontal walls (vertical doors)
        for y, line_segments in horizontal.items():
            for i in range(len(line_segments) - 1):
                gap_start = line_segments[i][1]  # End of current segment
                gap_end = line_segments[i + 1][0]  # Start of next segment
                gap_size = gap_end - gap_start
                gap_center = (gap_start + gap_end) // 2
                
                # Determine if this is an exterior gap (near image edge)
                is_exterior = (y < 50 or y > h - 50)
                
                # Use configured ranges
                if is_exterior:
                    # External openings (windows/sliding doors)
                    if external_range[0] < gap_size < external_range[1]:
                        door = Door(
                            id=f"window_{len(doors)}",
                            location=(gap_center, y),
                            bbox=(gap_start, y - 5, gap_size, 10),
                            confidence=0.7
                        )
                        doors.append(door)
                else:
                    # Internal doors
                    if internal_range[0] < gap_size < internal_range[1]:
                        door = Door(
                            id=f"door_{len(doors)}",
                            location=(gap_center, y),
                            bbox=(gap_start, y - 5, gap_size, 10),
                            confidence=0.9
                        )
                        doors.append(door)
        
        # Find gaps in vertical walls (horizontal doors)
        for x, line_segments in vertical.items():
            for i in range(len(line_segments) - 1):
                gap_start = line_segments[i][1]  # End of current segment
                gap_end = line_segments[i + 1][0]  # Start of next segment
                gap_size = gap_end - gap_start
                gap_center = (gap_start + gap_end) // 2
                
                # Determine if this is an exterior gap
                is_exterior = (x < 50 or x > w - 50)
                
                if is_exterior:
                    # External openings
                    if external_range[0] < gap_size < external_range[1]:
                        door = Door(
                            id=f"window_{len(doors)}",
                            location=(x, gap_center),
                            bbox=(x - 5, gap_start, 10, gap_size),
                            confidence=0.7
                        )
                        doors.append(door)
                else:
                    # Internal doors
                    if internal_range[0] < gap_size < internal_range[1]:
                        door = Door(
                            id=f"door_{len(doors)}",
                            location=(x, gap_center),
                            bbox=(x - 5, gap_start, 10, gap_size),
                            confidence=0.9
                        )
                        doors.append(door)
        
        return doors
    
    def detect_horizontal_lines(self, image: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """Detect horizontal wall segments"""
        h, w = image.shape[:2]
        horizontal_lines = {}
        
        # Group nearby horizontal lines together (within 3 pixels)
        line_groups = {}
        
        for y in range(h):
            row = image[y, :]
            # Find continuous white segments (walls)
            segments = []
            start = None
            
            for x in range(w):
                if row[x] > 200:  # Wall pixel
                    if start is None:
                        start = x
                else:  # Gap or non-wall
                    if start is not None and x - start > 20:  # Minimum wall length
                        segments.append((start, x))
                    start = None
            
            # Add last segment if exists
            if start is not None and w - start > 20:
                segments.append((start, w))
            
            if segments:
                # Check if this line should be grouped with a nearby line
                grouped = False
                for group_y in range(max(0, y-3), y):
                    if group_y in horizontal_lines:
                        # Merge with existing group
                        horizontal_lines[group_y].extend(segments)
                        grouped = True
                        break
                
                if not grouped:
                    horizontal_lines[y] = segments
        
        return horizontal_lines
    
    def detect_vertical_lines(self, image: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """Detect vertical wall segments"""
        h, w = image.shape[:2]
        vertical_lines = {}
        
        for x in range(w):
            col = image[:, x]
            # Find continuous white segments (walls)
            segments = []
            start = None
            
            for y in range(h):
                if col[y] > 200:  # Wall pixel
                    if start is None:
                        start = y
                else:  # Gap or non-wall
                    if start is not None and y - start > 20:  # Minimum wall length
                        segments.append((start, y))
                    start = None
            
            # Add last segment if exists
            if start is not None and h - start > 20:
                segments.append((start, h))
            
            if segments:
                vertical_lines[x] = segments
        
        return vertical_lines
    
    def create_door_templates(self) -> Dict[str, np.ndarray]:
        """Create door templates for matching"""
        templates = {}
        
        # Try to load actual door template
        if Path("door_template.png").exists():
            door_template = cv2.imread("door_template.png", cv2.IMREAD_GRAYSCALE)
            if door_template is not None:
                templates['door'] = door_template
                # Create rotated versions
                templates['door_90'] = cv2.rotate(door_template, cv2.ROTATE_90_CLOCKWISE)
                templates['door_180'] = cv2.rotate(door_template, cv2.ROTATE_180)
                templates['door_270'] = cv2.rotate(door_template, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Fallback templates if no file
        if not templates:
            # Vertical door template (simplified)
            v_door = np.zeros((60, 20), dtype=np.uint8)
            v_door[10:50, 5:15] = 255
            templates['vertical'] = v_door
            
            # Horizontal door template
            h_door = np.zeros((20, 60), dtype=np.uint8)
            h_door[5:15, 10:50] = 255
            templates['horizontal'] = h_door
        
        return templates
    
    def is_overlapping_door(self, pt: Tuple[int, int], existing_doors: List[Door], 
                           min_distance: int = 30) -> bool:
        """Check if a point overlaps with existing doors"""
        for door in existing_doors:
            dist = np.sqrt((pt[0] - door.location[0])**2 + (pt[1] - door.location[1])**2)
            if dist < min_distance:
                return True
        return False
    
    def separate_rooms_at_doors(self, image: np.ndarray, doors: List[Door]) -> np.ndarray:
        """
        Draw walls at door locations to separate rooms
        This ensures rooms don't merge together
        """
        result = image.copy()
        
        # Draw thick black lines at each door location to separate rooms
        for door in doors:
            if 'door' in door.id:  # Only for internal doors
                x, y, w, h = door.bbox
                # Draw a thick wall across the door opening
                if h > w:  # Vertical door
                    cv2.rectangle(result, (x-2, y), (x+w+2, y+h), 0, -1)
                else:  # Horizontal door
                    cv2.rectangle(result, (x, y-2), (x+w, y+h+2), 0, -1)
        
        return result
    
    def close_small_gaps_only(self, image: np.ndarray) -> np.ndarray:
        """
        Close only very small gaps (noise) but preserve door-sized gaps
        This prevents rooms from merging together at doorways
        """
        result = image.copy()
        
        # Use a small kernel to close only tiny gaps
        kernel_tiny = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
        
        return result
    
    def detect_room_labels_ocr(self, image_path: str) -> List[Dict]:
        """
        Use OCR to detect room labels in the floor plan
        Returns list of dictionaries with label text and position
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            print("Warning: pytesseract not available, skipping OCR")
            return []
        
        # Load image for OCR
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Run OCR with bounding boxes
        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"OCR failed: {e}")
            return []
        
        room_labels = []
        
        # More comprehensive room keywords (lowercase)
        room_keywords = {
            'bedroom': 'BED', 'bed': 'BED', 'master': 'BED',
            'living': 'LIVING', 'lounge': 'LIVING', 'family': 'LIVING',
            'kitchen': 'KITCHEN', 'dining': 'DINING',
            'bathroom': 'BATH', 'bath': 'BATH', 'wc': 'WC', 'toilet': 'BATH',
            'ensuite': 'BATH', 'ens': 'BATH', 'en-suite': 'BATH',
            'study': 'STUDY', 'office': 'STUDY',
            'garage': 'GARAGE', 'carport': 'GARAGE',
            'laundry': 'LAUNDRY', 'utility': 'LAUNDRY',
            'porch': 'PORCH', 'balcony': 'PORCH',
            'entry': 'ENTRY', 'foyer': 'ENTRY', 'hall': 'ENTRY',
            'pantry': 'KITCHEN', 'ldy': 'LAUNDRY', 'pwd': 'BATH',
            'robe': 'BED', 'store': 'STUDY'
        }
        
        # Process OCR results
        for i in range(len(data['text'])):
            text = data['text'][i].strip().lower()  # Use lowercase for matching
            conf = int(data['conf'][i])
            
            if conf > 20 and text:  # Lower confidence threshold
                # Check if text contains room keywords
                matched = False
                for keyword, room_type in room_keywords.items():
                    if keyword in text:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        # Center of text
                        cx = x + w // 2
                        cy = y + h // 2
                        
                        room_labels.append({
                            'text': data['text'][i].strip(),  # Keep original text
                            'type': room_type,
                            'position': (cx, cy),
                            'bbox': (x, y, w, h),
                            'confidence': conf
                        })
                        matched = True
                        break
                
                # Also check for room dimensions (e.g., "3.0x2.9")
                if not matched and 'x' in text and any(c.isdigit() for c in text):
                    # This might be a room dimension, look for nearby room type
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Store as potential room center
                    room_labels.append({
                        'text': data['text'][i].strip(),
                        'type': 'ROOM',  # Generic room
                        'position': (x + w//2, y + h//2),
                        'bbox': (x, y, w, h),
                        'confidence': conf
                    })
        
        return room_labels
    
    def detect_rooms_from_labels(self, image: np.ndarray, room_labels: List[Dict], 
                                 scale_factor: float) -> List[Room]:
        """
        Detect room boundaries starting from label positions
        """
        rooms = []
        
        # Define minimum room dimensions in meters
        min_room_sizes = {
            'BED': (3.0, 3.0),      # Bedroom: 3x3m minimum
            'BATH': (1.5, 2.0),     # Bathroom: 1.5x2m minimum
            'LIVING': (3.5, 4.0),   # Living room: 3.5x4m minimum
            'KITCHEN': (2.5, 3.0),  # Kitchen: 2.5x3m minimum
            'DINING': (3.0, 3.0),   # Dining: 3x3m minimum
            'STUDY': (2.5, 2.5),    # Study: 2.5x2.5m minimum
            'GARAGE': (3.0, 5.5),   # Garage: 3x5.5m minimum
            'LAUNDRY': (2.0, 2.0),  # Laundry: 2x2m minimum
            'ENTRY': (2.0, 2.0),    # Entry: 2x2m minimum
            'WC': (1.0, 1.5),       # WC: 1x1.5m minimum
        }
        
        # Invert image for room detection
        inverted = cv2.bitwise_not(image)
        
        for label in room_labels:
            cx, cy = label['position']
            room_type = label['type']
            
            # Get minimum size for this room type
            min_width_m, min_height_m = min_room_sizes.get(room_type, (2.0, 2.0))
            min_width_px = int(min_width_m * scale_factor)
            min_height_px = int(min_height_m * scale_factor)
            
            # Find room boundaries from label position
            room_contour = self.grow_room_from_point(inverted, cx, cy, 
                                                     min_width_px, min_height_px)
            
            if room_contour is not None:
                area = cv2.contourArea(room_contour)
                x, y, w, h = cv2.boundingRect(room_contour)
                
                # Skip if too small
                if area < 500:
                    continue
                
                room = Room(
                    id=f"room_{len(rooms)}",
                    contour=room_contour,
                    area=area,
                    center=(cx, cy),
                    bbox=(x, y, w, h),
                    room_type=label['text']  # Use actual label text
                )
                rooms.append(room)
        
        return rooms
    
    def grow_room_from_point(self, inverted: np.ndarray, cx: int, cy: int,
                             min_width: int, min_height: int) -> np.ndarray:
        """
        Find room boundary by detecting walls around a label position
        """
        h, w = inverted.shape[:2]
        
        # Ensure point is within bounds
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return None
        
        # Create edge map to find walls
        edges = cv2.Canny(inverted, 50, 150)
        
        # Find bounding walls by ray casting in 4 directions
        # Start from label position and move outward until hitting a wall
        
        # Cast rays to find walls
        left_wall = cx
        right_wall = cx
        top_wall = cy
        bottom_wall = cy
        
        # Search left
        for x in range(cx, max(0, cx - min_width), -1):
            if edges[cy, x] > 0 or x == 0:
                left_wall = x
                break
        
        # Search right
        for x in range(cx, min(w, cx + min_width)):
            if edges[cy, x] > 0 or x == w-1:
                right_wall = x
                break
        
        # Search up
        for y in range(cy, max(0, cy - min_height), -1):
            if edges[y, cx] > 0 or y == 0:
                top_wall = y
                break
        
        # Search down
        for y in range(cy, min(h, cy + min_height)):
            if edges[y, cx] > 0 or y == h-1:
                bottom_wall = y
                break
        
        # Create rectangular room boundary
        room_width = right_wall - left_wall
        room_height = bottom_wall - top_wall
        
        # Check minimum size
        if room_width < min_width * 0.5 or room_height < min_height * 0.5:
            return None
        
        # Create contour from rectangle
        room_contour = np.array([
            [[left_wall, top_wall]],
            [[right_wall, top_wall]],
            [[right_wall, bottom_wall]],
            [[left_wall, bottom_wall]]
        ], dtype=np.int32)
        
        return room_contour
    
    def detect_rooms_using_erosion(self, image: np.ndarray) -> List[Room]:
        """
        Detect rooms by eroding at doorways to separate connected rooms
        """
        # First close very small gaps
        kernel_close = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
        
        # Invert so rooms are white
        inverted = cv2.bitwise_not(closed)
        
        # Erode to break connections at doorways (which are thinner)
        kernel_erode = np.ones((11,11), np.uint8)  # Kernel to break at doorways
        eroded = cv2.erode(inverted, kernel_erode, iterations=1)
        
        # Find connected components (individual rooms)
        num_labels, labels = cv2.connectedComponents(eroded)
        
        rooms = []
        for label_id in range(1, num_labels):
            # Get mask for this component
            mask = (labels == label_id).astype(np.uint8) * 255
            
            # Dilate back to restore room size
            restored = cv2.dilate(mask, kernel_erode, iterations=1)
            
            # Mask with original room area
            restored = cv2.bitwise_and(restored, inverted)
            
            # Find contour
            contours, _ = cv2.findContours(restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                
                if area > 1000:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    room = Room(
                        id=f"room_{len(rooms)}",
                        contour=contour,
                        area=area,
                        center=(cx, cy),
                        bbox=(x, y, w, h),
                        room_type="Unknown"
                    )
                    rooms.append(room)
        
        return rooms
    
    def watershed_room_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Use watershed algorithm to segment connected rooms
        """
        # First close small gaps to connect walls
        kernel_close = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Invert so rooms are white
        inverted = cv2.bitwise_not(closed)
        
        # Distance transform to find room centers
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # Use adaptive threshold based on distance values
        # This creates multiple seed points for watershed
        local_maxima = cv2.dilate(dist, None, iterations=3)
        local_maxima = (dist >= 0.5 * local_maxima)
        
        # Clean up the markers
        markers = np.uint8(local_maxima) * 255
        kernel = np.ones((5,5), np.uint8)
        markers = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel)
        
        # Find connected components for markers
        num_markers, markers = cv2.connectedComponents(markers)
        
        # Add 1 to avoid label 0 (which watershed uses for unknown)
        markers = markers + 1
        
        # Mark the walls as unknown region (0)
        markers[closed == 255] = 0
        
        # Apply watershed
        img_for_watershed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR) if len(closed.shape) == 2 else closed
        markers = cv2.watershed(img_for_watershed, markers)
        
        return markers
    
    def extract_rooms_from_segments(self, markers: np.ndarray) -> List[Room]:
        """
        Extract individual rooms from watershed markers
        """
        rooms = []
        
        # Get unique marker values (excluding 0 and -1)
        unique_markers = np.unique(markers)
        unique_markers = unique_markers[(unique_markers > 0)]
        
        for marker_val in unique_markers:
            # Create mask for this room
            room_mask = (markers == marker_val).astype(np.uint8) * 255
            
            # Find contour
            contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                area = cv2.contourArea(contour)
                
                # Filter small areas
                if area < 500:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = x + w // 2
                    cy = y + h // 2
                
                room = Room(
                    id=f"room_{len(rooms)}",
                    contour=contour,
                    area=area,
                    center=(cx, cy),
                    bbox=(x, y, w, h),
                    room_type=self.classify_room_by_size(area)
                )
                rooms.append(room)
        
        return rooms
    
    def segment_rooms_watershed_old(self, image: np.ndarray, doors: List[Door]) -> np.ndarray:
        """
        Use watershed algorithm to segment rooms at door locations
        This prevents all rooms from merging into one when gaps are closed
        """
        # Invert image so rooms are white
        inverted = cv2.bitwise_not(image)
        
        # Create markers for watershed
        # Use distance transform to find room centers
        dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # Find peaks in distance transform (room centers)
        _, markers = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        markers = np.uint8(markers)
        
        # Find connected components of markers
        num_markers, markers = cv2.connectedComponents(markers)
        
        # Add door locations as barriers (markers = 1)
        for door in doors:
            x, y, w, h = door.bbox
            # Mark door area as barrier
            if h > w:  # Vertical door
                cv2.line(markers, (x + w//2, y), (x + w//2, y + h), 1, thickness=3)
            else:  # Horizontal door
                cv2.line(markers, (x, y + h//2), (x + w, y + h//2), 1, thickness=3)
        
        # Apply watershed
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
        markers = cv2.watershed(img_color, markers)
        
        # Create result image
        result = image.copy()
        
        # Mark watershed boundaries as walls
        result[markers == -1] = 0
        
        # Also ensure door gaps remain as walls
        for door in doors:
            x, y, w, h = door.bbox
            if 'door' in door.id:  # Only for internal doors
                # Draw lines to separate rooms at door locations
                if h > w:  # Vertical door
                    cv2.line(result, (x, y), (x + w, y), 0, thickness=2)
                    cv2.line(result, (x, y + h), (x + w, y + h), 0, thickness=2)
                else:  # Horizontal door
                    cv2.line(result, (x, y), (x, y + h), 0, thickness=2)
                    cv2.line(result, (x + w, y), (x + w, y + h), 0, thickness=2)
        
        return result
    
    def close_gaps(self, image: np.ndarray, doors: List[Door],
                   close_internal: bool = True, close_external: bool = False) -> np.ndarray:
        """
        Close gaps in walls, especially at door locations
        Only close INTERNAL doors, not external windows/doors
        Based on GapClosingAlgorithm from AFPARS
        """
        result = image.copy()
        
        # First, do a light closing to connect very close walls
        kernel_small = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Close gaps based on settings
        for door in doors:
            should_close = False
            if 'door' in door.id and close_internal:
                should_close = True
            elif 'window' in door.id and close_external:
                should_close = True
            
            if should_close:
                x, y, w, h = door.bbox
                
                # Draw lines to close the gap
                thickness = 2
                # Vertical door opening
                if h > w:
                    cv2.line(result, (x, y), (x + w, y), 0, thickness)
                    cv2.line(result, (x, y + h), (x + w, y + h), 0, thickness)
                # Horizontal door opening
                else:
                    cv2.line(result, (x, y), (x, y + h), 0, thickness)
                    cv2.line(result, (x + w, y), (x + w, y + h), 0, thickness)
        
        return result
    
    def detect_rooms_connected_components(self, image: np.ndarray) -> List[Room]:
        """
        Detect rooms using contour detection with hierarchy
        This better handles nested rooms and prevents merging
        """
        # Invert image (rooms should be white)
        inverted = cv2.bitwise_not(image)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours or hierarchy is None:
            return []
        
        rooms = []
        hierarchy = hierarchy[0]
        
        # Get image area for filtering
        img_area = image.shape[0] * image.shape[1]
        
        # Process each contour
        for i, contour in enumerate(contours):
            # Skip if this is a hole inside another contour
            if hierarchy[i][3] != -1:  # Has a parent
                continue
            
            area = cv2.contourArea(contour)
            
            # Filter out very small or very large contours
            if area < 1000 or area > img_area * 0.5:
                continue
            
            # Check if contour touches image border
            if self.is_on_border(contour, image.shape, border_threshold=5):
                continue
            
            # Check if this is a valid room shape (optional - can be disabled)
            # if not self.is_valid_room(contour):
            #     continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            room = Room(
                id=f"room_{len(rooms)}",
                contour=contour,
                area=area,
                center=(cx, cy),
                bbox=(x, y, w, h),
                room_type=self.classify_room_by_size(area)
            )
            rooms.append(room)
        
        return rooms
    
    def detect_rooms_connected_components_old(self, image: np.ndarray) -> List[Room]:
        """
        Detect rooms using connected component analysis
        Based on ConnectedComponentDetection from AFPARS
        """
        # Invert image (rooms should be white)
        inverted = cv2.bitwise_not(image)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        rooms = []
        
        # Get image dimensions for filtering
        img_area = image.shape[0] * image.shape[1]
        
        # Find the largest component (excluding background)
        if num_labels > 1:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            if areas:
                max_area = max(areas)
            else:
                max_area = 0
        else:
            max_area = 0
        
        # Process each component (skip background at index 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out very small components (noise) and very large ones (background)
            # Using 5% threshold as in AFPARS
            if max_area > 0 and (area / max_area) < 0.05:
                continue
                
            if area > img_area * 0.5:  # Skip if larger than 50% of image
                continue
            
            # Get component mask
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours of the component
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                contour = contours[0]
                
                # Check if contour touches image border (based on AFPARS borderApprox)
                if self.is_on_border(contour, image.shape, border_threshold=2):
                    continue
                
                # Validate room geometry
                if not self.is_valid_room(contour):
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                center = (int(centroids[i][0]), int(centroids[i][1]))
                
                room = Room(
                    id=f"room_{len(rooms)}",
                    contour=contour,
                    area=area,
                    center=center,
                    bbox=(x, y, w, h),
                    room_type=self.classify_room_by_size(area)
                )
                rooms.append(room)
        
        return rooms
    
    def is_valid_room(self, contour: np.ndarray, min_corners: int = 2) -> bool:
        """
        Validate if a contour represents a valid room
        A room should have at least 2 corners (approximately 90-degree angles)
        and be roughly rectangular
        """
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)  # More aggressive approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if we have enough vertices (at least 4 for a room)
        if len(approx) < 4:
            return False
        
        # Check if it's too irregular (too many vertices for a room)
        if len(approx) > 12:  # Rooms shouldn't have more than 12 corners
            return False
        
        # Calculate angles at each vertex
        corners_90deg = 0
        for i in range(len(approx)):
            p1 = approx[i-1][0]
            p2 = approx[i][0]
            p3 = approx[(i+1) % len(approx)][0]
            
            # Calculate angle using dot product
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Normalize vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                # Calculate angle
                dot_product = np.dot(v1_norm, v2_norm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                # Check if angle is approximately 90 degrees (within tolerance)
                if 60 <= angle_deg <= 120:  # More lenient angle tolerance
                    corners_90deg += 1
        
        # Room should have at least 2 corners with ~90 degree angles
        return corners_90deg >= min_corners
    
    def is_on_border(self, contour: np.ndarray, img_shape: Tuple[int, int], 
                      border_threshold: int = 2) -> bool:
        """Check if contour touches image border"""
        h, w = img_shape[:2]
        
        for point in contour:
            x, y = point[0]
            if (x <= border_threshold or x >= w - border_threshold or
                y <= border_threshold or y >= h - border_threshold):
                return True
        return False
    
    def classify_room_by_size(self, area: float) -> str:
        """Classify room type based on area"""
        # Simple classification based on area
        if area < 5000:
            return "Bathroom"
        elif area < 10000:
            return "Small Room"
        elif area < 20000:
            return "Bedroom"
        elif area < 30000:
            return "Living Room"
        else:
            return "Large Room"
    
    def calculate_room_areas(self):
        """Calculate room areas in square meters"""
        if self.scale_factor:
            for room in self.detected_rooms:
                # Convert pixel area to m²
                room.area_m2 = room.area / (self.scale_factor ** 2)
    
    def detect_walls(self, image: np.ndarray) -> np.ndarray:
        """Detect walls in the floor plan"""
        # Simple wall detection using edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Use Hough transform to detect lines (walls)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=50, maxLineGap=10)
        
        return lines
    
    def detect_external_perimeter(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the external perimeter of the floor plan
        Based on ExteriorWallClosing from AFPARS
        """
        # Invert the image (walls should be white)
        inverted = cv2.bitwise_not(image)
        
        # Apply morphological operations to connect broken walls
        kernel = np.ones((5, 5), np.uint8)
        
        # Dilate to connect nearby walls
        dilated = cv2.dilate(inverted, kernel, iterations=2)
        
        # Erode back to original size
        closed = cv2.erode(dilated, kernel, iterations=2)
        
        # Find all contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (should be the building outline)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour using convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Further simplify using Douglas-Peucker algorithm
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        return approx
    
    def visualize_results(self, image_path: str, output_path: str = "result.png"):
        """Visualize detection results"""
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # Draw rooms
        for room in self.detected_rooms:
            # Random color for each room
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(img, [room.contour], -1, color, -1)
            
            # Draw room info
            cv2.putText(img, f"{room.room_type}", room.center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if room.area_m2:
                cv2.putText(img, f"{room.area_m2:.1f}m²", 
                           (room.center[0], room.center[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw doors
        for door in self.detected_doors:
            x, y, w, h = door.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Door", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Add summary
        summary = f"Rooms: {len(self.detected_rooms)}, Doors: {len(self.detected_doors)}"
        if self.detected_rooms:
            total_area = sum(r.area_m2 for r in self.detected_rooms if r.area_m2)
            summary += f", Total: {total_area:.1f}m²"
        
        cv2.putText(img, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, img)
        return img


def main():
    """Test the advanced floor plan detector"""
    detector = AdvancedFloorPlanDetector()
    
    # Process a floor plan
    test_image = "granny_flat_eucalypt.png"
    if Path(test_image).exists():
        print(f"Processing {test_image}...")
        # Use a more realistic scale factor (assuming ~100 pixels per meter for typical floor plans)
        results = detector.process_floor_plan(test_image, scale_factor=100.0)
        
        print("\nDetection Results:")
        print(f"Found {results['num_rooms']} rooms")
        print(f"Found {results['num_doors']} doors")
        print(f"Total area: {results['total_area_m2']:.2f} m²")
        
        print("\nRooms:")
        for room in results['rooms']:
            print(f"  - {room['room_type']}: {room['area_m2']:.2f} m²")
        
        # Visualize results
        detector.visualize_results(test_image, "advanced_detection_result.png")
        print("\nVisualization saved to advanced_detection_result.png")
        
        # Save results to JSON
        with open("detection_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to detection_results.json")
    else:
        print(f"Test image {test_image} not found")


if __name__ == "__main__":
    main()