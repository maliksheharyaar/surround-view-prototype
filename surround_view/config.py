"""
System Configuration
====================

Modern configuration system using dataclasses for better type safety and validation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for individual camera settings."""
    name: str
    position: str  # 'front', 'back', 'left', 'right'
    
    @property
    def yaml_file(self) -> str:
        """Path to camera calibration YAML file."""
        return os.path.join(os.getcwd(), "yaml", f"{self.name}.yaml")
    
    @property
    def test_image(self) -> str:
        """Path to camera test image."""
        return os.path.join(os.getcwd(), "images", f"{self.name}.png")


@dataclass(frozen=True)
class ViewDimensions:
    """Bird's eye view dimensions and layout configuration."""
    
    # View extension parameters
    horizontal_extension: int = 300  # How far to extend view horizontally
    vertical_extension: int = 300    # How far to extend view vertically
    
    # Car boundary gaps
    car_horizontal_gap: int = 20     # Gap between calibration pattern and car
    car_vertical_gap: int = 50       # Gap between calibration pattern and car
    
    # Base calibration area (before extensions)
    base_width: int = 600
    base_height: int = 1000
    
    @property
    def total_dimensions(self) -> Tuple[int, int]:
        """Total width and height of the final view."""
        width = self.base_width + 2 * self.horizontal_extension
        height = self.base_height + 2 * self.vertical_extension
        return width, height
    
    @property
    def car_boundaries(self) -> Tuple[int, int, int, int]:
        """Car boundary coordinates: (x_left, x_right, y_top, y_bottom)."""
        total_w, total_h = self.total_dimensions
        
        x_left = self.horizontal_extension + 180 + self.car_horizontal_gap
        x_right = total_w - x_left
        y_top = self.vertical_extension + 200 + self.car_vertical_gap
        y_bottom = total_h - y_top
        
        return x_left, x_right, y_top, y_bottom


@dataclass(frozen=True)
class ProjectionTargets:
    """Target points for projection mapping calibration."""
    
    dimensions: ViewDimensions
    
    @property
    def keypoints(self) -> Dict[str, List[Tuple[int, int]]]:
        """Calibration keypoints for each camera view."""
        h_ext = self.dimensions.horizontal_extension
        v_ext = self.dimensions.vertical_extension
        
        return {
            "front": [
                (h_ext + 120, v_ext),
                (h_ext + 480, v_ext),
                (h_ext + 120, v_ext + 160),
                (h_ext + 480, v_ext + 160)
            ],
            "back": [
                (h_ext + 120, v_ext),
                (h_ext + 480, v_ext),
                (h_ext + 120, v_ext + 160),
                (h_ext + 480, v_ext + 160)
            ],
            "left": [
                (v_ext + 280, h_ext),
                (v_ext + 840, h_ext),
                (v_ext + 280, h_ext + 160),
                (v_ext + 840, h_ext + 160)
            ],
            "right": [
                (v_ext + 160, h_ext),
                (v_ext + 720, h_ext),
                (v_ext + 160, h_ext + 160),
                (v_ext + 720, h_ext + 160)
            ]
        }


class SystemConfig:
    """Main system configuration singleton."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Core configuration
        self.dimensions = ViewDimensions()
        self.projection_targets = ProjectionTargets(self.dimensions)
        
        # Camera configurations
        self.cameras = {
            name: CameraConfig(name=name, position=name)
            for name in ["front", "back", "left", "right"]
        }
        
        # Projection shapes for each camera
        total_w, total_h = self.dimensions.total_dimensions
        x_left, x_right, y_top, y_bottom = self.dimensions.car_boundaries
        
        self.projection_shapes = {
            "front": (total_w, y_top),
            "back": (total_w, y_top),
            "left": (total_h, x_left),
            "right": (total_h, x_left)
        }
        
        # Car image (load if available)
        self.car_image = self._load_car_image()
        
        self._initialized = True
    
    def _load_car_image(self) -> Optional[np.ndarray]:
        """Load and resize car overlay image."""
        try:
            # Try new location first: images/car/car.png
            car_path = os.path.join(os.getcwd(), "images", "car", "car.png")
            if not os.path.exists(car_path):
                # Fallback to old location: images/car.png
                car_path = os.path.join(os.getcwd(), "images", "car.png")
            
            if os.path.exists(car_path):
                image = cv2.imread(car_path)
                if image is not None:
                    # Validate the image
                    if image.size == 0:
                        print(f"⚠️  Car image is empty: {car_path}")
                        return None
                    
                    if len(image.shape) != 3 or image.shape[2] != 3:
                        print(f"⚠️  Car image has invalid format: {image.shape}")
                        return None
                    
                    # Get car boundaries
                    x_left, x_right, y_top, y_bottom = self.dimensions.car_boundaries
                    target_width = x_right - x_left
                    target_height = y_bottom - y_top
                    
                    if target_width <= 0 or target_height <= 0:
                        print(f"⚠️  Invalid car boundaries: {self.dimensions.car_boundaries}")
                        return None
                    
                    # Resize the image to fit car region
                    resized_image = cv2.resize(image, (target_width, target_height))
                    
                    # Validate the resized image
                    if resized_image is None or resized_image.size == 0:
                        print(f"⚠️  Car image resize failed")
                        return None
                    
                    print(f"✅ Car overlay loaded: {car_path}")
                    print(f"   Original size: {image.shape[1]}x{image.shape[0]}")
                    print(f"   Resized to: {target_width}x{target_height}")
                    
                    return resized_image
                else:
                    print(f"⚠️  Car image file exists but couldn't be read: {car_path}")
            else:
                print(f"⚠️  Car overlay not found at: {car_path}")
        except Exception as e:
            print(f"❌ Could not load car image: {e}")
        return None
    
    def reload_car_image(self) -> bool:
        """Reload car overlay image if it was lost."""
        try:
            new_car_image = self._load_car_image()
            if new_car_image is not None:
                self.car_image = new_car_image
                print("✅ Car overlay reloaded successfully")
                return True
            else:
                print("❌ Car overlay reload failed")
                return False
        except Exception as e:
            print(f"❌ Car overlay reload error: {e}")
            return False
    
    @property
    def camera_names(self) -> List[str]:
        """List of all camera names."""
        return list(self.cameras.keys())
    
    def get_camera_config(self, name: str) -> CameraConfig:
        """Get configuration for specific camera."""
        if name not in self.cameras:
            raise ValueError(f"Unknown camera: {name}. Available: {self.camera_names}")
        return self.cameras[name] 