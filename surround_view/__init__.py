"""
360° Surround View System - Minimal Implementation
==================================================

A modernized, streamlined implementation of a surround view system focused on:
- Projection mapping with interactive GUI
- Smooth weight matrix generation for seamless stitching
- Clean, maintainable code architecture

Author: Custom Implementation
Version: 2.0.0
"""

from .camera_model import FisheyeCamera
from .gui_tools import ImageDisplay, PointSelector, EnhancedPointSelector
from .view_processor import SurroundViewProcessor
from .config import SystemConfig

__version__ = "2.0.0"
__title__ = "360° Surround View System"
__author__ = "Custom Implementation"
__license__ = "MIT"

# Public API
__all__ = [
    "FisheyeCamera",
    "ImageDisplay", 
    "PointSelector",
    "EnhancedPointSelector",
    "SurroundViewProcessor",
    "SystemConfig",
]
