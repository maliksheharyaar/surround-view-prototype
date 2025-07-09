#!/usr/bin/env python3
"""
360° Surround View - Projection Mapping Tool
===========================================

Modern implementation of interactive projection matrix calibration.
Enhanced with real-time parameter adjustment sliders for optimal calibration.

Enhanced with:
- Interactive parameter adjustment sliders
- Real-time preview of undistortion parameters
- Brightness and contrast controls
- Two-phase calibration process

Usage:
    python create_projection_maps.py --camera front
    python create_projection_maps.py --camera front --scale 0.7 0.8 --shift -150 -100
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from surround_view import FisheyeCamera, EnhancedPointSelector, ImageDisplay, SystemConfig
from surround_view.config import CameraConfig
from surround_view.gui_tools import UserAction, validate_rectangle_points


class InteractiveProjectionMapper:
    """
    Advanced interactive tool for creating camera projection matrices.
    
    Features:
    - Two-phase calibration process (parameter adjustment + point selection)
    - Real-time slider controls for image optimization
    - Visual feedback and validation
    - Automatic parameter saving
    - Professional user interface
    """
    
    def __init__(self, camera_name: str):
        """
        Initialize interactive projection mapper.
        
        Args:
            camera_name: Name of camera to calibrate
        """
        self.config = SystemConfig()
        
        if camera_name not in self.config.camera_names:
            raise ValueError(f"Unknown camera: {camera_name}")
        
        self.camera_config = self.config.get_camera_config(camera_name)
        self.camera: Optional[FisheyeCamera] = None
        self._load_camera()
    
    def _load_camera(self) -> None:
        """Load camera model and validate calibration."""
        try:
            self.camera = FisheyeCamera(self.camera_config)
            print(f"✓ Loaded camera calibration for {self.camera_config.name}")
            
            if not self.camera.is_calibrated:
                raise RuntimeError("Camera calibration is incomplete")
                
        except Exception as e:
            print(f"✗ Failed to load camera: {e}")
            sys.exit(1)
    
    def set_initial_parameters(self, scale: Tuple[float, float], shift: Tuple[float, float]) -> None:
        """
        Set initial processing parameters (will be overridden by interactive adjustment).
        
        Args:
            scale: Initial horizontal and vertical scaling factors
            shift: Initial horizontal and vertical shift offsets
        """
        if self.camera:
            self.camera.set_scale_and_shift(scale, shift)
            print(f"✓ Set initial parameters: scale={scale}, shift={shift}")
            print("  Note: These will be adjustable via interactive sliders")
    
    def load_test_image(self) -> np.ndarray:
        """
        Load camera test image.
        
        Returns:
            Test image
            
        Raises:
            FileNotFoundError: If test image cannot be loaded
        """
        image_path = self.camera_config.test_image
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        print(f"✓ Loaded test image: {image_path}")
        return image
    
    def interactive_calibration_selection(self, test_image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """
        Run interactive calibration point selection with parameter adjustment.
        
        Args:
            test_image: Raw camera test image
            
        Returns:
            Tuple of (projection_matrix, (final_scale, final_shift)) or None if cancelled
        """
        print(f"\n🎛️  Interactive Calibration for {self.camera_config.name.upper()} Camera")
        print("=" * 60)
        print("Phase 1: Parameter Adjustment")
        print("  • Use sliders to optimize image appearance")
        print("  • Adjust scale, shift, brightness, and contrast")
        print("  • Press SPACE when image looks optimal for point selection")
        print()
        print("Phase 2: Point Selection") 
        print("  • Click 4 calibration points in the correct order")
        print("  • Points should form a rectangle on the ground pattern")
        print("  • Use right-click to remove incorrect points")
        print()
        
        target_points = self.config.projection_targets.keypoints[self.camera_config.name]
        
        try:
            # Create enhanced point selector with parameter adjustment
            selector = EnhancedPointSelector(
                original_image=test_image,
                camera_processor=self.camera,
                title=f"{self.camera_config.name.title()} Camera Calibration",
                target_points=4,
                validation_callback=validate_rectangle_points,
                camera_name=self.camera_config.name
            )
            
            # Run interactive selection process
            result = selector.run_interactive_selection()
            
            if result is None:
                print("❌ Calibration cancelled by user")
                return None
            
            selected_points, (final_scale, final_shift) = result
            
            print(f"✓ Selected {len(selected_points)} calibration points")
            print(f"✓ Final parameters: scale={final_scale}, shift={final_shift}")
            
            # Calculate projection matrix
            src_points = np.float32(selected_points)
            dst_points = np.float32(target_points)
            
            projection_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            print("✓ Calculated projection matrix")
            return projection_matrix, (final_scale, final_shift)
            
        except Exception as e:
            print(f"❌ Interactive calibration failed: {e}")
            return None
    
    def preview_projection(self, test_image: np.ndarray, 
                          projection_matrix: np.ndarray,
                          final_parameters: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        Show projection preview with final parameters and get user confirmation.
        
        Args:
            test_image: Original test image
            projection_matrix: Calculated projection matrix
            final_parameters: Final scale and shift parameters
            
        Returns:
            True if user confirms, False otherwise
        """
        try:
            # Apply final parameters to camera
            final_scale, final_shift = final_parameters
            self.camera.set_scale_and_shift(final_scale, final_shift)
            
            # Apply projection matrix and process image
            self.camera.set_projection_matrix(projection_matrix)
            
            # Process the full pipeline
            processed_image = self.camera.process_image(test_image)
            
            print("\n👁️  Final Projection Preview")
            print("   Review the complete bird's eye view transformation")
            print(f"   Using: scale={final_scale}, shift={final_shift}")
            
            action = ImageDisplay.show_image(
                f"{self.camera_config.name.title()} - Final Bird's Eye View",
                processed_image,
                show_help=True,
                max_size=(1200, 800)
            )
            
            return action == UserAction.CONFIRM
            
        except Exception as e:
            print(f"❌ Preview failed: {e}")
            return False
    
    def save_calibration_results(self, final_parameters: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """
        Save the projection matrix and final parameters to camera calibration file.
        
        Args:
            final_parameters: Final scale and shift parameters
            
        Returns:
            True if saved successfully
        """
        try:
            # Make sure camera has the final parameters
            final_scale, final_shift = final_parameters
            self.camera.set_scale_and_shift(final_scale, final_shift)
            
            # Save calibration (includes projection matrix and scale/shift parameters)
            self.camera.save_calibration()
            
            print(f"✅ Saved calibration results:")
            print(f"   • File: {self.camera_config.yaml_file}")
            print(f"   • Projection matrix: ✓")
            print(f"   • Scale parameters: {final_scale}")
            print(f"   • Shift parameters: {final_shift}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save calibration: {e}")
            return False
    
    def run_interactive_calibration(self) -> bool:
        """
        Run the complete interactive calibration workflow.
        
        Returns:
            True if calibration completed successfully
        """
        try:
            # Load test image
            test_image = self.load_test_image()
            
            # Interactive calibration with parameter adjustment
            result = self.interactive_calibration_selection(test_image)
            if result is None:
                return False
            
            projection_matrix, final_parameters = result
            
            # Preview final result and get confirmation
            if not self.preview_projection(test_image, projection_matrix, final_parameters):
                print("❌ Final result not confirmed")
                return False
            
            # Save calibration results
            return self.save_calibration_results(final_parameters)
            
        except Exception as e:
            print(f"❌ Interactive calibration failed: {e}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive projection matrix calibration with parameter adjustment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --camera front                    # Interactive calibration for front camera
  %(prog)s --camera back                     # Interactive calibration for back camera
  %(prog)s --camera left --scale 0.8 0.9     # With initial scale suggestion
  %(prog)s --camera right --shift -100 -50   # With initial shift suggestion

Interactive Process:
  1. Adjust parameters using sliders for optimal image quality
  2. Select 4 calibration points on the ground pattern
  3. Review the bird's eye view projection
  4. Confirm and save the calibration
        """
    )
    
    parser.add_argument(
        "--camera", 
        required=True,
        choices=["front", "back", "left", "right"],
        help="Camera to calibrate"
    )
    
    parser.add_argument(
        "--scale", 
        nargs=2, 
        type=float, 
        default=[1.0, 1.0],
        metavar=("X", "Y"),
        help="Initial image scaling factors (adjustable via sliders)"
    )
    
    parser.add_argument(
        "--shift", 
        nargs=2, 
        type=float, 
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Initial image shift offsets (adjustable via sliders)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    print("🚗 360° Surround View - Interactive Projection Mapping")
    print("=" * 55)
    print("✨ Enhanced with real-time parameter adjustment sliders!")
    print()
    
    try:
        args = parse_arguments()
        
        # Create interactive mapper
        mapper = InteractiveProjectionMapper(args.camera)
        
        # Set initial parameters (these will be adjustable)
        mapper.set_initial_parameters(
            scale=tuple(args.scale),
            shift=tuple(args.shift)
        )
        
        print(f"🎯 Starting interactive calibration for {args.camera.upper()} camera")
        print("   This process has two phases:")
        print("   1️⃣  Parameter adjustment with sliders")
        print("   2️⃣  Interactive point selection")
        print()
        
        # Run interactive calibration
        if mapper.run_interactive_calibration():
            print("\n🎉 Interactive projection mapping completed successfully!")
            print("   Your camera is now calibrated and ready for surround view!")
            return 0
        else:
            print("\n💥 Interactive projection mapping failed or cancelled")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1
    finally:
        # Safely close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Allow time for windows to close
        except:
            pass


if __name__ == "__main__":
    sys.exit(main()) 