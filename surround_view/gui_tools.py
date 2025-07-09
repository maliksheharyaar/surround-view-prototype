"""
GUI Tools for Interactive Calibration
=====================================

Modern, user-friendly GUI components for interactive point selection and image display.
Enhanced with real-time parameter adjustment sliders and improved user experience.
"""

from typing import List, Tuple, Optional, Callable
from enum import Enum
import cv2
import numpy as np
from .preset_manager import load_calibration_preset, save_calibration_preset, has_calibration_preset


class UserAction(Enum):
    """User action types for GUI interactions."""
    QUIT = -1
    CONTINUE = 0
    CONFIRM = 1


class ImageDisplay:
    """
    Enhanced image display utility with better error handling.
    
    Features:
    - Automatic window management
    - Keyboard shortcut help
    - Error recovery
    """
    
    @staticmethod
    def show_image(title: str, 
                   image: np.ndarray, 
                   show_help: bool = True,
                   max_size: Optional[Tuple[int, int]] = None) -> UserAction:
        """
        Display image and wait for user input.
        
        Args:
            title: Window title
            image: Image to display
            show_help: Whether to show keyboard shortcuts
            max_size: Maximum window size (width, height)
            
        Returns:
            UserAction indicating user choice
        """
        if image is None or image.size == 0:
            print("Error: Cannot display empty image")
            return UserAction.QUIT
        
        display_image = image.copy()
        
        # Resize if too large
        if max_size:
            h, w = display_image.shape[:2]
            max_w, max_h = max_size
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cv2.resize(display_image, (new_w, new_h))
        
        # Add help text overlay
        if show_help:
            help_text = [
                "Controls:",
                "  ENTER - Confirm/Accept",
                "  Q - Quit/Cancel", 
                "  ESC - Close window"
            ]
            
            y_offset = 30
            for i, text in enumerate(help_text):
                cv2.putText(display_image, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        try:
            cv2.imshow(title, display_image)
            
            while True:
                # Check if window was closed
                if cv2.getWindowProperty(title, cv2.WND_PROP_AUTOSIZE) < 0:
                    return UserAction.QUIT
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    return UserAction.QUIT
                elif key == 13:  # Enter
                    return UserAction.CONFIRM
                elif key == 27:  # Escape
                    cv2.destroyWindow(title)
                    return UserAction.QUIT
                    
        except Exception as e:
            print(f"Display error: {e}")
            return UserAction.QUIT
        
        finally:
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.destroyWindow(title)
            except cv2.error:
                pass  # Window doesn't exist or already destroyed


class ParameterAdjuster:
    """
    Interactive parameter adjustment with sliders for real-time image processing.
    
    Features:
    - Real-time scale and shift adjustment
    - Brightness and contrast controls
    - Live preview updates
    - Parameter reset functionality
    """
    
    def __init__(self, 
                 initial_scale: Tuple[float, float] = (0.6, 0.6),  # More zoomed out
                 initial_shift: Tuple[float, float] = (0.0, 0.0),
                 initial_distortion: Tuple[float, float, float, float] = (-0.29, 0.11, -0.0003, 0.003)):
        """
        Initialize parameter adjuster.
        
        Args:
            initial_scale: Initial horizontal and vertical scaling
            initial_shift: Initial horizontal and vertical shift
            initial_distortion: Initial fisheye distortion coefficients (k1, k2, k3, k4)
        """
        self.scale_x, self.scale_y = initial_scale
        self.shift_x, self.shift_y = initial_shift 
        self.k1, self.k2, self.k3, self.k4 = initial_distortion
        
        # Track if parameters changed
        self.parameters_changed = False
        
        # Preset saving options
        self.save_as_preset = False
        self.apply_to_all_cameras = False
        
        # Slider ranges
        self.scale_range = (0.1, 2.0)
        self.shift_range = (-500, 500)
        self.distortion_range = (-1.0, 1.0)  # Typical range for distortion coefficients
        
        # Slider precision multipliers (for integer sliders)
        self.scale_precision = 100  # 0.01 precision
        self.shift_precision = 1    # 1 pixel precision
        self.distortion_precision = 1000  # 0.001 precision for fine distortion control
    
    def create_sliders(self, window_name: str) -> None:
        """Create trackbars for parameter adjustment."""
        # Scale sliders
        cv2.createTrackbar(
            'ScaleX', window_name,
            int(self.scale_x * self.scale_precision),
            int(self.scale_range[1] * self.scale_precision),
            self._on_scale_x_change
        )
        
        cv2.createTrackbar(
            'ScaleY', window_name,
            int(self.scale_y * self.scale_precision),
            int(self.scale_range[1] * self.scale_precision),
            self._on_scale_y_change
        )
        
        # Shift sliders (offset to handle negative values)
        shift_offset = abs(self.shift_range[0])
        cv2.createTrackbar(
            'ShiftX', window_name,
            int(self.shift_x + shift_offset),
            int(self.shift_range[1] + shift_offset),
            self._on_shift_x_change
        )
        
        cv2.createTrackbar(
            'ShiftY', window_name,
            int(self.shift_y + shift_offset),
            int(self.shift_range[1] + shift_offset),
            self._on_shift_y_change
        )
        
        # Preset saving checkbox (0 = unchecked, 1 = checked)
        cv2.createTrackbar(
            'Save', window_name,
            0,  # Initially unchecked
            1,  # Max value (checkbox)
            self._on_preset_save_change
        )
        
        # Apply to all cameras checkbox
        cv2.createTrackbar(
            'All', window_name,
            0,  # Initially unchecked
            1,  # Max value (checkbox)
            self._on_apply_all_change
        )
        
        # Distortion coefficient sliders (offset to handle negative values)
        distortion_offset = abs(self.distortion_range[0])
        
        cv2.createTrackbar(
            'K1', window_name,
            int((self.k1 + distortion_offset) * self.distortion_precision),
            int((self.distortion_range[1] + distortion_offset) * self.distortion_precision),
            self._on_k1_change
        )
        
        cv2.createTrackbar(
            'K2', window_name,
            int((self.k2 + distortion_offset) * self.distortion_precision),
            int((self.distortion_range[1] + distortion_offset) * self.distortion_precision),
            self._on_k2_change
        )
        
        cv2.createTrackbar(
            'K3', window_name,
            int((self.k3 + distortion_offset) * self.distortion_precision),
            int((self.distortion_range[1] + distortion_offset) * self.distortion_precision),
            self._on_k3_change
        )
        
        cv2.createTrackbar(
            'K4', window_name,
            int((self.k4 + distortion_offset) * self.distortion_precision),
            int((self.distortion_range[1] + distortion_offset) * self.distortion_precision),
            self._on_k4_change
        )
    
    def _on_scale_x_change(self, value: int) -> None:
        """Handle scale X slider change."""
        new_value = max(self.scale_range[0], value / self.scale_precision)
        if abs(new_value - self.scale_x) > 0.001:
            self.scale_x = new_value
            self.parameters_changed = True
    
    def _on_scale_y_change(self, value: int) -> None:
        """Handle scale Y slider change."""
        new_value = max(self.scale_range[0], value / self.scale_precision)
        if abs(new_value - self.scale_y) > 0.001:
            self.scale_y = new_value
            self.parameters_changed = True
    
    def _on_shift_x_change(self, value: int) -> None:
        """Handle shift X slider change."""
        shift_offset = abs(self.shift_range[0])
        new_value = value - shift_offset
        if abs(new_value - self.shift_x) > 0.5:
            self.shift_x = float(new_value)
            self.parameters_changed = True
    
    def _on_shift_y_change(self, value: int) -> None:
        """Handle shift Y slider change."""
        shift_offset = abs(self.shift_range[0])
        new_value = value - shift_offset
        if abs(new_value - self.shift_y) > 0.5:
            self.shift_y = float(new_value)
            self.parameters_changed = True
    
    def _on_preset_save_change(self, value: int) -> None:
        """Handle preset save checkbox change."""
        self.save_as_preset = bool(value)
    
    def _on_apply_all_change(self, value: int) -> None:
        """Handle apply to all cameras checkbox change."""
        self.apply_to_all_cameras = bool(value)
    
    def _on_k1_change(self, value: int) -> None:
        """Handle K1 distortion coefficient slider change."""
        distortion_offset = abs(self.distortion_range[0])
        new_value = (value / self.distortion_precision) - distortion_offset
        new_value = max(self.distortion_range[0], min(self.distortion_range[1], new_value))
        if abs(new_value - self.k1) > 0.001:
            self.k1 = new_value
            self.parameters_changed = True
    
    def _on_k2_change(self, value: int) -> None:
        """Handle K2 distortion coefficient slider change."""
        distortion_offset = abs(self.distortion_range[0])
        new_value = (value / self.distortion_precision) - distortion_offset
        new_value = max(self.distortion_range[0], min(self.distortion_range[1], new_value))
        if abs(new_value - self.k2) > 0.001:
            self.k2 = new_value
            self.parameters_changed = True
    
    def _on_k3_change(self, value: int) -> None:
        """Handle K3 distortion coefficient slider change."""
        distortion_offset = abs(self.distortion_range[0])
        new_value = (value / self.distortion_precision) - distortion_offset
        new_value = max(self.distortion_range[0], min(self.distortion_range[1], new_value))
        if abs(new_value - self.k3) > 0.001:
            self.k3 = new_value
            self.parameters_changed = True
    
    def _on_k4_change(self, value: int) -> None:
        """Handle K4 distortion coefficient slider change."""
        distortion_offset = abs(self.distortion_range[0])
        new_value = (value / self.distortion_precision) - distortion_offset
        new_value = max(self.distortion_range[0], min(self.distortion_range[1], new_value))
        if abs(new_value - self.k4) > 0.001:
            self.k4 = new_value
            self.parameters_changed = True
    

    
    def get_scale_shift(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get current scale and shift parameters."""
        return (self.scale_x, self.scale_y), (self.shift_x, self.shift_y)
    
    def get_distortion_coefficients(self) -> Tuple[float, float, float, float]:
        """Get current distortion coefficients."""
        return (self.k1, self.k2, self.k3, self.k4)
    
    def get_all_parameters(self) -> dict:
        """Get all current parameters."""
        return {
            'scale': (self.scale_x, self.scale_y),
            'shift': (self.shift_x, self.shift_y),
            'distortion': (self.k1, self.k2, self.k3, self.k4)
        }
    
    def reset_parameters(self) -> None:
        """Reset all parameters to defaults."""
        self.scale_x, self.scale_y = 0.6, 0.6  # Use the improved default
        self.shift_x, self.shift_y = 0.0, 0.0
        self.k1, self.k2, self.k3, self.k4 = (-0.29, 0.11, -0.0003, 0.003)  # Reset to original values
        self.save_as_preset = False
        self.apply_to_all_cameras = False
        self.parameters_changed = True


class EnhancedPointSelector:
    """
    Advanced point selection tool with real-time parameter adjustment.
    
    Features:
    - Interactive sliders for image parameter adjustment
    - Real-time preview updates
    - Visual feedback for selections
    - Undo functionality
    - Point validation
    - Progress tracking
    """
    
    # Visual styling constants
    POINT_COLOR = (0, 0, 255)      # Red for points
    SELECTED_COLOR = (0, 255, 0)   # Green for completed selections
    HULL_COLOR = (0, 255, 255)     # Yellow for convex hull
    TEXT_COLOR = (255, 255, 255)   # White for text
    
    def __init__(self, 
                 original_image: np.ndarray,
                 camera_processor,  # FisheyeCamera instance
                 title: str = "Enhanced Point Selection",
                 target_points: int = 4,
                 validation_callback: Optional[Callable[[List[Tuple[int, int]]], bool]] = None,
                 camera_name: str = "unknown"):
        """
        Initialize enhanced point selector with parameter adjustment.
        
        Args:
            original_image: Raw camera image
            camera_processor: Camera model for real-time processing
            title: Window title
            target_points: Expected number of points to select
            validation_callback: Optional function to validate point selection
            camera_name: Name of the camera for preset management
        """
        if original_image is None or original_image.size == 0:
            raise ValueError("Invalid image provided")
        
        self.original_image = original_image.copy()
        self.camera = camera_processor
        self.title = title
        self.target_points = target_points
        self.validation_callback = validation_callback
        self.camera_name = camera_name
        
        # Point selection state
        self.selected_points: List[Tuple[int, int]] = []
        self._is_selection_complete = False
        
        # Parameter adjustment - initialize with preset if available
        initial_distortion = self.camera.get_distortion_coefficients()
        preset = load_calibration_preset(camera_name)
        
        if preset:
            print(f"🎯 Found preset for {camera_name} camera - loading saved values...")
            self.adjuster = ParameterAdjuster(
                initial_scale=preset['scale'],
                initial_shift=preset['shift'],
                initial_distortion=preset['distortion']
            )
        else:
            print(f"ℹ️  No preset found for {camera_name} camera - using defaults")
            self.adjuster = ParameterAdjuster(initial_distortion=initial_distortion)
        
        self.current_processed_image = None
        
        # Setup windows
        self.main_window = f"{title} - Main"
        self.controls_window = f"{title} - Controls"
        
        self._setup_windows()
    
    def _setup_windows(self) -> None:
        """Setup main and control windows."""
        # Main image window
        cv2.namedWindow(self.main_window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.main_window, self._mouse_callback)
        
        # Controls window with sliders (resizable for better fit)
        cv2.namedWindow(self.controls_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.controls_window, 500, 600)  # Set initial size
        self.adjuster.create_sliders(self.controls_window)
        
        # Create larger control image to accommodate all sliders
        control_image = np.zeros((600, 500, 3), np.uint8)
        
        # Add compact instructions
        instructions = [
            "Parameter Controls",
            "",
            "Scale X/Y: Zoom",
            "Shift X/Y: Center", 
            "K1-K4: Distortion",
            "Save/Apply: Presets",
            "",
            "R: Reset | Space: Continue"
        ]
        
        y_start = 15
        for i, text in enumerate(instructions):
            color = (255, 255, 255) if text else (100, 100, 100)
            if text:
                cv2.putText(control_image, text, (10, y_start + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow(self.controls_window, control_image)
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._remove_last_point()
    
    def _add_point(self, x: int, y: int) -> None:
        """Add a new point to the selection."""
        if len(self.selected_points) >= self.target_points:
            print(f"Maximum {self.target_points} points allowed")
            return
        
        # Validate point is within image bounds
        if self.current_processed_image is not None:
            h, w = self.current_processed_image.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                print("Point outside image bounds")
                return
        
        self.selected_points.append((x, y))
        print(f"Added point {len(self.selected_points)}: ({x}, {y})")
        
        # Check if selection is complete
        if len(self.selected_points) == self.target_points:
            if self.validation_callback is None or self.validation_callback(self.selected_points):
                self._is_selection_complete = True
                print("Selection complete! Press ENTER to confirm or continue selecting.")
        
        self._update_display()
    
    def _remove_last_point(self) -> None:
        """Remove the last selected point."""
        if self.selected_points:
            removed = self.selected_points.pop()
            print(f"Removed point: {removed}")
            self._is_selection_complete = False
            self._update_display()
        else:
            print("No points to remove")
    
    def _process_image(self) -> np.ndarray:
        """Process image with current parameters."""
        try:
            # Update camera parameters
            scale, shift = self.adjuster.get_scale_shift()
            distortion = self.adjuster.get_distortion_coefficients()
            
            self.camera.set_scale_and_shift(scale, shift)
            self.camera.set_distortion_coefficients(distortion)
            
            # Undistort image
            undistorted = self.camera.undistort(self.original_image)
            
            return undistorted
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return self.original_image
    
    def _update_display(self) -> None:
        """Update the main display with current point selection and processed image."""
        # Reprocess image if parameters changed
        if self.adjuster.parameters_changed or self.current_processed_image is None:
            self.current_processed_image = self._process_image()
            self.adjuster.parameters_changed = False
        
        display_image = self.current_processed_image.copy()
        
        # Draw selected points directly on the image
        for i, (x, y) in enumerate(self.selected_points):
            color = self.SELECTED_COLOR if self._is_selection_complete else self.POINT_COLOR
            cv2.circle(display_image, (x, y), 8, color, -1)
            cv2.circle(display_image, (x, y), 10, self.TEXT_COLOR, 2)
            
            # Add point number
            cv2.putText(display_image, str(i + 1), (x - 5, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
        
        # Draw connections between points
        if len(self.selected_points) >= 2:
            points = np.array(self.selected_points, dtype=np.int32)
            
            if len(self.selected_points) == 2:
                cv2.line(display_image, tuple(points[0]), tuple(points[1]), 
                        self.HULL_COLOR, 2)
            elif len(self.selected_points) > 2:
                hull = cv2.convexHull(points)
                cv2.polylines(display_image, [hull], True, self.HULL_COLOR, 2)
                cv2.fillPoly(display_image, [hull], (*self.HULL_COLOR, 30))
        
        # Add current parameter values
        scale, shift = self.adjuster.get_scale_shift()
        distortion = self.adjuster.get_distortion_coefficients()
        param_text = [
            f"ScaleX/Y: {scale[0]:.2f}, {scale[1]:.2f}",
            f"ShiftX/Y: {shift[0]:.0f}, {shift[1]:.0f}",
            f"K1: {distortion[0]:.3f} | K2: {distortion[1]:.3f}",
            f"K3: {distortion[2]:.4f} | K4: {distortion[3]:.4f}",
            f"Save: {'✓' if self.adjuster.save_as_preset else '✗'} | All: {'✓' if self.adjuster.apply_to_all_cameras else '✗'}",
            "",
            f"Points: {len(self.selected_points)}/{self.target_points}",
            "Click to select calibration points",
            "Use K1-K4 for distortion",
            "Checkboxes for presets",
            "L-click: Add | R-click: Remove",
            "R: Reset | SPACE: Continue",
            "ENTER: Confirm | Q: Quit",
            "Complete" if self._is_selection_complete else "In Progress"
        ]
        
        # Add semi-transparent background for text (compact area)
        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
        
        y_start = 25
        for i, text in enumerate(param_text):
            if not text:
                continue
            color = self.SELECTED_COLOR if i == len(param_text) - 1 and self._is_selection_complete else self.TEXT_COLOR
            cv2.putText(display_image, text, (15, y_start + i * 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        cv2.imshow(self.main_window, display_image)
    
    def _safe_destroy_windows(self) -> None:
        """Safely destroy OpenCV windows without throwing exceptions."""
        try:
            # Check if main window exists before destroying
            if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.destroyWindow(self.main_window)
        except cv2.error:
            pass  # Window doesn't exist or already destroyed
        
        try:
            # Check if controls window exists before destroying
            if cv2.getWindowProperty(self.controls_window, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.destroyWindow(self.controls_window)
        except cv2.error:
            pass  # Window doesn't exist or already destroyed
    
    def adjust_parameters(self) -> bool:
        """
        Run parameter adjustment phase.
        
        Returns:
            True if user proceeded to point selection, False if cancelled
        """
        print("🎛️  Parameter Adjustment Phase")
        print("   Use sliders to optimize image appearance for point selection")
        print("   💡 Adjust SCALE/SHIFT to position ground pattern in center area")
        print("   Press SPACE when ready to select points, R to reset, Q to quit")
        
        self._update_display()
        
        while True:
            # Check if windows were closed
            if (cv2.getWindowProperty(self.main_window, cv2.WND_PROP_AUTOSIZE) < 0 or
                cv2.getWindowProperty(self.controls_window, cv2.WND_PROP_AUTOSIZE) < 0):
                return False
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Parameter adjustment cancelled")
                return False
            elif key == ord('r') or key == ord('R'):
                print("Resetting parameters...")
                self.adjuster.reset_parameters()
                self._update_display()
            elif key == 32:  # Space
                print("Proceeding to point selection...")
                # Close controls window safely
                try:
                    if cv2.getWindowProperty(self.controls_window, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.destroyWindow(self.controls_window)
                except cv2.error:
                    pass  # Window doesn't exist or already destroyed
                return True
            elif key == 27:  # Escape
                print("Parameter adjustment cancelled")
                return False
            
            # Update display if parameters changed
            if self.adjuster.parameters_changed:
                self._update_display()
    
    def select_points(self) -> Optional[List[Tuple[int, int]]]:
        """
        Run the point selection interface.
        
        Returns:
            List of selected points or None if cancelled
        """
        print("🎯 Point Selection Phase")
        print(f"   Select {self.target_points} points by clicking on the image")
        print("   💡 Choose ground pattern corners for best calibration results")
        print("   Right-click or press 'D' to remove the last point")
        print("   Press ENTER when done, 'Q' to quit")
        
        self._update_display()
        
        while True:
            # Check if window was closed
            if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_AUTOSIZE) < 0:
                return None
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Selection cancelled")
                return None
            elif key == ord('d') or key == ord('D'):
                self._remove_last_point()
            elif key == 13:  # Enter
                if len(self.selected_points) >= self.target_points:
                    print("Selection confirmed")
                    return self.selected_points.copy()
                else:
                    print(f"Need {self.target_points} points, only have {len(self.selected_points)}")
            elif key == 27:  # Escape
                print("Selection cancelled")
                return None
    
    def run_interactive_selection(self) -> Optional[Tuple[List[Tuple[int, int]], Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """
        Run the complete interactive selection process.
        
        Returns:
            Tuple of (selected_points, (scale, shift)) or None if cancelled
        """
        try:
            # Phase 1: Parameter adjustment
            if not self.adjust_parameters():
                return None
            
            # Phase 2: Point selection
            points = self.select_points()
            if points is None:
                return None
            
            # Save preset if requested
            if self.adjuster.save_as_preset:
                params = self.adjuster.get_all_parameters()
                apply_to_all = self.adjuster.apply_to_all_cameras
                
                if save_calibration_preset(self.camera_name, params, apply_to_all):
                    if apply_to_all:
                        print(f"💾 Universal calibration preset saved for ALL cameras (based on {self.camera_name})")
                    else:
                        print(f"💾 Calibration preset saved for {self.camera_name} camera only")
                else:
                    print(f"⚠️  Failed to save preset for {self.camera_name} camera")
            
            # Return points and final parameters
            scale, shift = self.adjuster.get_scale_shift()
            return points, (scale, shift)
            
        finally:
            self._safe_destroy_windows()


# Backward compatibility alias
PointSelector = EnhancedPointSelector


def validate_rectangle_points(points: List[Tuple[int, int]]) -> bool:
    """
    Validate that 4 points form a reasonable rectangle.
    
    Args:
        points: List of 4 points
        
    Returns:
        True if points form a valid rectangle
    """
    if len(points) != 4:
        return False
    
    try:
        # Convert to numpy array
        pts = np.array(points, dtype=np.float32)
        
        # Check if points are roughly rectangular by computing area
        hull = cv2.convexHull(pts)
        area = cv2.contourArea(hull)
        
        # Check minimum area threshold
        if area < 1000:  # Minimum area requirement
            return False
        
        # Check that points aren't too close together
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if dist < 20:  # Minimum distance between points
                    return False
        
        return True
        
    except Exception:
        return False 