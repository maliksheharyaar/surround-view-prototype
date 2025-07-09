"""
Fisheye Camera Model
====================

Modern implementation of fisheye camera calibration and image processing
with improved error handling, type hints, and cleaner architecture.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from .config import SystemConfig, CameraConfig


class CameraCalibrationError(Exception):
    """Custom exception for camera calibration errors."""
    pass


class FisheyeCamera:
    """
    Advanced fisheye camera model for distortion correction and projection mapping.
    
    Features:
    - Robust error handling
    - Type safety with hints
    - Configurable scaling and shifting
    - Automatic calibration validation
    """
    
    def __init__(self, camera_config: CameraConfig):
        """
        Initialize fisheye camera model.
        
        Args:
            camera_config: Camera configuration object
            
        Raises:
            CameraCalibrationError: If calibration file cannot be loaded
        """
        self.config = camera_config
        self.system_config = SystemConfig()
        
        # Camera parameters
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.projection_matrix: Optional[np.ndarray] = None
        
        # Undistortion parameters
        self.scale_factors = (1.0, 1.0)
        self.shift_offset = (0, 0)
        self._undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        # Load calibration data
        self._load_calibration()
        
    def _load_calibration(self) -> None:
        """Load camera calibration parameters from YAML file."""
        if not os.path.exists(self.config.yaml_file):
            raise CameraCalibrationError(
                f"Calibration file not found: {self.config.yaml_file}"
            )
        
        try:
            fs = cv2.FileStorage(self.config.yaml_file, cv2.FILE_STORAGE_READ)
            
            # Required parameters
            self.camera_matrix = fs.getNode("camera_matrix").mat()
            self.distortion_coeffs = fs.getNode("dist_coeffs").mat()
            resolution_node = fs.getNode("resolution").mat()
            
            if any(param is None for param in [self.camera_matrix, self.distortion_coeffs, resolution_node]):
                raise CameraCalibrationError("Missing required calibration parameters")
            
            self.resolution = tuple(resolution_node.flatten().astype(int))
            
            # Optional parameters
            scale_node = fs.getNode("scale_xy").mat()
            if scale_node is not None:
                self.scale_factors = tuple(scale_node.flatten())
            
            shift_node = fs.getNode("shift_xy").mat()
            if shift_node is not None:
                self.shift_offset = tuple(shift_node.flatten())
            
            projection_node = fs.getNode("project_matrix").mat()
            if projection_node is not None:
                self.projection_matrix = projection_node
            
            fs.release()
            
        except Exception as e:
            raise CameraCalibrationError(f"Failed to load calibration: {e}")
        
        # Generate undistortion maps
        self._update_undistortion_maps()
    
    def _update_undistortion_maps(self) -> None:
        """Update undistortion maps based on current scale and shift parameters."""
        if self.camera_matrix is None or self.resolution is None:
            return
        
        # Create new camera matrix with scaling and shifting
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_factors[0]  # fx
        new_matrix[1, 1] *= self.scale_factors[1]  # fy
        new_matrix[0, 2] += self.shift_offset[0]   # cx
        new_matrix[1, 2] += self.shift_offset[1]   # cy
        
        width, height = self.resolution
        
        try:
            self._undistort_maps = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coeffs,
                np.eye(3),
                new_matrix,
                (width, height),
                cv2.CV_16SC2
            )
        except cv2.error as e:
            raise CameraCalibrationError(f"Failed to create undistortion maps: {e}")
    
    def set_scale_and_shift(self, 
                           scale: Tuple[float, float] = (1.0, 1.0),
                           shift: Tuple[float, float] = (0, 0)) -> 'FisheyeCamera':
        """
        Set scaling and shifting parameters for undistortion.
        
        Args:
            scale: Horizontal and vertical scaling factors
            shift: Horizontal and vertical shift offsets
            
        Returns:
            Self for method chaining
        """
        self.scale_factors = scale
        self.shift_offset = shift
        self._update_undistortion_maps()
        return self
    
    def set_distortion_coefficients(self, coeffs: Tuple[float, float, float, float]) -> 'FisheyeCamera':
        """
        Set distortion coefficients for fisheye undistortion.
        
        Args:
            coeffs: Four distortion coefficients (k1, k2, k3, k4)
            
        Returns:
            Self for method chaining
        """
        self.distortion_coeffs = np.array(coeffs, dtype=np.float64).reshape(4, 1)
        self._update_undistortion_maps()
        return self
    
    def get_distortion_coefficients(self) -> Tuple[float, float, float, float]:
        """
        Get current distortion coefficients.
        
        Returns:
            Four distortion coefficients (k1, k2, k3, k4)
        """
        if self.distortion_coeffs is None:
            return (0.0, 0.0, 0.0, 0.0)
        coeffs = self.distortion_coeffs.flatten()
        return tuple(coeffs.tolist())
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove fisheye distortion from image.
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
            
        Raises:
            CameraCalibrationError: If undistortion maps are not available
        """
        if self._undistort_maps is None:
            raise CameraCalibrationError("Undistortion maps not available")
        
        try:
            return cv2.remap(
                image, 
                *self._undistort_maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        except cv2.error as e:
            raise CameraCalibrationError(f"Undistortion failed: {e}")
    
    def project_to_birdview(self, image: np.ndarray) -> np.ndarray:
        """
        Project undistorted image to bird's eye view.
        
        Args:
            image: Undistorted input image
            
        Returns:
            Bird's eye view projected image
            
        Raises:
            CameraCalibrationError: If projection matrix is not available
        """
        if self.projection_matrix is None:
            raise CameraCalibrationError(
                f"Projection matrix not available for {self.config.name} camera"
            )
        
        projection_shape = self.system_config.projection_shapes[self.config.name]
        
        try:
            return cv2.warpPerspective(
                image, 
                self.projection_matrix, 
                projection_shape
            )
        except cv2.error as e:
            raise CameraCalibrationError(f"Projection failed: {e}")
    
    def apply_camera_specific_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply camera-specific transformations (rotation/flipping).
        
        Args:
            image: Input image
            
        Returns:
            Transformed image
        """
        camera_name = self.config.name
        
        if camera_name == "front":
            return image.copy()
        elif camera_name == "back":
            return image.copy()[::-1, ::-1, :]  # 180° rotation
        elif camera_name == "left":
            return cv2.transpose(image)[::-1]   # 90° CCW rotation
        elif camera_name == "right":
            return np.flip(cv2.transpose(image), 1)  # 90° CW rotation
        else:
            return image.copy()
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete image processing pipeline: undistort -> project -> transform.
        
        Args:
            image: Raw camera image
            
        Returns:
            Processed bird's eye view image
        """
        undistorted = self.undistort(image)
        projected = self.project_to_birdview(undistorted)
        transformed = self.apply_camera_specific_transform(projected)
        return transformed
    
    def set_projection_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the projection matrix for bird's eye view transformation.
        
        Args:
            matrix: 3x3 projection matrix
        """
        if matrix.shape != (3, 3):
            raise ValueError("Projection matrix must be 3x3")
        self.projection_matrix = matrix
    
    def save_calibration(self) -> None:
        """Save current calibration parameters to YAML file."""
        try:
            fs = cv2.FileStorage(self.config.yaml_file, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", self.camera_matrix)
            fs.write("dist_coeffs", self.distortion_coeffs)
            fs.write("resolution", np.array(self.resolution, dtype=np.int32))
            
            if self.projection_matrix is not None:
                fs.write("project_matrix", self.projection_matrix)
            
            fs.write("scale_xy", np.array(self.scale_factors, dtype=np.float32))
            fs.write("shift_xy", np.array(self.shift_offset, dtype=np.float32))
            fs.release()
            
        except Exception as e:
            raise CameraCalibrationError(f"Failed to save calibration: {e}")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if camera is properly calibrated."""
        return all(param is not None for param in [
            self.camera_matrix, 
            self.distortion_coeffs, 
            self.resolution
        ])
    
    @property
    def has_projection_matrix(self) -> bool:
        """Check if projection matrix is available."""
        return self.projection_matrix is not None
    
    def __str__(self) -> str:
        """String representation of camera model."""
        return (f"FisheyeCamera({self.config.name}, "
                f"calibrated={self.is_calibrated}, "
                f"projection={self.has_projection_matrix})") 