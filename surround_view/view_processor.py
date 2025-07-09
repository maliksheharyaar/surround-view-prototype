"""
360° View Processing Engine
==========================

Advanced image processing pipeline for creating seamless surround view from multiple cameras.
Features modern algorithms for blending, color correction, and artifact reduction.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import cv2
from PIL import Image

from .config import SystemConfig
from .camera_model import FisheyeCamera


class BlendingMethod(Enum):
    """Available blending methods for image stitching."""
    DISTANCE_BASED = "distance"
    MULTI_BAND = "multiband"  
    SIMPLE_AVERAGE = "average"


@dataclass
class ProcessingStats:
    """Statistics from the processing pipeline."""
    processing_time_ms: float
    blend_regions_count: int
    color_correction_applied: bool
    final_image_size: Tuple[int, int]


class ImageRegionProcessor:
    """Utility class for processing different regions of the bird's eye view."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.x_left, self.x_right, self.y_top, self.y_bottom = config.dimensions.car_boundaries
        self.total_w, self.total_h = config.dimensions.total_dimensions
    
    # Region extraction methods
    def front_left(self, image: np.ndarray) -> np.ndarray:
        """Extract front-left region."""
        return image[:, :self.x_left]
    
    def front_right(self, image: np.ndarray) -> np.ndarray:
        """Extract front-right region."""
        return image[:, self.x_right:]
    
    def front_center(self, image: np.ndarray) -> np.ndarray:
        """Extract front-center region."""
        return image[:, self.x_left:self.x_right]
    
    def back_left(self, image: np.ndarray) -> np.ndarray:
        """Extract back-left region."""
        return image[:, :self.x_left]
    
    def back_right(self, image: np.ndarray) -> np.ndarray:
        """Extract back-right region."""
        return image[:, self.x_right:]
    
    def back_center(self, image: np.ndarray) -> np.ndarray:
        """Extract back-center region."""
        return image[:, self.x_left:self.x_right]
    
    def left_top(self, image: np.ndarray) -> np.ndarray:
        """Extract left-top region."""
        return image[:self.y_top, :]
    
    def left_bottom(self, image: np.ndarray) -> np.ndarray:
        """Extract left-bottom region."""
        return image[self.y_bottom:, :]
    
    def left_center(self, image: np.ndarray) -> np.ndarray:
        """Extract left-center region."""
        return image[self.y_top:self.y_bottom, :]
    
    def right_top(self, image: np.ndarray) -> np.ndarray:
        """Extract right-top region."""
        return image[:self.y_top, :]
    
    def right_bottom(self, image: np.ndarray) -> np.ndarray:
        """Extract right-bottom region."""
        return image[self.y_bottom:, :]
    
    def right_center(self, image: np.ndarray) -> np.ndarray:
        """Extract right-center region."""
        return image[self.y_top:self.y_bottom, :]


class AdvancedBlender:
    """Advanced image blending with multiple algorithms."""
    
    @staticmethod
    def create_distance_weights(mask_a: np.ndarray, mask_b: np.ndarray, 
                               threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create distance-based blending weights.
        
        Args:
            mask_a: Binary mask for first image
            mask_b: Binary mask for second image
            threshold: Distance threshold for blending
            
        Returns:
            Weight matrices for images A and B
        """
        # Find overlap region
        overlap = cv2.bitwise_and(mask_a, mask_b)
        overlap_indices = np.where(overlap == 255)
        
        # Initialize weights
        weights_a = (mask_a / 255.0).astype(np.float32)
        weights_b = np.zeros_like(weights_a)
        
        if len(overlap_indices[0]) == 0:
            return weights_a, weights_b
        
        # Find contours for distance calculation
        contours_a, _ = cv2.findContours(
            cv2.bitwise_and(mask_a, cv2.bitwise_not(overlap)),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_b, _ = cv2.findContours(
            cv2.bitwise_and(mask_b, cv2.bitwise_not(overlap)),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours_a or not contours_b:
            return weights_a, weights_b
        
        # Get largest contours
        poly_a = cv2.approxPolyDP(
            max(contours_a, key=cv2.contourArea),
            0.009 * cv2.arcLength(max(contours_a, key=cv2.contourArea), True),
            True
        )
        poly_b = cv2.approxPolyDP(
            max(contours_b, key=cv2.contourArea),
            0.009 * cv2.arcLength(max(contours_b, key=cv2.contourArea), True),
            True
        )
        
        # Calculate distance-based weights
        for y, x in zip(*overlap_indices):
            point = (int(x), int(y))
            
            dist_a = cv2.pointPolygonTest(poly_a, point, True)
            dist_b = cv2.pointPolygonTest(poly_b, point, True)
            
            if dist_b < threshold:
                dist_a_sq = dist_a * dist_a
                dist_b_sq = dist_b * dist_b
                total_dist = dist_a_sq + dist_b_sq
                
                if total_dist > 0:
                    weights_a[y, x] = dist_b_sq / total_dist
                    weights_b[y, x] = dist_a_sq / total_dist
        
        return weights_a, weights_b
    
    @staticmethod
    def blend_images(image_a: np.ndarray, image_b: np.ndarray, 
                    weights_a: np.ndarray, weights_b: np.ndarray) -> np.ndarray:
        """
        Blend two images using weight matrices.
        
        Args:
            image_a: First image
            image_b: Second image  
            weights_a: Weights for first image
            weights_b: Weights for second image
            
        Returns:
            Blended image
        """
        # Ensure weights are 3-channel
        if len(weights_a.shape) == 2:
            weights_a = np.stack([weights_a] * 3, axis=2)
        if len(weights_b.shape) == 2:
            weights_b = np.stack([weights_b] * 3, axis=2)
        
        # Normalize weights
        total_weights = weights_a + weights_b
        mask = total_weights > 0
        
        weights_a = np.where(mask, weights_a / total_weights, 0)
        weights_b = np.where(mask, weights_b / total_weights, 0)
        
        # Blend images
        blended = (image_a.astype(np.float32) * weights_a + 
                  image_b.astype(np.float32) * weights_b)
        
        return blended.astype(np.uint8)


class ColorCorrector:
    """Advanced color correction for consistent appearance across cameras."""
    
    @staticmethod
    def balance_luminance(images: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Balance luminance across camera images for consistent brightness.
        
        Args:
            images: List of camera images [front, back, left, right]
            masks: List of overlap masks
            
        Returns:
            Luminance-corrected images
        """
        if len(images) != 4 or len(masks) != 4:
            print(f"⚠️  Color correction: Expected 4 images and 4 masks, got {len(images)} images and {len(masks)} masks")
            return images
        
        # Validate input images
        for i, img in enumerate(images):
            if img is None or img.size == 0:
                print(f"⚠️  Color correction: Image {i} is None or empty, skipping correction")
                return images
        
        front, back, left, right = images
        region_processor = ImageRegionProcessor(SystemConfig())
        
        try:
            # Split into color channels
            channels = []
            for img in images:
                if img.shape[2] != 3:
                    print(f"⚠️  Color correction: Image has {img.shape[2]} channels, expected 3")
                    return images
                channels.append(cv2.split(img))
            
            # Calculate luminance ratios for overlapping regions
            ratios = []
            
            for i in range(3):  # For each color channel
                try:
                    overlap_pairs = [
                        (region_processor.right_top(channels[3][i]), region_processor.front_right(channels[0][i]), masks[1]),
                        (region_processor.back_right(channels[1][i]), region_processor.right_bottom(channels[3][i]), masks[3]),
                        (region_processor.left_bottom(channels[2][i]), region_processor.back_left(channels[1][i]), masks[2]),
                        (region_processor.front_left(channels[0][i]), region_processor.left_top(channels[2][i]), masks[0])
                    ]
                    
                    channel_ratios = []
                    for j, (region_a, region_b, mask) in enumerate(overlap_pairs):
                        try:
                            # Validate regions
                            if region_a is None or region_b is None or mask is None:
                                print(f"⚠️  Color correction: Channel {i}, pair {j} has None region, using ratio 1.0")
                                channel_ratios.append(1.0)
                                continue
                                
                            if region_a.shape != region_b.shape or region_a.shape != mask.shape:
                                print(f"⚠️  Color correction: Channel {i}, pair {j} shape mismatch, using ratio 1.0")
                                channel_ratios.append(1.0)
                                continue
                            
                            mask_norm = (mask / 255.0).astype(np.float32)
                            sum_a = np.sum(region_a.astype(np.float32) * mask_norm)
                            sum_b = np.sum(region_b.astype(np.float32) * mask_norm)
                            
                            if sum_b > 0 and sum_a > 0:
                                ratio = sum_a / sum_b
                                # Clamp ratio to reasonable range
                                ratio = max(0.1, min(10.0, ratio))
                                channel_ratios.append(ratio)
                            else:
                                channel_ratios.append(1.0)
                        except Exception as e:
                            print(f"⚠️  Color correction: Channel {i}, pair {j} failed: {e}, using ratio 1.0")
                            channel_ratios.append(1.0)
                    
                    ratios.append(channel_ratios)
                    
                except Exception as e:
                    print(f"⚠️  Color correction: Channel {i} processing failed: {e}, using default ratios")
                    ratios.append([1.0, 1.0, 1.0, 1.0])
            
            # Calculate correction factors
            correction_factors = []
            for i in range(3):
                try:
                    # Ensure we have valid ratios
                    if len(ratios[i]) != 4:
                        print(f"⚠️  Color correction: Channel {i} has {len(ratios[i])} ratios, expected 4")
                        correction_factors.append([1.0, 1.0, 1.0, 1.0])
                        continue
                    
                    # Calculate geometric mean with protection against invalid values
                    valid_ratios = [r for r in ratios[i] if r > 0]
                    if len(valid_ratios) < 4:
                        print(f"⚠️  Color correction: Channel {i} has {len(valid_ratios)} valid ratios, using defaults")
                        correction_factors.append([1.0, 1.0, 1.0, 1.0])
                        continue
                    
                    geometric_mean = np.power(np.prod(valid_ratios), 0.25)
                    
                    factors = []
                    for j in range(4):
                        try:
                            if j == 0:  # front
                                factor = geometric_mean / np.sqrt(ratios[i][3] / ratios[i][0])
                            elif j == 1:  # back  
                                factor = geometric_mean / np.sqrt(ratios[i][1] / ratios[i][2])
                            elif j == 2:  # left
                                factor = geometric_mean / np.sqrt(ratios[i][2] / ratios[i][3])
                            else:  # right
                                factor = geometric_mean / np.sqrt(ratios[i][0] / ratios[i][1])
                            
                            # Validate factor
                            if not np.isfinite(factor) or factor <= 0:
                                factor = 1.0
                            
                            # Apply smoothing function
                            if factor >= 1:
                                factor = factor * np.exp((1 - factor) * 0.5)
                            else:
                                factor = factor * np.exp((1 - factor) * 0.8)
                            
                            # Clamp to reasonable range
                            factor = max(0.1, min(10.0, factor))
                            factors.append(factor)
                            
                        except Exception as e:
                            print(f"⚠️  Color correction: Factor calculation failed for channel {i}, camera {j}: {e}")
                            factors.append(1.0)
                    
                    correction_factors.append(factors)
                    
                except Exception as e:
                    print(f"⚠️  Color correction: Correction factor calculation failed for channel {i}: {e}")
                    correction_factors.append([1.0, 1.0, 1.0, 1.0])
            
            # Apply corrections
            corrected_images = []
            for img_idx in range(4):
                try:
                    corrected_channels = []
                    for ch_idx in range(3):
                        try:
                            factor = correction_factors[ch_idx][img_idx]
                            if not np.isfinite(factor) or factor <= 0:
                                factor = 1.0
                            
                            # Apply correction with overflow protection
                            corrected = np.clip(channels[img_idx][ch_idx].astype(np.float32) * factor, 0, 255).astype(np.uint8)
                            corrected_channels.append(corrected)
                        except Exception as e:
                            print(f"⚠️  Color correction: Channel correction failed for image {img_idx}, channel {ch_idx}: {e}")
                            corrected_channels.append(channels[img_idx][ch_idx])
                    
                    corrected_images.append(cv2.merge(corrected_channels))
                    
                except Exception as e:
                    print(f"⚠️  Color correction: Image {img_idx} correction failed: {e}")
                    corrected_images.append(images[img_idx])
            
            print("✅ Color correction completed successfully")
            return corrected_images
            
        except Exception as e:
            print(f"❌ Luminance correction failed: {e}")
            return images
    
    @staticmethod
    def white_balance(image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction.
        
        Args:
            image: Input image
            
        Returns:
            White-balanced image
        """
        try:
            b, g, r = cv2.split(image)
            
            # Calculate channel means
            mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
            overall_mean = (mean_b + mean_g + mean_r) / 3
            
            # Calculate correction factors
            factor_b = overall_mean / mean_b if mean_b > 0 else 1.0
            factor_g = overall_mean / mean_g if mean_g > 0 else 1.0
            factor_r = overall_mean / mean_r if mean_r > 0 else 1.0
            
            # Apply corrections
            b_corrected = np.minimum(b * factor_b, 255).astype(np.uint8)
            g_corrected = np.minimum(g * factor_g, 255).astype(np.uint8)
            r_corrected = np.minimum(r * factor_r, 255).astype(np.uint8)
            
            return cv2.merge([b_corrected, g_corrected, r_corrected])
            
        except Exception as e:
            print(f"Warning: White balance failed: {e}")
            return image


class SurroundViewProcessor:
    """
    Main processor for creating 360° surround view from multiple camera feeds.
    
    Features:
    - Advanced blending algorithms
    - Color correction
    - Real-time processing capabilities
    - Configurable quality settings
    """
    
    def __init__(self, method: BlendingMethod = BlendingMethod.DISTANCE_BASED):
        """
        Initialize the surround view processor.
        
        Args:
            method: Blending method to use
        """
        self.config = SystemConfig()
        self.method = method
        self.region_processor = ImageRegionProcessor(self.config)
        self.blender = AdvancedBlender()
        self.color_corrector = ColorCorrector()
        
        # Processing state
        self.weight_matrices: Optional[List[np.ndarray]] = None
        self.mask_matrices: Optional[List[np.ndarray]] = None
        self.result_image = np.zeros((*self.config.dimensions.total_dimensions[::-1], 3), np.uint8)
    
    def _get_image_mask(self, image: np.ndarray) -> np.ndarray:
        """Extract binary mask from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        return mask
    
    def _get_overlap_mask(self, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        """Find overlapping region between two images."""
        mask_a = self._get_image_mask(image_a)
        mask_b = self._get_image_mask(image_b)
        overlap = cv2.bitwise_and(mask_a, mask_b)
        
        # Dilate to ensure good coverage
        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(overlap, kernel, iterations=2)
    
    def _process_overlap_region(self, region_name: str, image_a: np.ndarray, image_b: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
        """Process a single overlap region in parallel."""
        try:
            weights, mask = self.blender.create_distance_weights(
                self._get_image_mask(image_a),
                self._get_image_mask(image_b)
            )
            return region_name, weights, mask
        except Exception as e:
            print(f"❌ Failed to process {region_name}: {e}")
            return region_name, None, None

    def generate_weight_matrices(self, projected_images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate blending weight matrices and masks from projected images using multi-threading.
        
        Args:
            projected_images: List of projected images [front, back, left, right]
            
        Returns:
            Tuple of (weight_matrices, mask_matrices) as 4-channel images
        """
        if len(projected_images) != 4:
            raise ValueError("Expected 4 projected images")
        
        front, back, left, right = projected_images
        
        # Define overlap regions for parallel processing
        overlap_tasks = [
            ("FL", self.region_processor.front_left(front), self.region_processor.left_top(left)),
            ("FR", self.region_processor.front_right(front), self.region_processor.right_top(right)),
            ("BL", self.region_processor.back_left(back), self.region_processor.left_bottom(left)),
            ("BR", self.region_processor.back_right(back), self.region_processor.right_bottom(right))
        ]
        
        # Process overlap regions in parallel
        weight_results = {}
        mask_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all overlap region tasks
            future_to_region = {
                executor.submit(self._process_overlap_region, region_name, img_a, img_b): region_name
                for region_name, img_a, img_b in overlap_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_region):
                region_name, weights, mask = future.result()
                
                if weights is None or mask is None:
                    raise RuntimeError(f"Failed to process overlap region {region_name}")
                
                weight_results[region_name] = weights
                mask_results[region_name] = mask
        
        # Maintain order: FL, FR, BL, BR
        weight_matrices = [weight_results[region] for region in ["FL", "FR", "BL", "BR"]]
        mask_matrices = [mask_results[region] for region in ["FL", "FR", "BL", "BR"]]
        
        # Store for later use
        self.weight_matrices = [np.stack([w] * 3, axis=2) for w in weight_matrices]
        self.mask_matrices = [(m / 255.0).astype(int) for m in mask_matrices]
        
        # Stack into 4-channel arrays for saving
        weight_array = np.stack(weight_matrices, axis=2)
        mask_array = np.stack(mask_matrices, axis=2)
        
        return weight_array, mask_array
    
    def _process_center_region(self, region_name: str, image: np.ndarray) -> Tuple[str, np.ndarray]:
        """Process a center region in parallel."""
        try:
            if region_name == "front_center":
                processed = self.region_processor.front_center(image)
            elif region_name == "back_center":
                processed = self.region_processor.back_center(image)
            elif region_name == "left_center":
                processed = self.region_processor.left_center(image)
            elif region_name == "right_center":
                processed = self.region_processor.right_center(image)
            else:
                raise ValueError(f"Unknown center region: {region_name}")
            
            return region_name, processed
        except Exception as e:
            print(f"❌ Failed to process center region {region_name}: {e}")
            return region_name, None
    
    def _process_blend_region(self, region_info: Tuple[str, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[str, np.ndarray]:
        """Process a blend region in parallel."""
        try:
            region_name, image_a, image_b, weights = region_info
            blended = self.blender.blend_images(image_a, image_b, weights, 1 - weights)
            return region_name, blended
        except Exception as e:
            print(f"❌ Failed to process blend region {region_name}: {e}")
            return region_name, None

    def stitch_images(self, projected_images: List[np.ndarray], 
                     apply_color_correction: bool = True) -> np.ndarray:
        """
        Stitch projected images into final surround view using multi-threading.
        
        Args:
            projected_images: List of projected images [front, back, left, right]
            apply_color_correction: Whether to apply color correction
            
        Returns:
            Final stitched surround view image
        """
        if len(projected_images) != 4:
            raise ValueError("Expected 4 projected images")
        
        # Apply color correction if requested
        if apply_color_correction and self.mask_matrices:
            projected_images = self.color_corrector.balance_luminance(
                projected_images, self.mask_matrices
            )
        
        front, back, left, right = projected_images
        
        # Clear result image
        self.result_image.fill(0)
        
        # Get boundaries
        xl, xr, yt, yb = self.config.dimensions.car_boundaries
        
        # Process center regions in parallel
        center_tasks = [
            ("front_center", front),
            ("back_center", back),
            ("left_center", left),
            ("right_center", right)
        ]
        
        center_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit center region tasks
            future_to_region = {
                executor.submit(self._process_center_region, name, img): name
                for name, img in center_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_region):
                region_name, processed = future.result()
                if processed is not None:
                    center_results[region_name] = processed
        
        # Copy center regions to result image
        if "front_center" in center_results:
            self.result_image[:yt, xl:xr] = center_results["front_center"]
        if "back_center" in center_results:
            self.result_image[yb:, xl:xr] = center_results["back_center"]
        if "left_center" in center_results:
            self.result_image[yt:yb, :xl] = center_results["left_center"]
        if "right_center" in center_results:
            self.result_image[yt:yb, xr:] = center_results["right_center"]
        
        # Process blend regions in parallel if weights are available
        if self.weight_matrices and len(self.weight_matrices) == 4:
            blend_tasks = [
                ("FL", self.region_processor.front_left(front), self.region_processor.left_top(left), self.weight_matrices[0]),
                ("FR", self.region_processor.front_right(front), self.region_processor.right_top(right), self.weight_matrices[1]),
                ("BL", self.region_processor.back_left(back), self.region_processor.left_bottom(left), self.weight_matrices[2]),
                ("BR", self.region_processor.back_right(back), self.region_processor.right_bottom(right), self.weight_matrices[3])
            ]
            
            blend_results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit blend region tasks
                future_to_region = {
                    executor.submit(self._process_blend_region, task): task[0]
                    for task in blend_tasks
                }
                
                # Collect results
                for future in as_completed(future_to_region):
                    region_name, blended = future.result()
                    if blended is not None:
                        blend_results[region_name] = blended
            
            # Copy blend regions to result image
            if "FL" in blend_results:
                self.result_image[:yt, :xl] = blend_results["FL"]
            if "FR" in blend_results:
                self.result_image[:yt, xr:] = blend_results["FR"]
            if "BL" in blend_results:
                self.result_image[yb:, :xl] = blend_results["BL"]
            if "BR" in blend_results:
                self.result_image[yb:, xr:] = blend_results["BR"]
        
        # Apply white balance to final result
        if apply_color_correction:
            self.result_image = self.color_corrector.white_balance(self.result_image)
        
        # Add car overlay if available
        if self.config.car_image is not None:
            self.result_image[yt:yb, xl:xr] = self.config.car_image
        
        return self.result_image
    
    def save_weight_matrices(self, weight_array: np.ndarray, mask_array: np.ndarray,
                           weights_file: str = "weights.png", masks_file: str = "masks.png") -> None:
        """
        Save weight matrices and masks to image files.
        
        Args:
            weight_array: 4-channel weight matrix array
            mask_array: 4-channel mask array  
            weights_file: Output filename for weights
            masks_file: Output filename for masks
        """
        try:
            # Save weight matrices (convert to 0-255 range)
            weights_image = (weight_array * 255).astype(np.uint8)
            Image.fromarray(weights_image).save(weights_file)
            
            # Save mask matrices
            masks_image = mask_array.astype(np.uint8)
            Image.fromarray(masks_image).save(masks_file)
            
            print(f"✅ Saved weight matrices: {weights_file}")
            print(f"✅ Saved mask matrices: {masks_file}")
            
        except Exception as e:
            print(f"❌ Failed to save matrices: {e}")
    
    def process_camera_feeds(self, camera_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Complete processing pipeline from raw camera images to final surround view.
        
        Args:
            camera_images: Dictionary mapping camera names to raw images
            
        Returns:
            Final surround view image
        """
        try:
            # Load camera models
            cameras = {}
            for name in self.config.camera_names:
                cameras[name] = FisheyeCamera(self.config.get_camera_config(name))
            
            # Process each camera image
            projected_images = []
            for name in self.config.camera_names:
                if name not in camera_images:
                    raise ValueError(f"Missing camera image: {name}")
                
                image = camera_images[name]
                processed = cameras[name].process_image(image)
                projected_images.append(processed)
            
            # Generate weights if not available
            if self.weight_matrices is None:
                self.generate_weight_matrices(projected_images)
            
            # Stitch final image
            return self.stitch_images(projected_images)
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return self.result_image 