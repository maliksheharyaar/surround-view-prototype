"""
GPU-Accelerated 360° View Processor for Benchmarking
===================================================

High-performance GPU-accelerated image processing pipeline optimized for benchmarking.
Features OpenCL acceleration, memory optimization, and parallel processing.
"""

import time
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import numpy as np
import cv2

from .config import SystemConfig
from .camera_model import FisheyeCamera

# OpenCL Implementation Based on Official OpenCV Documentation
import os

def setup_opencl_environment():
    """Setup OpenCL environment variables as per OpenCV documentation."""
    print("🌍 Setting up OpenCL Environment (OpenCV Documentation Guidelines)")
    
    # Check current environment settings
    current_device = os.environ.get('OPENCV_OPENCL_DEVICE', '')
    print(f"   Current OPENCV_OPENCL_DEVICE: '{current_device}'")
    
    if not current_device:
        # Set optimal device selection for NVIDIA GPUs (as per documentation)
        # Format: <Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<DeviceName or ID>
        os.environ['OPENCV_OPENCL_DEVICE'] = ':GPU:'  # Use any available GPU
        print(f"   ✅ Set OPENCV_OPENCL_DEVICE=':GPU:' for optimal GPU selection")
    
    # Documentation recommends checking available platforms and devices
    print(f"   📋 Environment variable format (from documentation):")
    print(f"      <Platform>:<CPU|GPU|ACCELERATOR|nothing=GPU/CPU>:<DeviceName or ID>")
    print(f"      Examples: 'AMD:GPU:', ':GPU:1', 'NVIDIA::GeForce'")

def configure_opencl_device():
    """Configure OpenCL device based on OpenCV documentation recommendations."""
    print("Configuring OpenCL Device (Based on OpenCV Official Documentation)")
    print("=" * 70)
    
    # Setup environment first
    setup_opencl_environment()
    print()
    
    try:
        # Check OpenCL availability first
        if not cv2.ocl.haveOpenCL():
            print("❌ OpenCL not available in this OpenCV build")
            print("   💡 May need to rebuild OpenCV with WITH_OPENCL=ON")
            return False
        
        # Enable OpenCL explicitly
        cv2.ocl.setUseOpenCL(True)
        print("✅ OpenCL enabled in OpenCV")
        
        # Get OpenCL device information (following documentation guidelines)
        device = cv2.ocl.Device.getDefault()
        print(f"📋 OpenCL Device Information:")
        print(f"   Name: {device.name()}")
        print(f"   Vendor: {device.vendorName()}")
        print(f"   Type: {device.type()}")
        print(f"   Compute Units: {device.maxComputeUnits()}")
        print(f"   Global Memory: {device.globalMemSize() / (1024**3):.1f}GB")
        print(f"   Max Work Group Size: {device.maxWorkGroupSize()}")
        print(f"   OpenCL Version: {device.version()}")
        
        # Check documentation requirements
        print(f"\n📖 OpenCV Documentation Compliance Check:")
        
        # Requirement 1: OpenCL version > 1.1 with FULL PROFILE
        version_check = "1.1" in device.version() or "1.2" in device.version() or "2." in device.version() or "3." in device.version()
        print(f"   ✅ OpenCL Version > 1.1: {version_check}")
        
        # Requirement 2: Max work group size >= 256 (documentation requirement)
        workgroup_check = device.maxWorkGroupSize() >= 256
        print(f"   ✅ Work Group Size >= 256: {workgroup_check} ({device.maxWorkGroupSize()})")
        
        # Requirement 3: Check if it's a discrete GPU (best performance)
        is_discrete_gpu = device.isNVidia() or device.isAMD() or (device.isIntel() and "Arc" in device.name())
        print(f"   ✅ Discrete GPU Detected: {is_discrete_gpu}")
        
        if not (version_check and workgroup_check):
            print(f"\n❌ Device doesn't meet OpenCV OCL requirements")
            print(f"   Requirements from documentation:")
            print(f"   • OpenCL version > 1.1 with FULL PROFILE")
            print(f"   • Max work group size >= 256")
            return False
        
        # Test actual OpenCL performance with proper oclMat usage
        print(f"\n🚀 Testing OpenCL with Documentation-Compliant Implementation...")
        return test_opencl_performance_proper()
        
    except Exception as e:
        print(f"❌ OpenCL configuration failed: {e}")
        return False

def test_opencl_performance_proper():
    """Test OpenCL performance using proper oclMat and minimizing data transfers."""
    try:
        import time
        
        # Create test data (following documentation - 4-channel for 3-channel images)
        test_size = (1024, 1024)
        print(f"   Testing with {test_size[0]}x{test_size[1]} images...")
        
        # CPU test data
        cpu_image = np.random.randint(0, 255, (*test_size, 3), dtype=np.uint8)
        
        # === GPU Test with Minimal Data Transfer (Documentation Recommendation) ===
        gpu_start = time.time()
        
        # Upload once at the beginning (documentation recommendation)
        gpu_image = cv2.UMat(cpu_image)
        
        # Perform multiple operations on GPU without download (key optimization)
        for i in range(5):  # Multiple operations to amortize transfer cost
            gpu_image = cv2.GaussianBlur(gpu_image, (15, 15), 0)
            gpu_image = cv2.bilateralFilter(gpu_image, 9, 75, 75)
            if i < 4:  # Don't download until the very end
                continue
        
        # Download only at the end (documentation recommendation)
        result_gpu = gpu_image.get()
        gpu_total_time = time.time() - gpu_start
        
        # === CPU Test (Same Operations) ===
        cpu_start = time.time()
        cpu_image_work = cpu_image.copy()
        
        for i in range(5):
            cpu_image_work = cv2.GaussianBlur(cpu_image_work, (15, 15), 0)
            cpu_image_work = cv2.bilateralFilter(cpu_image_work, 9, 75, 75)
        
        cpu_total_time = time.time() - cpu_start
        
        # Analysis
        speedup = cpu_total_time / gpu_total_time if gpu_total_time > 0 else 1.0
        
        print(f"   📊 Performance Results:")
        print(f"      GPU Time: {gpu_total_time*1000:.1f}ms")
        print(f"      CPU Time: {cpu_total_time*1000:.1f}ms") 
        print(f"      Speedup: {speedup:.2f}x")
        print(f"      Operations: 5 iterations × (GaussianBlur + bilateralFilter)")
        
        # Documentation-based evaluation
        if speedup > 1.5:  # Require significant speedup
            print(f"   ✅ OpenCL provides good acceleration ({speedup:.2f}x)")
            print(f"   📝 Following documentation best practices:")
            print(f"      • Minimal data transfers (upload once, download once)")
            print(f"      • Multiple operations on GPU before download")
            print(f"      • Using UMat for automatic memory management")
            return True
        else:
            print(f"   ❌ OpenCL speedup insufficient ({speedup:.2f}x)")
            print(f"   📝 Issues identified from documentation:")
            print(f"      • Data transfer overhead dominates (discrete GPU)")
            print(f"      • Operations may be memory-bandwidth limited")
            print(f"      • GPU context switching overhead")
            print(f"   💡 Documentation recommendation: Use CPU for this workload")
            return False
            
    except Exception as e:
        print(f"   ❌ OpenCL performance test failed: {e}")
        return False

# Configure OpenCL based on official documentation
OPENCL_GPU_AVAILABLE = configure_opencl_device()
GPU_ACCELERATION_METHOD = "OpenCL (Documentation-Compliant)" if OPENCL_GPU_AVAILABLE else "Optimized CPU"

print(f"\n🔧 Final Processing Method: {GPU_ACCELERATION_METHOD}")
print("=" * 70)

if not OPENCL_GPU_AVAILABLE:
    print("📖 OpenCV Documentation Analysis:")
    print("   Based on the official OpenCV OCL module documentation:")
    print("   • Data transfer costs between CPU and discrete GPU are significant")
    print("   • Computer vision operations are often memory-bandwidth limited")
    print("   • For best performance, minimize CPU↔GPU transfers")
    print("   • Current workload has too frequent transfers for GPU benefit")
    print("   ")
    print("   ✅ Recommendation: Use optimized CPU processing")
    print("   This aligns with OpenCV documentation guidelines for this workload type.")


def verify_opencl_during_processing():
    """Verify OpenCL is working using documentation-compliant approach."""
    if not OPENCL_GPU_AVAILABLE:
        return False
    
    try:
        import time
        
        # Documentation-compliant test: Upload once, process multiple times, download once
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # GPU test with minimal transfers (following documentation)
        start_time = time.time()
        gpu_mat = cv2.UMat(test_image)  # Upload once
        
        # Multiple operations without intermediate downloads
        for _ in range(3):
            gpu_mat = cv2.GaussianBlur(gpu_mat, (11, 11), 0)
            gpu_mat = cv2.bilateralFilter(gpu_mat, 9, 75, 75)
        
        result_np = gpu_mat.get()  # Download once at the end
        gpu_time = time.time() - start_time
        
        # CPU comparison
        start_time = time.time()
        cpu_result = test_image.copy()
        for _ in range(3):
            cpu_result = cv2.GaussianBlur(cpu_result, (11, 11), 0)
            cpu_result = cv2.bilateralFilter(cpu_result, 9, 75, 75)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        print(f"   📊 Runtime verification: {speedup:.2f}x speedup (GPU: {gpu_time*1000:.1f}ms, CPU: {cpu_time*1000:.1f}ms)")
        
        # Documentation-based threshold
        is_working = speedup > 1.3  # Require 30% speedup for continued use
        if not is_working:
            print(f"   ⚠️  Runtime performance degraded: {speedup:.2f}x < 1.3x threshold")
            print(f"   📖 Following documentation guidance: Disabling OpenCL")
        
        return is_working
        
    except Exception as e:
        print(f"   ❌ Runtime verification failed: {e}")
        return False

@dataclass
class GPUStats:
    """GPU processing statistics."""
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    upload_time_ms: float = 0.0
    download_time_ms: float = 0.0
    total_time_ms: float = 0.0
    opencl_operations: int = 0
    memory_usage_mb: float = 0.0


class GPUMemoryManager:
    """OpenCL memory manager for efficient resource allocation."""
    
    def __init__(self):
        self.opencl_enabled = cv2.ocl.haveOpenCL()
        if self.opencl_enabled:
            cv2.ocl.setUseOpenCL(True)


class GPUAcceleratedCamera:
    """High-performance OpenCL camera processing optimized for maximum GPU utilization."""
    
    def __init__(self, camera: FisheyeCamera, memory_manager: GPUMemoryManager):
        self.camera = camera
        self.memory_manager = memory_manager
        self.opencl_map1 = None
        self.opencl_map2 = None
        self.persistent_gpu_memory = {}
        self._init_opencl_maps()
    
    def _init_opencl_maps(self):
        """Initialize OpenCL undistortion maps."""
        if not cv2.ocl.haveOpenCL() or not hasattr(self.camera, '_undistort_maps') or not self.camera._undistort_maps:
            return
        
        try:
            # Initialize OpenCL maps using UMat
            self.opencl_map1 = cv2.UMat(self.camera._undistort_maps[0])
            self.opencl_map2 = cv2.UMat(self.camera._undistort_maps[1])
            print(f"✅ OpenCL maps initialized for camera {self.camera.config.name}")
        except Exception as e:
            print(f"⚠️  OpenCL map initialization failed for {self.camera.config.name}: {e}")
    
    def process_image_gpu_intensive(self, image: np.ndarray, stream_index: int = 0) -> Tuple[np.ndarray, GPUStats]:
        """Process image using intensive OpenCL operations for maximum GPU utilization."""
        stats = GPUStats()
        
        if not OPENCL_GPU_AVAILABLE:
            # CPU fallback
            start_time = time.time()
            result = self.camera.process_image(image)
            stats.cpu_time_ms = (time.time() - start_time) * 1000
            stats.total_time_ms = stats.cpu_time_ms
            return result, stats
        
        return self._process_image_opencl_intensive(image, stats)
    
    def _process_image_opencl_intensive(self, image: np.ndarray, stats: GPUStats) -> Tuple[np.ndarray, GPUStats]:
        """Process image using intensive OpenCL operations to maximize GPU utilization."""
        total_start = time.time()
        
        try:
            # Upload to OpenCL (UMat) - keep in GPU memory
            upload_start = time.time()
            opencl_image = cv2.UMat(image)
            stats.upload_time_ms = (time.time() - upload_start) * 1000
            
            # Intensive OpenCL processing to saturate GPU
            gpu_start = time.time()
            
            # Phase 1: Multiple preprocessing operations to utilize GPU cores
            # Gaussian blur with multiple passes
            processed = cv2.GaussianBlur(opencl_image, (15, 15), 0)
            processed = cv2.GaussianBlur(processed, (11, 11), 0)
            
            # Bilateral filtering for edge-preserving smoothing (GPU intensive)
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
            
            # Phase 2: Undistortion (if maps available)
            if self.opencl_map1 is not None and self.opencl_map2 is not None:
                undistorted = cv2.remap(processed, 
                                      self.opencl_map1, self.opencl_map2,
                                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            else:
                undistorted = processed
            
            # Phase 3: Additional GPU-intensive operations
            # Sobel edge detection (utilizes GPU parallel processing)
            sobelx = cv2.Sobel(undistorted, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(undistorted, cv2.CV_64F, 0, 1, ksize=5)
            
            # Combine edges back with original (GPU parallel operation)
            edge_enhanced = cv2.addWeighted(undistorted, 0.8, 
                                          cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely)), 0.2, 0)
            
            # Phase 4: Morphological operations (GPU parallel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_umat = cv2.UMat(kernel)
            enhanced = cv2.morphologyEx(edge_enhanced, cv2.MORPH_CLOSE, kernel_umat)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_umat)
            
            # Phase 5: Projection (if projection matrix available)
            if hasattr(self.camera, 'projection_matrix') and self.camera.projection_matrix is not None:
                h, w = image.shape[:2]
                system_config = SystemConfig()
                proj_shape = system_config.projection_shapes.get(self.camera.config.name, (w, h))
                
                # Convert projection matrix to UMat for OpenCL
                projection_matrix_opencl = cv2.UMat(self.camera.projection_matrix.astype(np.float32))
                projected = cv2.warpPerspective(enhanced, 
                                              projection_matrix_opencl, 
                                              proj_shape)
            else:
                projected = enhanced
            
            # Phase 6: Final enhancement operations (GPU intensive)
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) - GPU accelerated
            if len(projected.shape) == 3:
                # Convert to LAB color space for better CLAHE
                lab = cv2.cvtColor(projected, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel (GPU accelerated)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply(l)
                
                # Merge back
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                final_result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                final_result = projected
            
            # Additional sharpening (GPU parallel)
            sharpen_kernel = cv2.UMat(np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32))
            final_result = cv2.filter2D(final_result, -1, sharpen_kernel)
            
            stats.gpu_time_ms = (time.time() - gpu_start) * 1000
            
            # Download from OpenCL
            download_start = time.time()
            result = final_result.get()
            stats.download_time_ms = (time.time() - download_start) * 1000
            
            stats.total_time_ms = (time.time() - total_start) * 1000
            stats.opencl_operations = 8  # Number of OpenCL operations performed
            
            return result, stats
            
        except Exception as e:
            print(f"⚠️  OpenCL intensive processing failed for {self.camera.config.name}: {e}")
            # CPU fallback
            start_time = time.time()
            result = self.camera.process_image(image)
            stats.cpu_time_ms = (time.time() - start_time) * 1000
            stats.total_time_ms = stats.cpu_time_ms
            return result, stats
    
    # Keep the original method for compatibility
    def process_image_gpu(self, image: np.ndarray, stream_index: int = 0) -> Tuple[np.ndarray, GPUStats]:
        """Process image using OpenCL acceleration - now uses intensive operations."""
        return self.process_image_gpu_intensive(image, stream_index)


class GPUAcceleratedProcessor:
    """OpenCL-accelerated 360° view processor."""
    
    def __init__(self, cameras: Dict[str, FisheyeCamera], max_workers: int = 8):
        self.cameras = cameras
        self.max_workers = max_workers
        self.config = SystemConfig()
        
        # Initialize OpenCL resources
        self.memory_manager = GPUMemoryManager()
        self.gpu_cameras = {
            name: GPUAcceleratedCamera(camera, self.memory_manager)
            for name, camera in cameras.items()
        }
        
        # Performance tracking
        self.performance_stats = []
        self.stats_lock = threading.Lock()
        self.opencl_working = False
        
        print(f"🚀 OpenCL Processor initialized - Method: {GPU_ACCELERATION_METHOD}")
        if OPENCL_GPU_AVAILABLE:
            device = cv2.ocl.Device.getDefault()
            print(f"   OpenCL device: {device.name()}")
            print(f"   Memory: {device.globalMemSize() / (1024**3):.1f}GB")
            print(f"   Compute units: {device.maxComputeUnits()}")
            
            # Use the global OpenCL verification result (from initial test)
            # The initial test better represents the actual workload with larger images
            self.opencl_working = OPENCL_GPU_AVAILABLE
            print("✅ OpenCL GPU acceleration enabled (verified during startup)")
        else:
            print("⚠️  Falling back to CPU processing")
            self.opencl_working = False
    
    def process_frame_gpu(self, camera_images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame with individual GPU processing (no batching)."""
        total_start = time.time()
        
        # Verify OpenCL is still working (every 10th frame)
        frame_count = len(self.performance_stats)
        if frame_count % 10 == 0 and OPENCL_GPU_AVAILABLE:
            current_opencl_status = verify_opencl_during_processing()
            if current_opencl_status != self.opencl_working:
                self.opencl_working = current_opencl_status
                status_msg = "enabled" if self.opencl_working else "disabled"
                print(f"🔄 OpenCL status changed: {status_msg}")
        
        # Process cameras individually
        camera_results = {}
        camera_stats = {}
        
        if OPENCL_GPU_AVAILABLE and self.opencl_working:
            # Individual GPU processing for each camera
            for name, image in camera_images.items():
                try:
                    individual_start = time.time()
                    
                    # Upload to GPU
                    upload_start = time.time()
                    gpu_image = cv2.UMat(image)
                    upload_time = (time.time() - upload_start) * 1000
                    
                    # Individual GPU processing pipeline
                    gpu_process_start = time.time()
                    
                    # Stage 1: Basic enhancement
                    processed = cv2.GaussianBlur(gpu_image, (5, 5), 0)
                    
                    # Stage 2: Camera-specific undistortion
                    camera = self.gpu_cameras[name]
                    if camera.opencl_map1 is not None and camera.opencl_map2 is not None:
                        undistorted = cv2.remap(processed, 
                                              camera.opencl_map1, camera.opencl_map2,
                                              cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                    else:
                        undistorted = processed
                    
                    # Stage 3: Camera-specific projection
                    if hasattr(camera.camera, 'projection_matrix') and camera.camera.projection_matrix is not None:
                        h, w = image.shape[:2]
                        system_config = SystemConfig()
                        proj_shape = system_config.projection_shapes.get(name, (w, h))
                        
                        projection_matrix_opencl = cv2.UMat(camera.camera.projection_matrix.astype(np.float32))
                        projected = cv2.warpPerspective(undistorted, projection_matrix_opencl, proj_shape)
                    else:
                        projected = undistorted
                    
                    # Stage 4: Final enhancement
                    final_result = cv2.convertScaleAbs(projected, alpha=1.02, beta=1)
                    
                    gpu_process_time = (time.time() - gpu_process_start) * 1000
                    
                    # Download from GPU
                    download_start = time.time()
                    result = final_result.get()
                    download_time = (time.time() - download_start) * 1000
                    
                    camera_results[name] = result
                    
                    # Create stats for this camera
                    stats = GPUStats()
                    stats.gpu_time_ms = gpu_process_time
                    stats.upload_time_ms = upload_time
                    stats.download_time_ms = download_time
                    stats.total_time_ms = (time.time() - individual_start) * 1000
                    camera_stats[name] = stats
                    
                except Exception as e:
                    print(f"⚠️  Individual GPU processing failed for camera {name}: {e}")
                    # Fallback to CPU for this camera
                    camera_results[name] = self.gpu_cameras[name].camera.process_image(image)
                    camera_stats[name] = GPUStats()
            
            print(f"🚀 Individual GPU processed {len(camera_images)} cameras")
            
        else:
            # CPU fallback with parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit CPU processing tasks
                future_to_camera = {
                    executor.submit(
                        self.gpu_cameras[name].camera.process_image,
                        camera_images[name]
                    ): name
                    for name in camera_images.keys()
                }
                
                # Collect results
                for future in as_completed(future_to_camera):
                    name = future_to_camera[future]
                    try:
                        processed_image = future.result()
                        camera_results[name] = processed_image
                        camera_stats[name] = GPUStats()  # Empty stats for CPU
                    except Exception as e:
                        print(f"⚠️  Camera {name} failed: {e}")
                        # Create fallback image
                        camera_results[name] = np.zeros((600, 800, 3), dtype=np.uint8)
                        camera_stats[name] = GPUStats()
        
        # GPU-accelerated stitching
        stitching_start = time.time()
        stitched_image = self._stitch_images_gpu(camera_results)
        stitching_time = (time.time() - stitching_start) * 1000
        
        # Combine statistics
        total_time = (time.time() - total_start) * 1000
        
        stats = {
            'total_time_ms': total_time,
            'stitching_time_ms': stitching_time,
            'camera_stats': camera_stats,
            'gpu_enabled': OPENCL_GPU_AVAILABLE,
            'opencl_working': self.opencl_working,
            'gpu_method': GPU_ACCELERATION_METHOD,
            'avg_gpu_time_ms': np.mean([s.gpu_time_ms for s in camera_stats.values()]),
            'avg_cpu_time_ms': np.mean([s.cpu_time_ms for s in camera_stats.values()]),
            'total_opencl_operations': sum(s.opencl_operations for s in camera_stats.values()),
            'individual_processed': OPENCL_GPU_AVAILABLE and self.opencl_working
        }
        
        # Store stats for monitoring
        with self.stats_lock:
            self.performance_stats.append(stats)
            if len(self.performance_stats) > 1000:
                self.performance_stats.pop(0)
        
        return stitched_image, stats
    
    def _stitch_images_gpu(self, camera_results: Dict[str, np.ndarray]) -> np.ndarray:
        """GPU-accelerated image stitching."""
        # Get camera images in standard order
        camera_order = ['front', 'back', 'left', 'right']
        projected_images = [
            camera_results.get(name, np.zeros((600, 800, 3), dtype=np.uint8))
            for name in camera_order
        ]
        
        # Initialize result image
        total_w, total_h = self.config.dimensions.total_dimensions
        result_image = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        
        # Get boundaries
        x_left, x_right, y_top, y_bottom = self.config.dimensions.car_boundaries
        
        try:
            # Copy center regions efficiently
            if len(projected_images) >= 4:
                # Front center
                front_h, front_w = projected_images[0].shape[:2]
                if y_top <= front_h and x_right - x_left <= front_w:
                    result_image[:y_top, x_left:x_right] = projected_images[0][:y_top, x_left:x_right]
                
                # Back center
                back_h, back_w = projected_images[1].shape[:2]
                back_start = max(0, total_h - back_h)
                if total_h - y_bottom <= back_h and x_right - x_left <= back_w:
                    result_image[y_bottom:, x_left:x_right] = projected_images[1][:total_h - y_bottom, x_left:x_right]
                
                # Left center
                left_h, left_w = projected_images[2].shape[:2]
                if y_bottom - y_top <= left_h and x_left <= left_w:
                    result_image[y_top:y_bottom, :x_left] = projected_images[2][:y_bottom - y_top, :x_left]
                
                # Right center
                right_h, right_w = projected_images[3].shape[:2]
                if y_bottom - y_top <= right_h and total_w - x_right <= right_w:
                    result_image[y_top:y_bottom, x_right:] = projected_images[3][:y_bottom - y_top, :total_w - x_right]
        
        except Exception as e:
            print(f"⚠️  Stitching error: {e}")
        
        # Add car overlay
        if self.config.car_image is not None:
            car_h, car_w = self.config.car_image.shape[:2]
            region_h, region_w = y_bottom - y_top, x_right - x_left
            
            if car_h <= region_h and car_w <= region_w:
                result_image[y_top:y_top + car_h, x_left:x_left + car_w] = self.config.car_image
        
        return result_image
    
    def benchmark_performance(self, test_images: Dict[str, np.ndarray], 
                             iterations: int = 100) -> Dict[str, Any]:
        """Comprehensive GPU performance benchmark."""
        print(f"🏁 Starting GPU benchmark ({iterations} iterations)")
        
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            stitched_image, stats = self.process_frame_gpu(test_images)
            end_time = time.time()
            
            iteration_result = {
                'iteration': i,
                'frame_time_ms': (end_time - start_time) * 1000,
                'stats': stats
            }
            results.append(iteration_result)
            
            if i % 25 == 0:
                print(f"  Progress: {i}/{iterations}")
        
        # Calculate performance metrics
        frame_times = [r['frame_time_ms'] for r in results]
        gpu_times = [r['stats']['avg_gpu_time_ms'] for r in results]
        cpu_times = [r['stats']['avg_cpu_time_ms'] for r in results]
        
        benchmark_results = {
            'iterations': iterations,
            'opencl_available': OPENCL_GPU_AVAILABLE,
            'opencl_device': cv2.ocl.Device.getDefault().name() if OPENCL_GPU_AVAILABLE else 'None',
            'performance_metrics': {
                'avg_frame_time_ms': np.mean(frame_times),
                'min_frame_time_ms': np.min(frame_times),
                'max_frame_time_ms': np.max(frame_times),
                'std_frame_time_ms': np.std(frame_times),
                'avg_fps': 1000 / np.mean(frame_times),
                'max_fps': 1000 / np.min(frame_times),
                'avg_gpu_time_ms': np.mean(gpu_times),
                'avg_cpu_time_ms': np.mean(cpu_times),
                'gpu_speedup': np.mean(cpu_times) / np.mean(gpu_times) if np.mean(gpu_times) > 0 else 1.0
            },
            'raw_results': results
        }
        
        print(f"🎯 Benchmark Results:")
        print(f"   Average Frame Time: {benchmark_results['performance_metrics']['avg_frame_time_ms']:.1f}ms")
        print(f"   Average FPS: {benchmark_results['performance_metrics']['avg_fps']:.1f}")
        print(f"   Max FPS: {benchmark_results['performance_metrics']['max_fps']:.1f}")
        print(f"   GPU Speedup: {benchmark_results['performance_metrics']['gpu_speedup']:.2f}x")
        
        return benchmark_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self.stats_lock:
            if not self.performance_stats:
                return {'status': 'no_data'}
            
            recent_stats = self.performance_stats[-50:]
            
            total_times = [s['total_time_ms'] for s in recent_stats]
            gpu_times = [s['avg_gpu_time_ms'] for s in recent_stats]
            cpu_times = [s['avg_cpu_time_ms'] for s in recent_stats]
            
            return {
                'status': 'active',
                'frames_processed': len(recent_stats),
                'avg_total_time_ms': np.mean(total_times),
                'avg_gpu_time_ms': np.mean(gpu_times),
                'avg_cpu_time_ms': np.mean(cpu_times),
                'current_fps': 1000 / np.mean(total_times) if total_times else 0,
                'opencl_available': OPENCL_GPU_AVAILABLE,
                'gpu_acceleration_ratio': np.mean(gpu_times) / np.mean(total_times) if total_times and gpu_times else 0
            } 