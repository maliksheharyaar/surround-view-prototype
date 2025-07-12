#!/usr/bin/env python3
"""
System Profiler Installation Test
=================================

Quick test script to verify that the system profiler and all its dependencies
are properly installed and working.
"""

import sys
import time
import traceback


def test_basic_imports():
    """Test basic Python imports."""
    print("📦 Testing basic imports...")
    
    try:
        import psutil
        print(f"   ✅ psutil {psutil.__version__}")
    except ImportError as e:
        print(f"   ❌ psutil: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"   ❌ numpy: {e}")
        return False
    
    try:
        import cv2
        print(f"   ✅ opencv {cv2.__version__}")
    except ImportError as e:
        print(f"   ❌ opencv: {e}")
        return False
    
    return True


def test_optional_imports():
    """Test optional imports."""
    print("\n🎨 Testing optional imports...")
    
    matplotlib_available = False
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"   ✅ matplotlib {matplotlib.__version__}")
        matplotlib_available = True
    except ImportError as e:
        print(f"   ⚠️  matplotlib: {e} (GUI mode will be unavailable)")
    
    tkinter_available = False
    try:
        import tkinter as tk
        print(f"   ✅ tkinter (GUI support available)")
        tkinter_available = True
    except ImportError as e:
        print(f"   ⚠️  tkinter: {e} (GUI mode will be unavailable)")
    
    nvml_available = False
    try:
        import py3nvml.py3nvml as nvml
        print(f"   ✅ py3nvml (Enhanced GPU monitoring)")
        nvml_available = True
    except ImportError as e:
        print(f"   ⚠️  py3nvml: {e} (Basic GPU monitoring only)")
    
    return {
        'matplotlib': matplotlib_available,
        'tkinter': tkinter_available,
        'nvml': nvml_available
    }


def test_system_monitoring():
    """Test basic system monitoring functionality."""
    print("\n🔍 Testing system monitoring...")
    
    try:
        import psutil
        
        # CPU test
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"   ✅ CPU monitoring: {cpu_percent:.1f}%")
        
        # Memory test
        memory = psutil.virtual_memory()
        print(f"   ✅ Memory monitoring: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB used)")
        
        # Disk test
        disk_io = psutil.disk_io_counters()
        if disk_io:
            print(f"   ✅ Disk I/O monitoring: Available")
        else:
            print(f"   ⚠️  Disk I/O monitoring: Not available on this system")
        
        # Network test
        network = psutil.net_io_counters()
        if network:
            print(f"   ✅ Network monitoring: Available")
        else:
            print(f"   ⚠️  Network monitoring: Not available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ System monitoring failed: {e}")
        return False


def test_gpu_monitoring():
    """Test GPU monitoring functionality."""
    print("\n🎮 Testing GPU monitoring...")
    
    # Test OpenCV OpenCL
    try:
        import cv2
        opencl_available = cv2.ocl.haveOpenCL()
        print(f"   OpenCV OpenCL: {'✅ Available' if opencl_available else '❌ Not available'}")
        
        if opencl_available:
            device = cv2.ocl.Device.getDefault()
            print(f"   OpenCL Device: {device.name()}")
    except Exception as e:
        print(f"   ⚠️  OpenCV OpenCL test failed: {e}")
    
    # Test NVIDIA ML
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        gpu_count = nvml.nvmlDeviceGetCount()
        print(f"   ✅ NVIDIA ML: {gpu_count} GPU(s) detected")
        
        for i in range(min(gpu_count, 2)):  # Test first 2 GPUs
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"      GPU {i}: {name} ({memory_info.total / (1024**3):.1f}GB)")
            
    except ImportError:
        print(f"   ⚠️  NVIDIA ML not available (install py3nvml for enhanced GPU monitoring)")
    except Exception as e:
        print(f"   ⚠️  NVIDIA ML test failed: {e}")
    
    # Test nvidia-smi fallback
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = result.stdout.strip().split('\n')
            print(f"   ✅ nvidia-smi fallback: {len(gpu_names)} GPU(s)")
        else:
            print(f"   ⚠️  nvidia-smi: Not available or no GPUs found")
    except Exception as e:
        print(f"   ⚠️  nvidia-smi test failed: {e}")


def test_profiler_classes():
    """Test profiler class initialization."""
    print("\n⚙️  Testing profiler classes...")
    
    try:
        # Import the profiler module
        sys.path.append('.')
        from system_profiler import SystemMonitor, ConsoleProfiler
        
        # Test SystemMonitor
        monitor = SystemMonitor()
        print(f"   ✅ SystemMonitor: Initialized successfully")
        
        # Test getting a snapshot
        snapshot = monitor.get_system_snapshot()
        print(f"   ✅ SystemSnapshot: CPU {snapshot.cpu_percent:.1f}%, RAM {snapshot.memory_percent:.1f}%")
        
        # Test ConsoleProfiler
        console_profiler = ConsoleProfiler(monitor)
        print(f"   ✅ ConsoleProfiler: Initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Profiler class test failed: {e}")
        traceback.print_exc()
        return False


def test_opencv_display():
    """Test OpenCV display functionality."""
    print("\n📺 Testing OpenCV display...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        test_image[:] = (50, 50, 50)  # Dark gray background
        
        # Add some text
        cv2.putText(test_image, "Profiler Test", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(test_image, "Press any key to close", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the image
        cv2.imshow("Profiler Display Test", test_image)
        print(f"   ✅ OpenCV window created - press any key in the window to continue")
        
        # Wait for key press with timeout
        key = cv2.waitKey(3000)  # 3 second timeout
        cv2.destroyAllWindows()
        
        if key >= 0:
            print(f"   ✅ OpenCV input detected")
        else:
            print(f"   ⚠️  OpenCV input timeout (window may not be visible)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV display test failed: {e}")
        return False


def test_recommended_usage():
    """Show recommended usage based on available features."""
    print("\n💡 Recommended Usage:")
    
    optional = test_optional_imports()
    
    if optional['matplotlib'] and optional['tkinter']:
        print("   🎨 GUI Mode: RECOMMENDED")
        print("      python system_profiler.py --mode gui")
        print("      python launch_with_profiler.py --mode gui")
    
    print("   📊 OpenCV Mode: AVAILABLE")
    print("      python system_profiler.py --mode opencv")
    print("      python launch_with_profiler.py --profiler-mode opencv")
    
    print("   📝 Console Mode: AVAILABLE")
    print("      python system_profiler.py --mode console --log-file perf.csv")
    
    if optional['nvml']:
        print("   🚀 Enhanced GPU monitoring enabled")
    else:
        print("   ⚠️  Install py3nvml for enhanced GPU monitoring:")
        print("      pip install py3nvml")


def main():
    """Run all tests."""
    print("🧪 System Profiler - Installation Test")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Basic imports (required)
    total_tests += 1
    if test_basic_imports():
        tests_passed += 1
    
    # Optional imports
    test_optional_imports()
    
    # System monitoring (required)
    total_tests += 1
    if test_system_monitoring():
        tests_passed += 1
    
    # GPU monitoring (optional)
    test_gpu_monitoring()
    
    # Profiler classes (required)
    total_tests += 1
    if test_profiler_classes():
        tests_passed += 1
    
    # OpenCV display (required)
    total_tests += 1
    if test_opencv_display():
        tests_passed += 1
    
    # Show recommendations
    test_recommended_usage()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print(f"   ✅ All core tests passed - System profiler is ready to use!")
        print(f"\n🚀 Quick start:")
        print(f"   python profiler_examples.py")
        return 0
    else:
        print(f"   ⚠️  Some tests failed - check error messages above")
        print(f"   Basic functionality should still work with available components")
        return 1


if __name__ == "__main__":
    sys.exit(main())
