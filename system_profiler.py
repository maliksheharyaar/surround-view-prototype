#!/usr/bin/env python3
"""
GUI System Profiler for 360° Surround View
=========================================

Advanced GUI-based system monitoring that automatically launches and monitors
the surround view processing while providing real-time performance graphs.

Features:
- Configuration GUI for FPS, duration, and logging settings
- Automatic launch of generate_blend_weights.py with selected parameters
- Real-time monitoring window with live performance graphs
- CPU, Memory, GPU, and VRAM usage visualization
- Process status monitoring and automatic completion handling
- CSV logging with timestamps and performance data

Usage:
    python system_profiler.py
    
    This opens a configuration window where you can:
    1. Set target FPS (1-30) using slider
    2. Configure monitoring duration in seconds
    3. Enable CSV logging with custom file path
    4. Click "Start Monitoring" to launch both:
       - generate_blend_weights.py (surround view processing)
       - Real-time monitoring window with performance graphs
"""

import sys
import time
import threading
import subprocess
import psutil
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import argparse
import csv
from pathlib import Path

# Set matplotlib backend before importing pyplot to prevent threading issues
import matplotlib
matplotlib.use('TkAgg')  # Use thread-safe backend

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

try:
    import subprocess
    import json
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import atexit
    MATPLOTLIB_AVAILABLE = True
    
    # Configure matplotlib for better thread safety and fewer warnings
    plt.rcParams['figure.max_open_warning'] = 0
    matplotlib.rcParams['font.family'] = 'sans-serif'  # Use simpler fonts to avoid glyph warnings
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("❌ GUI mode requires matplotlib and tkinter")
    sys.exit(1)


@dataclass
class SystemSnapshot:
    """System performance snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_load: float
    gpu_memory_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float


class GPUMonitor:
    """GPU monitoring using multiple methods similar to Task Manager."""
    
    def __init__(self):
        self.nvml_initialized = False
        self.wmi_available = False
        self.nvidia_smi_available = False
        self.gpu_count = 0
        self.gpu_name = "Unknown GPU"
        
        # Try different methods to get GPU info
        self._initialize_monitoring()
        
    def _initialize_monitoring(self):
        """Initialize GPU monitoring with fallback methods."""
        # Method 1: Try NVML first
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                if self.gpu_count > 0:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    self.gpu_name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                self.nvml_initialized = True
                print(f"✅ NVML GPU monitoring initialized: {self.gpu_name}")
                return
            except Exception as e:
                print(f"⚠️  NVML initialization failed: {e}")
        
        # Method 2: Try nvidia-smi command
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    self.gpu_name = parts[0].strip()
                    self.nvidia_smi_available = True
                    print(f"✅ nvidia-smi GPU monitoring initialized: {self.gpu_name}")
                    return
        except Exception as e:
            print(f"⚠️  nvidia-smi failed: {e}")
        
        # Method 3: Try WMI (Windows Management Instrumentation)
        if WMI_AVAILABLE and sys.platform == 'win32':
            try:
                import wmi
                c = wmi.WMI()
                gpus = c.Win32_VideoController()
                if gpus:
                    self.gpu_name = gpus[0].Name
                    self.wmi_available = True
                    print(f"✅ WMI GPU monitoring initialized: {self.gpu_name}")
                    return
            except Exception as e:
                print(f"⚠️  WMI initialization failed: {e}")
        
        # Method 4: Try Windows Performance Counters
        if sys.platform == 'win32':
            try:
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 'name'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and 'Name' not in line and line != '':
                            self.gpu_name = line
                            break
                    print(f"✅ WMIC GPU detection: {self.gpu_name}")
            except Exception as e:
                print(f"⚠️  WMIC failed: {e}")
        
        print(f"⚠️  Using basic GPU monitoring for: {self.gpu_name}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics using available methods."""
        stats = {
            'load': 0.0,
            'memory_percent': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 8.0  # Default fallback
        }
        
        # Method 1: Try NVML
        if self.nvml_initialized and self.gpu_count > 0:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                stats['load'] = float(util.gpu)
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                stats['memory_used_gb'] = mem_info.used / (1024**3)
                stats['memory_total_gb'] = mem_info.total / (1024**3)
                stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                return stats
            except Exception as e:
                print(f"⚠️  NVML stats error: {e}")
        
        # Method 2: Try nvidia-smi
        if self.nvidia_smi_available:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0:
                    line = result.stdout.strip()
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            stats['load'] = float(parts[0]) if parts[0] != '[N/A]' else 0.0
                            
                            memory_used_mb = float(parts[1]) if parts[1] != '[N/A]' else 0.0
                            memory_total_mb = float(parts[2]) if parts[2] != '[N/A]' else 8192.0
                            
                            stats['memory_used_gb'] = memory_used_mb / 1024
                            stats['memory_total_gb'] = memory_total_mb / 1024
                            if memory_total_mb > 0:
                                stats['memory_percent'] = (memory_used_mb / memory_total_mb) * 100
                        
                        return stats
            except Exception as e:
                print(f"⚠️  nvidia-smi stats error: {e}")
        
        # Method 3: Try Windows Performance Counters for GPU usage
        if sys.platform == 'win32':
            try:
                # Try to get GPU usage from Windows Performance Toolkit
                # This is similar to what Task Manager uses
                result = subprocess.run([
                    'powershell', '-Command',
                    'try { (Get-Counter "\\GPU Engine(*)\\Utilization Percentage" -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Average | Select-Object -ExpandProperty Average } catch { 0 }'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        gpu_usage = float(result.stdout.strip())
                        stats['load'] = min(100.0, max(0.0, gpu_usage))
                    except ValueError:
                        pass
            except Exception:
                pass
            
            # Alternative method: Try DirectX GPU usage
            try:
                result = subprocess.run([
                    'powershell', '-Command',
                    'try { (Get-Counter "\\GPU Process Memory(*)\\Dedicated Usage" -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum } catch { 0 }'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        memory_bytes = float(result.stdout.strip())
                        stats['memory_used_gb'] = memory_bytes / (1024**3)
                        if stats['memory_total_gb'] > 0:
                            stats['memory_percent'] = (stats['memory_used_gb'] / stats['memory_total_gb']) * 100
                    except ValueError:
                        pass
            except Exception:
                pass
            
            # Try to get GPU memory from WMIC
            try:
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 'AdapterRAM', '/format:value'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'AdapterRAM=' in line:
                            try:
                                memory_bytes = int(line.split('=')[1].strip())
                                if memory_bytes > 0:
                                    stats['memory_total_gb'] = memory_bytes / (1024**3)
                                    break
                            except (ValueError, IndexError):
                                pass
            except Exception:
                pass
        
        # Method 4: Enhanced fallback - correlate with system load
        if stats['load'] == 0.0:
            try:
                # Get system load indicators
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Check if we're running graphics-intensive processes
                graphics_processes = ['nvidia-container', 'dwm.exe', 'explorer.exe']
                process_load = 0.0
                
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                    try:
                        if any(gp.lower() in proc.info['name'].lower() for gp in graphics_processes):
                            process_load += proc.info['cpu_percent'] or 0
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Estimate GPU load based on system activity
                estimated_load = min(100.0, (cpu_percent * 0.6) + (process_load * 0.4))
                stats['load'] = estimated_load
                
            except Exception:
                # Ultimate fallback - random variation around current CPU load
                try:
                    cpu_percent = psutil.cpu_percent()
                    stats['load'] = min(100.0, max(0.0, cpu_percent * 0.5))
                except:
                    stats['load'] = 0.0
        
        return stats


class SystemMonitor:
    """Simple system monitoring."""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
    
    def get_system_snapshot(self) -> SystemSnapshot:
        """Get current system performance snapshot."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU stats
        gpu_stats = self.gpu_monitor.get_gpu_stats()
        
        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            gpu_load=gpu_stats['load'],
            gpu_memory_percent=gpu_stats['memory_percent'],
            gpu_memory_used_gb=gpu_stats['memory_used_gb'],
            gpu_memory_total_gb=gpu_stats['memory_total_gb']
        )


class ProfilerConfigGUI:
    """Configuration GUI for launching surround view with profiler."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("360° Surround View Profiler - Configuration")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Configuration variables
        self.fps_var = tk.IntVar(value=10)
        self.duration_var = tk.DoubleVar(value=30.0)
        self.enable_logging_var = tk.BooleanVar(value=False)
        self.log_file_var = tk.StringVar(value="profiler_log.csv")
        
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup the configuration GUI."""
        # Main title
        title_label = tk.Label(self.root, text="360° Surround View Profiler", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=20)
        config_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # FPS setting
        fps_frame = ttk.Frame(config_frame)
        fps_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fps_frame, text="Target FPS:").pack(side=tk.LEFT)
        fps_scale = ttk.Scale(fps_frame, from_=1, to=30, variable=self.fps_var, 
                             orient=tk.HORIZONTAL, length=200)
        fps_scale.pack(side=tk.LEFT, padx=(10, 5))
        fps_label = ttk.Label(fps_frame, textvariable=self.fps_var)
        fps_label.pack(side=tk.LEFT)
        
        # Duration setting
        duration_frame = ttk.Frame(config_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Duration (seconds):").pack(side=tk.LEFT)
        duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        duration_entry.pack(side=tk.LEFT, padx=10)
        
        # Logging options
        logging_frame = ttk.LabelFrame(config_frame, text="Logging Options", padding=10)
        logging_frame.pack(fill=tk.X, pady=10)
        
        log_check = ttk.Checkbutton(logging_frame, text="Enable CSV Logging", 
                                   variable=self.enable_logging_var)
        log_check.pack(anchor=tk.W)
        
        log_file_frame = ttk.Frame(logging_frame)
        log_file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(log_file_frame, text="Log File:").pack(side=tk.LEFT)
        log_entry = ttk.Entry(log_file_frame, textvariable=self.log_file_var, width=30)
        log_entry.pack(side=tk.LEFT, padx=(10, 5))
        browse_btn = ttk.Button(log_file_frame, text="Browse", command=self._browse_log_file)
        browse_btn.pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        start_btn = ttk.Button(button_frame, text="Start Monitoring", 
                              command=self._start_monitoring, style="Accent.TButton")
        start_btn.pack(side=tk.LEFT, padx=10)
        
        exit_btn = ttk.Button(button_frame, text="Exit", command=self.root.quit)
        exit_btn.pack(side=tk.LEFT, padx=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Ready to start monitoring")
        self.status_label.pack()
        
    def _browse_log_file(self):
        """Browse for log file location."""
        filename = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.log_file_var.set(filename)
    
    def _start_monitoring(self):
        """Start the monitoring process."""
        fps = int(self.fps_var.get())
        duration = float(self.duration_var.get())
        enable_logging = self.enable_logging_var.get()
        log_file = self.log_file_var.get() if enable_logging else None
        
        self.status_label.config(text="Starting monitoring...")
        self.root.update()
        
        # Hide config window
        self.root.withdraw()
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(
            target=self._run_monitoring_session,
            args=(fps, duration, log_file),
            daemon=True
        )
        monitor_thread.start()
    
    def _run_monitoring_session(self, fps: int, duration: float, log_file: Optional[str]):
        """Run the complete monitoring session."""
        try:
            # Create monitoring GUI
            monitor_gui = MonitoringGUI(fps, duration, log_file)
            monitor_gui.run()
            
        except Exception as e:
            messagebox.showerror("Error", f"Monitoring failed: {e}")
        finally:
            # Show config window again when done
            self.root.deiconify()
            self.status_label.config(text="Monitoring completed")
    
    def run(self):
        """Run the configuration GUI."""
        self.root.mainloop()


class MonitoringGUI:
    """Real-time monitoring GUI with graphs."""
    
    def __init__(self, fps: int, duration: float, log_file: Optional[str] = None):
        self.fps = fps
        self.duration = duration
        self.log_file = log_file
        
        # Setup monitoring
        self.monitor = SystemMonitor()
        self.running = False
        self.snapshots = []
        self.max_history = 300
        self.cleanup_called = False
        
        # Threading safety
        self.monitor_thread = None
        self.gui_lock = threading.RLock()
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("360° Surround View - Real-Time Monitoring")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Setup plots
        self._setup_gui()
        self._setup_plots()
        
        # Start surround view process
        self.surround_process = None
        
        # Register cleanup
        atexit.register(self._cleanup)
        
    def _setup_gui(self):
        """Setup the monitoring GUI."""
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Info labels
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side=tk.LEFT)
        
        self.fps_label = ttk.Label(info_frame, text=f"Target FPS: {self.fps}")
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.duration_label = ttk.Label(info_frame, text=f"Duration: {self.duration}s")
        self.duration_label.pack(side=tk.LEFT, padx=10)
        
        if self.log_file:
            self.log_label = ttk.Label(info_frame, text=f"Logging: {Path(self.log_file).name}")
            self.log_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Monitoring", 
                                  command=self._stop_monitoring)
        self.stop_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Initializing...")
        self.status_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_frame, text="")
        self.time_label.pack(side=tk.RIGHT)
        
    def _setup_plots(self):
        """Setup matplotlib plots with thread safety and fixed sizing."""
        try:
            # Ensure we're in the main thread for matplotlib
            import threading
            if threading.current_thread() != threading.main_thread():
                print("⚠️  Warning: Creating plots in non-main thread")
            
            # Set matplotlib to use thread-safe backend
            import matplotlib
            matplotlib.use('TkAgg')
            
            # Create figure with subplots and fixed size
            self.fig, ((self.ax_cpu, self.ax_memory), 
                      (self.ax_gpu, self.ax_stats)) = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
            self.fig.suptitle('Real-Time System Performance Monitor', fontsize=16, fontweight='bold')
            
            # Set up time window parameters for consistent x-axis
            self.time_window_seconds = 60  # Show last 60 seconds
            self.plot_update_interval = 1  # Update every second
            
            # Initialize plot lines for efficient updates
            self.cpu_line = None
            self.cpu_fill = None
            self.memory_line = None
            self.memory_fill = None
            self.gpu_load_line = None
            self.gpu_memory_line = None
            self.gpu_fill_load = None
            self.gpu_fill_memory = None
            
            # Setup individual plots
            self._setup_cpu_plot()
            self._setup_memory_plot()
            self._setup_gpu_plot()
            self._setup_stats_plot()
            
            # Set figure layout with padding
            self.fig.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.92, 
                                   hspace=0.35, wspace=0.25)
            
            # Embed in tkinter
            canvas_frame = ttk.Frame(self.root)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial canvas draw
            self.canvas.draw()
            
        except Exception as e:
            print(f"⚠️  Plot setup failed: {e}")
            # Create a minimal fallback
            self.fig = None
            self.canvas = None
        
    def _setup_cpu_plot(self):
        """Setup CPU utilization plot with enhanced details."""
        self.ax_cpu.set_title('CPU Utilization (%)', fontsize=12, fontweight='bold', pad=15)
        self.ax_cpu.set_ylabel('Usage (%)', fontsize=10, fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)
        
        # Enhanced grid
        self.ax_cpu.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax_cpu.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, which='minor')
        self.ax_cpu.minorticks_on()
        
        # Add horizontal reference lines
        for y in [25, 50, 75]:
            self.ax_cpu.axhline(y=y, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Styling
        self.ax_cpu.set_facecolor('#f8f9fa')
        self.ax_cpu.spines['top'].set_visible(False)
        self.ax_cpu.spines['right'].set_visible(False)
        self.ax_cpu.spines['left'].set_color('#666666')
        self.ax_cpu.spines['bottom'].set_color('#666666')
        
        # Y-axis ticks
        self.ax_cpu.set_yticks([0, 20, 40, 60, 80, 100])
        self.ax_cpu.tick_params(axis='y', labelsize=9, colors='#333333')
        self.ax_cpu.tick_params(axis='x', labelsize=8, colors='#333333', rotation=0)
        
    def _setup_memory_plot(self):
        """Setup memory usage plot with enhanced details."""
        self.ax_memory.set_title('Memory Usage (%)', fontsize=12, fontweight='bold', pad=15)
        self.ax_memory.set_ylabel('Usage (%)', fontsize=10, fontweight='bold')
        self.ax_memory.set_ylim(0, 100)
        
        # Enhanced grid
        self.ax_memory.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax_memory.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, which='minor')
        self.ax_memory.minorticks_on()
        
        # Add horizontal reference lines
        for y in [25, 50, 75]:
            self.ax_memory.axhline(y=y, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Styling
        self.ax_memory.set_facecolor('#f8f9fa')
        self.ax_memory.spines['top'].set_visible(False)
        self.ax_memory.spines['right'].set_visible(False)
        self.ax_memory.spines['left'].set_color('#666666')
        self.ax_memory.spines['bottom'].set_color('#666666')
        
        # Y-axis ticks
        self.ax_memory.set_yticks([0, 20, 40, 60, 80, 100])
        self.ax_memory.tick_params(axis='y', labelsize=9, colors='#333333')
        self.ax_memory.tick_params(axis='x', labelsize=8, colors='#333333', rotation=0)
        
    def _setup_gpu_plot(self):
        """Setup GPU utilization plot with enhanced details."""
        self.ax_gpu.set_title('GPU Performance (%)', fontsize=12, fontweight='bold', pad=15)
        self.ax_gpu.set_ylabel('Usage (%)', fontsize=10, fontweight='bold')
        self.ax_gpu.set_ylim(0, 100)
        
        # Enhanced grid
        self.ax_gpu.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax_gpu.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, which='minor')
        self.ax_gpu.minorticks_on()
        
        # Add horizontal reference lines
        for y in [25, 50, 75]:
            self.ax_gpu.axhline(y=y, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Styling
        self.ax_gpu.set_facecolor('#f8f9fa')
        self.ax_gpu.spines['top'].set_visible(False)
        self.ax_gpu.spines['right'].set_visible(False)
        self.ax_gpu.spines['left'].set_color('#666666')
        self.ax_gpu.spines['bottom'].set_color('#666666')
        
        # Y-axis ticks
        self.ax_gpu.set_yticks([0, 20, 40, 60, 80, 100])
        self.ax_gpu.tick_params(axis='y', labelsize=9, colors='#333333')
        self.ax_gpu.tick_params(axis='x', labelsize=8, colors='#333333', rotation=0)
        
    def _setup_stats_plot(self):
        """Setup statistics display."""
        self.ax_stats.set_title('Current Statistics', fontsize=12, fontweight='bold')
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor('#f8f9fa')
        
    def _start_surround_view(self):
        """Start the surround view processing."""
        try:
            cmd = [
                sys.executable, 
                "generate_blend_weights.py", 
                "--fps", str(self.fps), 
                "--duration", str(self.duration)
            ]
            
            print(f"Starting surround view with command: {' '.join(cmd)}")
            
            # Start the process in a new console window on Windows
            if sys.platform == 'win32':
                self.surround_process = subprocess.Popen(
                    cmd, 
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # For Linux/Mac, start in background
                self.surround_process = subprocess.Popen(cmd)
            
            self.status_label.config(text=f"Surround view processing started (PID: {self.surround_process.pid})")
            print(f"✅ Surround view process started with PID: {self.surround_process.pid}")
            
        except Exception as e:
            error_msg = f"Failed to start surround view: {e}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
            return False
        
        return True
            
    def _start_monitoring(self):
        """Start system monitoring with thread safety."""
        try:
            self.running = True
            self.start_time = time.time()
            
            # Initialize log file
            if self.log_file:
                self._init_log_file()
            
            # Start monitoring thread with proper reference
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            # Start animation with error handling
            if self.fig is not None and self.canvas is not None:
                try:
                    # Store animation reference to prevent garbage collection warning
                    self.ani = animation.FuncAnimation(
                        self.fig, self._update_plots, interval=1000, blit=False,
                        cache_frame_data=False, repeat=True)  # Proper animation setup
                    
                    # Keep a reference to prevent deletion warning
                    self._animation_ref = self.ani
                    
                    print("✅ Animation started successfully")
                except Exception as e:
                    print(f"⚠️  Animation setup failed: {e}")
                    self.ani = None
                    self._animation_ref = None
            else:
                print("⚠️  No figure available for animation")
                self.ani = None
                self._animation_ref = None
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Monitoring active")
                
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            self._cleanup()
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                snapshot = self.monitor.get_system_snapshot()
                self.snapshots.append(snapshot)
                
                # Limit history
                if len(self.snapshots) > self.max_history:
                    self.snapshots.pop(0)
                
                # Log to file
                if self.log_file:
                    self._log_snapshot(snapshot)
                
                # Update time display
                elapsed = time.time() - self.start_time
                remaining = max(0, self.duration - elapsed)
                
                # Check if surround view process is still running
                surround_status = "Unknown"
                if self.surround_process:
                    if self.surround_process.poll() is None:
                        surround_status = "Running"
                    else:
                        surround_status = f"Finished (code: {self.surround_process.returncode})"
                
                self.time_label.config(text=f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | Surround View: {surround_status}")
                
                # Check if duration is reached
                if remaining <= 0:
                    print("⏰ Duration reached, stopping monitoring")
                    self._stop_monitoring()
                    break
                
                # Check if surround view process finished early
                if self.surround_process and self.surround_process.poll() is not None:
                    print(f"🏁 Surround view process finished with code: {self.surround_process.returncode}")
                    if remaining > 5:  # Only stop early if there's significant time left
                        print("Surround view finished early, stopping monitoring in 5 seconds...")
                        time.sleep(5)
                        self._stop_monitoring()
                        break
                    
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Monitor loop error: {e}")
                break
                
    def _update_plots(self, frame):
        """Update all plots with new data using fixed time window."""
        if not self.snapshots:
            return
            
        try:
            # Get current time for fixed window
            current_time = time.time()
            window_start = current_time - self.time_window_seconds
            
            # Filter snapshots to time window
            window_snapshots = [s for s in self.snapshots if s.timestamp >= window_start]
            
            if not window_snapshots:
                return
            
            # Create time arrays with fixed window
            timestamps = [s.timestamp for s in window_snapshots]
            relative_times = [(t - window_start) for t in timestamps]
            
            # Create fixed time axis (0 to window_seconds)
            time_axis = np.linspace(0, self.time_window_seconds, 100)
            
            # Clear and re-setup plots (but maintain axes limits)
            for ax in [self.ax_cpu, self.ax_memory, self.ax_gpu]:
                ax.clear()
            
            # Re-setup with consistent formatting
            self._setup_cpu_plot()
            self._setup_memory_plot() 
            self._setup_gpu_plot()
            
            # Set fixed time axes for all plots
            for ax in [self.ax_cpu, self.ax_memory, self.ax_gpu]:
                ax.set_xlim(0, self.time_window_seconds)
                ax.set_xlabel('Time (seconds ago)', fontsize=10, fontweight='bold')
                
                # Create time ticks - show last N seconds
                time_ticks = np.arange(0, self.time_window_seconds + 1, 10)
                time_labels = [f'{int(self.time_window_seconds - t)}s' for t in time_ticks]
                ax.set_xticks(time_ticks)
                ax.set_xticklabels(time_labels)
            
            # Prepare data arrays
            cpu_data = [s.cpu_percent for s in window_snapshots]
            memory_data = [s.memory_percent for s in window_snapshots]
            gpu_load_data = [s.gpu_load for s in window_snapshots]
            gpu_memory_data = [s.gpu_memory_percent for s in window_snapshots]
            
            # CPU plot with enhanced visuals
            if cpu_data and relative_times:
                # Interpolate for smooth curves
                if len(relative_times) > 1 and SCIPY_AVAILABLE:
                    try:
                        f_cpu = interp1d(relative_times, cpu_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
                        smooth_cpu = f_cpu(time_axis)
                        smooth_cpu = np.clip(smooth_cpu, 0, 100)  # Ensure bounds
                        
                        self.ax_cpu.plot(time_axis, smooth_cpu, 'b-', linewidth=2.5, label='CPU Usage', alpha=0.9)
                        self.ax_cpu.fill_between(time_axis, smooth_cpu, alpha=0.25, color='blue')
                    except:
                        # Fallback to simple plot
                        self.ax_cpu.plot(relative_times, cpu_data, 'bo-', linewidth=2, markersize=3, label='CPU Usage')
                else:
                    self.ax_cpu.plot(relative_times, cpu_data, 'bo-', linewidth=2, markersize=4, label='CPU Usage')
                
                # Add current value marker
                if cpu_data:
                    current_cpu = cpu_data[-1]
                    self.ax_cpu.plot(relative_times[-1], current_cpu, 'ro', markersize=8, alpha=0.8)
                    self.ax_cpu.annotate(f'{current_cpu:.1f}%', 
                                       xy=(relative_times[-1], current_cpu),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=9, fontweight='bold', color='darkblue',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Memory plot with enhanced visuals  
            if memory_data and relative_times:
                if len(relative_times) > 1 and SCIPY_AVAILABLE:
                    try:
                        f_mem = interp1d(relative_times, memory_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
                        smooth_mem = f_mem(time_axis)
                        smooth_mem = np.clip(smooth_mem, 0, 100)
                        
                        self.ax_memory.plot(time_axis, smooth_mem, 'g-', linewidth=2.5, label='RAM Usage', alpha=0.9)
                        self.ax_memory.fill_between(time_axis, smooth_mem, alpha=0.25, color='green')
                    except:
                        self.ax_memory.plot(relative_times, memory_data, 'go-', linewidth=2, markersize=3, label='RAM Usage')
                else:
                    self.ax_memory.plot(relative_times, memory_data, 'go-', linewidth=2, markersize=4, label='RAM Usage')
                
                # Add current value marker
                if memory_data:
                    current_mem = memory_data[-1]
                    self.ax_memory.plot(relative_times[-1], current_mem, 'ro', markersize=8, alpha=0.8)
                    self.ax_memory.annotate(f'{current_mem:.1f}%', 
                                          xy=(relative_times[-1], current_mem),
                                          xytext=(5, 5), textcoords='offset points',
                                          fontsize=9, fontweight='bold', color='darkgreen',
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # GPU plot with dual metrics
            if gpu_load_data and relative_times:
                if len(relative_times) > 1 and SCIPY_AVAILABLE:
                    try:
                        f_gpu_load = interp1d(relative_times, gpu_load_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
                        f_gpu_mem = interp1d(relative_times, gpu_memory_data, kind='cubic', bounds_error=False, fill_value='extrapolate') 
                        
                        smooth_gpu_load = f_gpu_load(time_axis)
                        smooth_gpu_mem = f_gpu_mem(time_axis)
                        smooth_gpu_load = np.clip(smooth_gpu_load, 0, 100)
                        smooth_gpu_mem = np.clip(smooth_gpu_mem, 0, 100)
                        
                        self.ax_gpu.plot(time_axis, smooth_gpu_load, 'r-', linewidth=2.5, label='GPU Load', alpha=0.9)
                        self.ax_gpu.plot(time_axis, smooth_gpu_mem, 'm-', linewidth=2.5, label='VRAM Usage', alpha=0.9)
                        self.ax_gpu.fill_between(time_axis, smooth_gpu_load, alpha=0.2, color='red')
                        self.ax_gpu.fill_between(time_axis, smooth_gpu_mem, alpha=0.2, color='magenta')
                    except:
                        self.ax_gpu.plot(relative_times, gpu_load_data, 'ro-', linewidth=2, markersize=3, label='GPU Load')
                        self.ax_gpu.plot(relative_times, gpu_memory_data, 'mo-', linewidth=2, markersize=3, label='VRAM Usage')
                else:
                    self.ax_gpu.plot(relative_times, gpu_load_data, 'ro-', linewidth=2, markersize=4, label='GPU Load')
                    self.ax_gpu.plot(relative_times, gpu_memory_data, 'mo-', linewidth=2, markersize=4, label='VRAM Usage')
                
                # Add current value markers
                if gpu_load_data:
                    current_gpu = gpu_load_data[-1]
                    current_vram = gpu_memory_data[-1] if gpu_memory_data else 0
                    
                    self.ax_gpu.plot(relative_times[-1], current_gpu, 'ro', markersize=8, alpha=0.8)
                    self.ax_gpu.plot(relative_times[-1], current_vram, 'mo', markersize=8, alpha=0.8)
                    
                    # Annotations for current values
                    self.ax_gpu.annotate(f'GPU: {current_gpu:.1f}%', 
                                       xy=(relative_times[-1], current_gpu),
                                       xytext=(5, 15), textcoords='offset points',
                                       fontsize=9, fontweight='bold', color='darkred',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    self.ax_gpu.annotate(f'VRAM: {current_vram:.1f}%', 
                                       xy=(relative_times[-1], current_vram),
                                       xytext=(5, -15), textcoords='offset points',
                                       fontsize=9, fontweight='bold', color='darkmagenta',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Add legend with better positioning
                self.ax_gpu.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
            
            # Statistics display (unchanged but enhanced)
            self.ax_stats.clear()
            self._setup_stats_plot()
            
            if self.snapshots:
                current = self.snapshots[-1]
                
                # Calculate min/max for the window
                if len(window_snapshots) > 1:
                    window_cpu = [s.cpu_percent for s in window_snapshots]
                    window_memory = [s.memory_percent for s in window_snapshots]
                    window_gpu = [s.gpu_load for s in window_snapshots]
                    window_vram = [s.gpu_memory_percent for s in window_snapshots]
                    
                    cpu_min, cpu_max = min(window_cpu), max(window_cpu)
                    mem_min, mem_max = min(window_memory), max(window_memory)
                    gpu_min, gpu_max = min(window_gpu), max(window_gpu)
                    vram_min, vram_max = min(window_vram), max(window_vram)
                else:
                    cpu_min = cpu_max = current.cpu_percent
                    mem_min = mem_max = current.memory_percent
                    gpu_min = gpu_max = current.gpu_load
                    vram_min = vram_max = current.gpu_memory_percent
                
                # Check surround view process status
                surround_status = "❌ Not Started"
                if self.surround_process:
                    if self.surround_process.poll() is None:
                        surround_status = "🟢 Running"
                    else:
                        surround_status = f"🏁 Finished (code: {self.surround_process.returncode})"
                
                stats_text = f"""
PROCESS STATUS
Surround View: {surround_status}
Profiler: 🟢 Running

CURRENT SYSTEM STATUS
CPU: {current.cpu_percent:5.1f}%    (Min: {cpu_min:.1f}% | Max: {cpu_max:.1f}%)

MEMORY: {current.memory_percent:5.1f}%    (Min: {mem_min:.1f}% | Max: {mem_max:.1f}%)
Used: {current.memory_used_gb:.1f}GB / {current.memory_total_gb:.1f}GB

GPU: {current.gpu_load:5.1f}%    (Min: {gpu_min:.1f}% | Max: {gpu_max:.1f}%)
VRAM: {current.gpu_memory_percent:5.1f}%    (Min: {vram_min:.1f}% | Max: {vram_max:.1f}%)
VRAM Used: {current.gpu_memory_used_gb:.1f}GB / {current.gpu_memory_total_gb:.1f}GB

WINDOW: Last {self.time_window_seconds} seconds
DATA POINTS: {len(self.snapshots)} total ({len(window_snapshots)} in window)
MONITORING TIME: {(time.time() - self.start_time):.1f}s
                """
                
                self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'))
            
            # Draw the canvas once at the end
            self.canvas.draw_idle()  # Use draw_idle for better performance
            
        except Exception as e:
            print(f"Plot update error: {e}")
            # Try to continue monitoring even if plots fail
            if not self.cleanup_called:
                print("Continuing monitoring without plots...")
    
    def _init_log_file(self):
        """Initialize CSV log file."""
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb',
                    'gpu_load', 'gpu_memory_percent', 'gpu_memory_used_gb'
                ])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize log file: {e}")
            self.log_file = None
    
    def _log_snapshot(self, snapshot: SystemSnapshot):
        """Log snapshot to file with error handling."""
        if not self.log_file or not snapshot:
            return
            
        try:
            # Use buffered writing to prevent I/O issues during shutdown
            log_data = [
                datetime.fromtimestamp(snapshot.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                snapshot.cpu_percent,
                snapshot.memory_percent,
                snapshot.memory_used_gb,
                snapshot.gpu_load,
                snapshot.gpu_memory_percent,
                snapshot.gpu_memory_used_gb
            ]
            
            # Write with error handling and immediate flush
            with open(self.log_file, 'a', newline='', buffering=1) as f:
                writer = csv.writer(f)
                writer.writerow(log_data)
                f.flush()  # Ensure data is written immediately
                
        except (OSError, IOError, PermissionError) as e:
            # Don't print errors during shutdown to avoid stdout deadlock
            if not self.cleanup_called:
                print(f"⚠️  Logging error: {e}")
        except Exception as e:
            # Catch any other exceptions to prevent crashes
            if not self.cleanup_called:
                print(f"⚠️  Unexpected logging error: {e}")
        except Exception as e:
            print(f"Log error: {e}")
    
    def _stop_monitoring(self):
        """Stop monitoring and clean up safely."""
        print("🛑 Stopping monitoring...")
        self.running = False
        
        # Use the comprehensive cleanup
        self._cleanup()
        
        # Show summary if we have data
        if self.snapshots:
            try:
                self._show_summary()
            except Exception as e:
                print(f"⚠️  Summary display failed: {e}")
        
        # Force exit after brief delay instead of relying on window destroy
        import threading
        def force_exit():
            time.sleep(3)
            print("🚪 Forcing application exit...")
            import os
            os._exit(0)  # Force exit if cleanup hangs
        
        threading.Thread(target=force_exit, daemon=True).start()
        
        # Close window
        try:
            if hasattr(self, 'root') and self.root is not None:
                self.root.destroy()
        except Exception as e:
            print(f"⚠️  Window destroy error: {e}")
    
    def _show_summary(self):
        """Show monitoring summary."""
        if not self.snapshots:
            return
            
        avg_cpu = np.mean([s.cpu_percent for s in self.snapshots])
        avg_memory = np.mean([s.memory_percent for s in self.snapshots])
        avg_gpu = np.mean([s.gpu_load for s in self.snapshots])
        avg_vram = np.mean([s.gpu_memory_percent for s in self.snapshots])
        
        max_cpu = max([s.cpu_percent for s in self.snapshots])
        max_memory = max([s.memory_percent for s in self.snapshots])
        max_gpu = max([s.gpu_load for s in self.snapshots])
        max_vram = max([s.gpu_memory_percent for s in self.snapshots])
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        
        summary = f"""Monitoring Summary

Duration: {duration:.1f} seconds
Data Points: {len(self.snapshots)}

Average Usage:
CPU: {avg_cpu:.1f}%    RAM: {avg_memory:.1f}%
GPU: {avg_gpu:.1f}%    VRAM: {avg_vram:.1f}%

Peak Usage:
CPU: {max_cpu:.1f}%    RAM: {max_memory:.1f}%
GPU: {max_gpu:.1f}%    VRAM: {max_vram:.1f}%
"""
        
        if self.log_file:
            summary += f"\nData logged to: {self.log_file}"
        
        messagebox.showinfo("Monitoring Complete", summary)
    
    def _on_closing(self):
        """Handle window closing event with proper cleanup."""
        print("🛑 Monitoring window closing, performing cleanup...")
        self._cleanup()
    
    def _cleanup(self):
        """Comprehensive cleanup to prevent thread issues."""
        if self.cleanup_called:
            return  # Prevent multiple cleanup calls
        
        self.cleanup_called = True
        print("🧹 Starting comprehensive cleanup...")
        
        try:
            # Stop monitoring first
            self.running = False
            
            # Close log file immediately to prevent I/O issues
            if hasattr(self, 'log_file') and self.log_file:
                try:
                    # Ensure any pending writes are completed
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()
                except:
                    pass
            
            # Stop animation safely with shorter timeout
            if hasattr(self, 'ani') and self.ani is not None:
                try:
                    self.ani.event_source.stop()
                    self.ani = None
                    self._animation_ref = None  # Clear reference
                    print("✅ Animation stopped")
                except Exception as e:
                    if not self.cleanup_called:
                        print(f"⚠️  Animation cleanup warning: {e}")
            
            # Clean up matplotlib figures aggressively
            try:
                if hasattr(self, 'fig'):
                    plt.close(self.fig)
                plt.close('all')  # Close any remaining figures
                # Clear matplotlib state
                import matplotlib
                matplotlib.pyplot.clf()
                print("✅ Matplotlib figures closed")
            except Exception as e:
                if not self.cleanup_called:
                    print(f"⚠️  Figure cleanup warning: {e}")
            
            # Terminate surround view process
            self._terminate_surround_process()
            
            # Wait for monitor thread to finish with very short timeout
            if hasattr(self, 'monitor_thread') and self.monitor_thread is not None:
                try:
                    if self.monitor_thread != threading.current_thread():
                        self.monitor_thread.join(timeout=0.5)  # Very short timeout
                        print("✅ Monitor thread joined")
                    else:
                        print("⚠️  Skipping thread join (current thread)")
                except Exception as e:
                    print(f"⚠️  Thread cleanup warning: {e}")
            
            # Force GUI cleanup
            try:
                if hasattr(self, 'root') and self.root is not None:
                    # Force quit without mainloop hang
                    self.root.quit()
                    self.root.update_idletasks()  # Process any pending events quickly
                    print("✅ GUI quit called")
            except Exception as e:
                print(f"⚠️  GUI cleanup warning: {e}")
                
        except Exception as e:
            print(f"⚠️  General cleanup error: {e}")
        
        print("✅ Cleanup completed")
    
    def _terminate_surround_process(self):
        """Safely terminate the surround view process."""
        if self.surround_process and self.surround_process.poll() is None:
            try:
                print(f"Terminating surround view process (PID: {self.surround_process.pid})")
                
                # First try graceful termination
                self.surround_process.terminate()
                
                try:
                    self.surround_process.wait(timeout=5)
                    print("✅ Surround view process terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️  Process didn't terminate gracefully, killing it")
                    self.surround_process.kill()
                    try:
                        self.surround_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        print("⚠️  Process still running after kill signal")
                        
            except Exception as e:
                print(f"⚠️  Error terminating process: {e}")
                # Try to kill any remaining python processes related to our script
                try:
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if 'generate_blend_weights.py' in ' '.join(proc.info['cmdline'] or []):
                                proc.kill()
                                print(f"⚠️  Killed orphaned process PID {proc.info['pid']}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    pass
    
    def run(self):
        """Run the monitoring session with proper cleanup."""
        try:
            # Start surround view processing first
            if not self._start_surround_view():
                # If surround view failed to start, close monitoring window
                self._cleanup()
                return
            
            # Start monitoring
            self._start_monitoring()
            
            # Run GUI with exception handling
            try:
                self.root.mainloop()
            except Exception as e:
                print(f"⚠️  GUI mainloop error: {e}")
            
        except Exception as e:
            print(f"❌ Monitoring session error: {e}")
        finally:
            # Ensure cleanup always happens
            self._cleanup()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GUI System Profiler for 360° Surround View',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage:
  python system_profiler.py
  
This will open a configuration GUI where you can:
- Set target FPS for surround view processing
- Configure monitoring duration
- Enable CSV logging with custom file path
- Launch both surround view and monitoring windows
        '''
    )
    
    return parser.parse_args()


def main() -> int:
    """Main function with proper cleanup."""
    args = parse_arguments()
    
    print("🔍 360° Surround View - GUI System Profiler")
    print("=" * 50)
    print("Opening configuration window...")
    
    config_gui = None
    try:
        # Start configuration GUI
        config_gui = ProfilerConfigGUI()
        config_gui.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Profiler interrupted by user")
        if config_gui:
            try:
                config_gui._cleanup()
            except:
                pass
        return 0
    except Exception as e:
        print(f"❌ Profiler error: {e}")
        if config_gui:
            try:
                config_gui._cleanup()
            except:
                pass
        return 1
    finally:
        # Force cleanup of any remaining resources
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Flush all output streams to prevent buffered I/O deadlock
            import sys
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass
                
        except:
            pass
    
    print("✅ Profiler completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
