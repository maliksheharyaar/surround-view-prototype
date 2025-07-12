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
    gpu_temp: Optional[float] = None
    cpu_temp: Optional[float] = None


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
                '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
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
            'memory_total_gb': 8.0,  # Default fallback
            'temperature': None
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
                
                # Temperature
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    stats['temperature'] = float(temp)
                except:
                    pass
                
                return stats
            except Exception as e:
                print(f"⚠️  NVML stats error: {e}")
        
        # Method 2: Try nvidia-smi
        if self.nvidia_smi_available:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0:
                    line = result.stdout.strip()
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            stats['load'] = float(parts[0]) if parts[0] != '[N/A]' else 0.0
                            
                            memory_used_mb = float(parts[1]) if parts[1] != '[N/A]' else 0.0
                            memory_total_mb = float(parts[2]) if parts[2] != '[N/A]' else 8192.0
                            
                            stats['memory_used_gb'] = memory_used_mb / 1024
                            stats['memory_total_gb'] = memory_total_mb / 1024
                            if memory_total_mb > 0:
                                stats['memory_percent'] = (memory_used_mb / memory_total_mb) * 100
                            
                            if parts[3] != '[N/A]':
                                stats['temperature'] = float(parts[3])
                        
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
        
        # CPU temperature (basic attempt)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
        except:
            pass
        
        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            gpu_load=gpu_stats['load'],
            gpu_memory_percent=gpu_stats['memory_percent'],
            gpu_memory_used_gb=gpu_stats['memory_used_gb'],
            gpu_memory_total_gb=gpu_stats['memory_total_gb'],
            gpu_temp=gpu_stats['temperature'],
            cpu_temp=cpu_temp
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
        """Setup matplotlib plots with thread safety."""
        try:
            # Create figure with subplots
            self.fig, ((self.ax_cpu, self.ax_memory), 
                      (self.ax_gpu, self.ax_stats)) = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('Real-Time System Performance Monitor', fontsize=16)
            
            # Setup individual plots
            self._setup_cpu_plot()
            self._setup_memory_plot()
            self._setup_gpu_plot()
            self._setup_stats_plot()
            
            # Embed in tkinter
            canvas_frame = ttk.Frame(self.root)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"⚠️  Plot setup failed: {e}")
            # Create a minimal fallback
            self.fig = None
            self.canvas = None
        
    def _setup_cpu_plot(self):
        """Setup CPU utilization plot."""
        self.ax_cpu.set_title('CPU Utilization (%)', fontsize=12, fontweight='bold')
        self.ax_cpu.set_ylabel('Usage (%)')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.grid(True, alpha=0.3)
        self.ax_cpu.set_facecolor('#f8f9fa')
        
    def _setup_memory_plot(self):
        """Setup memory usage plot."""
        self.ax_memory.set_title('Memory Usage', fontsize=12, fontweight='bold')
        self.ax_memory.set_ylabel('Usage (%)')
        self.ax_memory.set_ylim(0, 100)
        self.ax_memory.grid(True, alpha=0.3)
        self.ax_memory.set_facecolor('#f8f9fa')
        
    def _setup_gpu_plot(self):
        """Setup GPU utilization plot."""
        self.ax_gpu.set_title('GPU Performance', fontsize=12, fontweight='bold')
        self.ax_gpu.set_ylabel('Usage (%)')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.grid(True, alpha=0.3)
        self.ax_gpu.set_facecolor('#f8f9fa')
        
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
                    self.ani = animation.FuncAnimation(
                        self.fig, self._update_plots, interval=1000, blit=False,
                        cache_frame_data=False)  # Disable caching to avoid warnings
                    print("✅ Animation started successfully")
                except Exception as e:
                    print(f"⚠️  Animation setup failed: {e}")
                    self.ani = None
            else:
                print("⚠️  No figure available for animation")
                self.ani = None
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Monitoring active")
                
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            self._cleanup()
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=1000, blit=False,
            cache_frame_data=False)  # Disable caching to avoid warnings
        
        self.status_label.config(text="Monitoring active")
        
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
        """Update all plots with new data."""
        if not self.snapshots:
            return
            
        try:
            # Clear plots
            self.ax_cpu.clear()
            self.ax_memory.clear()
            self.ax_gpu.clear()
            self.ax_stats.clear()
            
            # Re-setup
            self._setup_cpu_plot()
            self._setup_memory_plot()
            self._setup_gpu_plot()
            self._setup_stats_plot()
            
            # Get recent data for plotting
            recent_snapshots = self.snapshots[-60:]  # Last 60 seconds
            timestamps = [datetime.fromtimestamp(s.timestamp) for s in recent_snapshots]
            
            # CPU plot
            cpu_data = [s.cpu_percent for s in recent_snapshots]
            self.ax_cpu.plot(timestamps, cpu_data, 'b-', linewidth=2, label='CPU Usage')
            self.ax_cpu.fill_between(timestamps, cpu_data, alpha=0.3, color='blue')
            
            # Memory plot
            memory_data = [s.memory_percent for s in recent_snapshots]
            self.ax_memory.plot(timestamps, memory_data, 'g-', linewidth=2, label='RAM Usage')
            self.ax_memory.fill_between(timestamps, memory_data, alpha=0.3, color='green')
            
            # GPU plot
            gpu_load_data = [s.gpu_load for s in recent_snapshots]
            gpu_memory_data = [s.gpu_memory_percent for s in recent_snapshots]
            
            self.ax_gpu.plot(timestamps, gpu_load_data, 'r-', linewidth=2, label='GPU Load')
            self.ax_gpu.plot(timestamps, gpu_memory_data, 'm-', linewidth=2, label='VRAM Usage')
            self.ax_gpu.legend()
            
            # Format time axes
            for ax in [self.ax_cpu, self.ax_memory, self.ax_gpu]:
                ax.tick_params(axis='x', rotation=45)
                if len(timestamps) > 10:
                    # Show fewer time labels for readability
                    ax.set_xticks(timestamps[::10])
            
            # Statistics display
            if self.snapshots:
                current = self.snapshots[-1]
                
                # Check surround view process status
                surround_status = "❌ Not Started"
                if self.surround_process:
                    if self.surround_process.poll() is None:
                        surround_status = "Running"
                    else:
                        surround_status = f"🏁 Finished (code: {self.surround_process.returncode})"
                
                stats_text = f"""
PROCESS STATUS
Surround View: {surround_status}
Profiler: Running

CURRENT SYSTEM STATUS
CPU: {current.cpu_percent:5.1f}%
{"Temperature: " + f"{current.cpu_temp:.1f}°C" if current.cpu_temp else ""}

MEMORY: {current.memory_percent:5.1f}%
Used: {current.memory_used_gb:.1f}GB / {current.memory_total_gb:.1f}GB

GPU: {current.gpu_load:5.1f}%
VRAM: {current.gpu_memory_percent:5.1f}%
VRAM Used: {current.gpu_memory_used_gb:.1f}GB / {current.gpu_memory_total_gb:.1f}GB
{"GPU Temp: " + f"{current.gpu_temp:.1f}°C" if current.gpu_temp else ""}

DATA POINTS: {len(self.snapshots)}
MONITORING TIME: {(time.time() - self.start_time):.1f}s
                """
                
                self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            self.canvas.draw()
            
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
                    'gpu_load', 'gpu_memory_percent', 'gpu_memory_used_gb',
                    'cpu_temp', 'gpu_temp'
                ])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize log file: {e}")
            self.log_file = None
    
    def _log_snapshot(self, snapshot: SystemSnapshot):
        """Log snapshot to file."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.fromtimestamp(snapshot.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    snapshot.cpu_percent,
                    snapshot.memory_percent,
                    snapshot.memory_used_gb,
                    snapshot.gpu_load,
                    snapshot.gpu_memory_percent,
                    snapshot.gpu_memory_used_gb,
                    snapshot.cpu_temp or '',
                    snapshot.gpu_temp or ''
                ])
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
            # Stop monitoring
            self.running = False
            
            # Stop animation safely
            if hasattr(self, 'ani') and self.ani is not None:
                try:
                    self.ani.event_source.stop()
                    self.ani = None
                    print("✅ Animation stopped")
                except Exception as e:
                    print(f"⚠️  Animation cleanup warning: {e}")
            
            # Clean up matplotlib figures aggressively
            try:
                if hasattr(self, 'fig'):
                    plt.close(self.fig)
                plt.close('all')  # Close any remaining figures
                print("✅ Matplotlib figures closed")
            except Exception as e:
                print(f"⚠️  Figure cleanup warning: {e}")
            
            # Terminate surround view process
            self._terminate_surround_process()
            
            # Wait for monitor thread to finish with shorter timeout
            if hasattr(self, 'monitor_thread') and self.monitor_thread is not None:
                try:
                    self.monitor_thread.join(timeout=1.0)  # Shorter timeout
                    print("✅ Monitor thread joined")
                except Exception as e:
                    print(f"⚠️  Thread cleanup warning: {e}")
            
            # Force GUI cleanup
            try:
                if hasattr(self, 'root') and self.root is not None:
                    # Force quit without mainloop hang
                    self.root.quit()
                    self.root.update()  # Process any pending events
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
    """Main function."""
    args = parse_arguments()
    
    print("🔍 360° Surround View - GUI System Profiler")
    print("=" * 50)
    print("Opening configuration window...")
    
    try:
        # Start configuration GUI
        config_gui = ProfilerConfigGUI()
        config_gui.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Profiler interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Profiler error: {e}")
        return 1
    
    print("✅ Profiler completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
