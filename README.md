# 360° Surround View System v2.0

A **modern, streamlined implementation** of a 360-degree surround view system with advanced image processing capabilities. This system transforms multiple fisheye camera feeds into a seamless bird's eye view for enhanced spatial awareness.

## 🌟 Key Features

- **Modern Python Architecture**: Type-safe code with dataclasses and advanced error handling
- **Interactive GUI**: Enhanced point selection with real-time validation and visual feedback  
- **Advanced Blending**: Distance-based weight calculation for seamless image stitching
- **Color Correction**: Automatic luminance balancing and white balance correction
- **Real-time Preview**: Live preview during calibration and processing
- **Configurable System**: Clean configuration management with validation
- **Quality Metrics**: Automatic assessment of processing results

## 📁 Project Structure

```
360-surround-view/
├── images/                         # Camera test images
│   ├── front.png, back.png
│   ├── left.png, right.png
│   └── car.png                     # Vehicle overlay
├── yaml/                           # Camera calibration data
│   ├── front.yaml, back.yaml
│   ├── left.yaml, right.yaml
│   └── [projection matrices]
├── surround_view/                  # Core system modules
│   ├── __init__.py                # Public API
│   ├── config.py                  # Configuration system
│   ├── camera_model.py            # Fisheye camera processing
│   ├── gui_tools.py               # Interactive GUI components
│   ├── view_processor.py          # Image stitching engine
├── create_projection_maps.py      # 🎯 Projection calibration tool
├── generate_blend_weights.py      # ⚖️ Weight generation tool
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `opencv-python>=4.5.0` - Computer vision operations
- `PyQt5>=5.15.0` - GUI framework and threading
- `numpy>=1.19.0` - Numerical computing
- `Pillow>=8.0.0` - Image processing
- `PyYAML>=5.4.0` - Configuration files

### 2. Create Projection Maps

Generate bird's eye view transformations for each camera:

```bash
# Basic usage
python create_projection_maps.py --camera front

# With image adjustments for better feature visibility
python create_projection_maps.py --camera front --scale 0.7 0.8 --shift -150 -100
python create_projection_maps.py --camera back --scale 0.7 0.8 --shift -150 -100
python create_projection_maps.py --camera left --scale 0.7 0.8 --shift -150 -100
python create_projection_maps.py --camera right --scale 0.7 0.8 --shift -150 -100
```

**📋 Interactive Instructions:**
1. **View the undistorted image** - Camera feed with fisheye correction applied
2. **Click 4 calibration points** - Select feature points on the ground pattern in order
3. **Use right-click or 'D'** to remove incorrectly placed points
4. **Press ENTER** when selection is complete
5. **Review the projection** - Verify the bird's eye view looks correct
6. **Press ENTER** to save or 'Q' to cancel

### 3. Generate Blending Weights

Create seamless stitching weights and final surround view:

```bash
# Full processing with preview
python generate_blend_weights.py

# Fast processing without preview windows
python generate_blend_weights.py --no-preview

# Generate without saving weight files
python generate_blend_weights.py --no-save-weights
```

**🔄 Processing Pipeline:**
1. **Load calibrations** - Validate all camera models and projection matrices
2. **Process images** - Apply undistortion, projection, and camera-specific transforms
3. **Generate weights** - Calculate smooth blending matrices for overlap regions
4. **Apply corrections** - Luminance balancing and white balance adjustment
5. **Create final view** - Stitch all cameras into complete 360° image
6. **Save results** - Export weight matrices and final surround view

## 🏗️ System Architecture

### **Modern Configuration System**
- **Dataclass-based** configuration with type safety
- **Singleton pattern** for consistent system-wide settings
- **Automatic validation** of parameters and file paths
- **Flexible dimensions** - easily adjust view area and car placement

### **Advanced Camera Model** 
- **Robust error handling** with custom exceptions
- **Type-safe** operations with full type hints
- **Method chaining** for fluent API usage
- **Automatic validation** of calibration completeness

### **Enhanced GUI Tools**
- **Context managers** for automatic cleanup
- **Real-time validation** of point selections
- **Visual feedback** with progress indicators
- **Keyboard shortcuts** for efficient operation

### **Sophisticated View Processor**
- **Multiple blending algorithms** (distance-based, multi-band, simple average)
- **Advanced color correction** with luminance balancing
- **Quality metrics** and automatic assessment
- **Configurable processing** parameters

## 🎨 Advanced Features

### **Distance-Based Blending**
Our advanced blending algorithm calculates smooth transition weights based on distance to camera boundaries, eliminating ghosting artifacts and creating seamless transitions between camera views.

### **Automatic Color Correction** 
- **Luminance balancing** across overlapping regions
- **White balance** adjustment for consistent color temperature
- **Channel-wise** correction for optimal color reproduction

### **Interactive Calibration**
- **Real-time visual feedback** during point selection
- **Automatic validation** of calibration point quality
- **Progressive preview** showing results at each step

### **Quality Assessment**
- **Coverage analysis** - percentage of final image with valid data
- **Processing timing** - performance metrics for optimization
- **Blend region counting** - validation of overlap detection

## 🔧 Configuration

The system uses a modern configuration architecture:

```python
from surround_view import SystemConfig

config = SystemConfig()

# Access camera configurations
camera_config = config.get_camera_config("front")

# View dimensions and boundaries
dimensions = config.dimensions
total_w, total_h = dimensions.total_dimensions
car_bounds = dimensions.car_boundaries

# Projection targets for calibration
targets = config.projection_targets.keypoints["front"]
```

### **Customizable Parameters**
- **View extensions**: Adjust how far the view extends beyond calibration area
- **Car boundaries**: Configure vehicle placement and size
- **Projection shapes**: Define output dimensions for each camera
- **Calibration points**: Set target locations for interactive selection

## 📊 Output Files

| File | Description | Usage |
|------|-------------|-------|
| `weights.png` | 4-channel blending weight matrix | Smooth transitions between cameras |
| `masks.png` | 4-channel valid region masks | Define active areas for each camera |
| `surround_view_result.jpg` | Final stitched 360° view | Demonstration of complete system |
| `[camera].yaml` | Updated calibration files | Contains projection matrices |

## 🐛 Troubleshooting

### **Common Issues**

**Calibration file not found**
```
✗ Failed to load camera: Calibration file not found: yaml/front.yaml
```
*Solution*: Ensure YAML calibration files are present in the `yaml/` directory

**Missing projection matrix**
```
✗ front: Camera front missing projection matrix
```
*Solution*: Run `create_projection_maps.py --camera front` to generate projection matrix

**Image processing failed**
```
✗ front: processing failed - Undistortion maps not available
```
*Solution*: Verify camera calibration parameters are complete and valid

### **Performance Tips**

- Use `--no-preview` flag for faster processing in automated workflows
- Adjust `--scale` parameters to optimize feature point visibility
- Ensure test images are high quality for better projection results

## 🔬 Technical Details

### **Coordinate System**
- **Origin**: Top-left corner of final image
- **Units**: Pixels (1 pixel = 1 cm in real world for default configuration)
- **Car placement**: Centered in designated rectangular region

### **Processing Pipeline**
1. **Fisheye correction** using OpenCV's fisheye model
2. **Perspective projection** to bird's eye view
3. **Camera-specific transforms** (rotation/flipping)
4. **Distance-based blending** for overlap regions
5. **Color correction** and white balance
6. **Final composition** with car overlay

### **Blending Algorithm**
Our distance-based blending calculates weights using:
```
weight_A = distance_to_B² / (distance_to_A² + distance_to_B²)
weight_B = distance_to_A² / (distance_to_A² + distance_to_B²)
```

This creates smooth, natural transitions that adapt to the geometry of each overlap region.

## 🤝 Contributing

This is a modern, educational implementation designed for learning and experimentation. Key areas for enhancement:

- **Additional blending algorithms** (multi-band, Laplacian pyramids)
- **Real-time processing** optimizations
- **Camera auto-detection** and calibration
- **Web-based interface** for remote operation
- **Machine learning** integration for automatic feature detection

## 📄 License

MIT License - Feel free to use, modify, and distribute for educational and commercial purposes.

---

**🚗 Drive with confidence using 360° situational awareness!**
