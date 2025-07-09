# 360° Surround View System

This project provides a 360-degree surround view system that stitches together images from multiple fisheye cameras to create a top-down, bird's-eye view.

## Getting Started

Follow these steps to install the necessary dependencies and run the project.

### 1. Installation

This project uses Python. It is recommended to use a virtual environment.

First, clone the repository and navigate into the project directory.

Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

For GPU-accelerated processing (optional), install the dependencies from `requirements_gpu.txt`:

```bash
pip install -r requirements_gpu.txt
```

### 2. Usage

**Quick Run (with pre-calibrated settings)**

If you want to run the project immediately using the existing pre-calibrated camera settings, you can skip the calibration step and directly generate the surround view image by running:

```bash
python generate_blend_weights.py --duration 15
```

This will use the projection matrices already present in the `yaml/` directory to create the final stitched image. The result will be saved as `surround_view_result.jpg`.

**Full Calibration and Generation**

For custom calibration, the system requires a two-step process:

**Step 1: Create Projection Maps (Calibration)**

This is the most critical step. It generates the bird's-eye view transformation for each camera. You must run this command for each of the four cameras (`front`, `back`, `left`, `right`).

**Command Examples:**
```bash
# Calibrate the front camera with default settings
python create_projection_maps.py --camera front

# Calibrate the back camera, adjusting image scale and position for better visibility
python create_projection_maps.py --camera back --scale 0.8 0.8 --shift -100 -50
```

**How to Select Anchor Points:**

The interactive tool will show you the undistorted camera view. For an accurate projection, you need to select four points on the ground that form a rectangle in the real world.

*   **Best Practice:** Use a calibration mat with a grid pattern on the ground. Click the four corner points of a known rectangle on this mat.
*   **Order Matters:** Click the points in order: top-left, top-right, bottom-left, bottom-right relative to the rectangle on the ground.
*   **Tip:** If you don't have a mat, use any clearly visible rectangular object (like a large piece of cardboard) or markings on the ground. Precision is key to a good result.

**Interactive Calibration Instructions:**
1.  A window will show the undistorted image from the camera.
2.  Click on the four anchor points on the ground as described above.
3.  To remove the last point, right-click or press the 'd' key.
4.  Press `Enter` to confirm the four points.
5.  A preview of the projected bird's-eye view will be displayed. Verify that it looks flat and rectangular.
6.  Press `Enter` again to save the projection map to the corresponding `yaml/[camera_name].yaml` file. Press `q` to quit without saving.

Repeat this process for all four cameras.

**Step 2: Run the Real-Time Surround View Stream**

Once all projection maps are created (or if you are using the pre-calibrated ones), run the following command to start the real-time surround view stream. The script will stitch the camera feeds together and display them in a live window.

**Command-line Parameters:**

*   `--fps [number]`: Sets the target frames per second for the stream. Default is `10`.
*   `--duration [seconds]`: Specifies how long the stream should run. Default is `30` seconds.
*   `--buffer-size [number]`: The size of the frame buffer for smoother streaming. Default is `10`.

**Command Examples:**

```bash
# Run the stream for 60 seconds at 20 FPS
python generate_blend_weights.py --duration 60 --fps 20

# Run with a larger buffer for potentially smoother video
python generate_blend_weights.py --buffer-size 20
```

This script will:
1.  Load the camera calibrations and projection maps.
2.  Continuously process and warp the image sequences from each camera.
3.  Calculate blending weights for the overlapping regions to create a seamless image.
4.  Apply color correction in real-time.
5.  Display the live stitched view in a window.

## Project Structure

```
/
├── images/                 # Sample camera images
├── yaml/                   # Camera calibration and projection data
├── surround_view/          # Core source code
├── create_projection_maps.py # Step 1: Projection calibration tool
├── generate_blend_weights.py # Step 2: Blending and stitching tool
├── requirements.txt        # Main dependencies
├── requirements_gpu.txt    # Optional GPU dependencies
└── README.md
```

## Configuration

The system is configured through YAML files in the `yaml/` directory and presets in `presets/calibration_presets.json`. These files define camera properties, projection targets, and other system parameters.

## Output Files

| File                      | Description                               |
| ------------------------- | ----------------------------------------- |
| `weights.png`             | 4-channel blending weight matrix.         |
| `masks.png`               | 4-channel valid region masks.             |
| `surround_view_result.jpg`| Final stitched 360° view.                 |
| `[camera].yaml`           | Updated calibration files with projection matrices. |
