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

The system requires a two-step process to generate the surround view:

**Step 1: Create Projection Maps**

This step generates the bird's-eye view transformation for each camera. You will need to run this command for each of the four cameras (`front`, `back`, `left`, `right`).

An interactive window will open where you need to select four calibration points on the ground.

```bash
python create_projection_maps.py --camera front
```

**Interactive Calibration Instructions:**
1.  A window will show the undistorted image from the camera.
2.  Click on four points on the ground in the image to define the projection area.
3.  If you make a mistake, right-click or press 'd' to remove the last point.
4.  Press `Enter` to confirm the points.
5.  A preview of the bird's-eye view will be shown. Press `Enter` to save the projection map or `q` to quit without saving.

Repeat this process for all four cameras.

**Step 2: Generate Blending Weights and Final Image**

Once all projection maps are created, run the following command to generate the final stitched surround view image.

```bash
python generate_blend_weights.py
```

This script will:
1.  Load the camera calibrations and projection maps.
2.  Process and warp the images from each camera.
3.  Calculate blending weights for the overlapping regions to create a seamless image.
4.  Apply color correction.
5.  Stitch all views together and save the final image as `surround_view_result.jpg`.

You can use the `--no-preview` flag to run the script without displaying intermediate windows.

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
