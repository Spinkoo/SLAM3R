# Pyoctomap Integration for SLAM3R

This module provides integration between SLAM3R dense 3D reconstruction and [pyoctomap](https://github.com/OctoMap/pyoctomap) for efficient octree-based 3D representation. This integration allows you to build incremental octree maps from SLAM3R's point cloud reconstructions, which is useful for robotics applications, path planning, and efficient 3D storage.

## Features

- **Incremental Octree Building**: Build octree maps as SLAM3R processes frames
- **Color Support**: Preserve RGB colors from SLAM3R reconstruction
- **Efficient Storage**: Compressed .bt (binary) or .ot (text) octree formats
- **ROS Compatible**: Export octrees compatible with ROS octomap packages
- **Memory Efficient**: Process large scenes without loading all points into memory

## Installation

### 1. Install pyoctomap

```bash
pip install pyoctomap
```

For more installation options and dependencies, see the [pyoctomap repository](https://github.com/OctoMap/pyoctomap).

### 2. Install SLAM3R Dependencies

Make sure you have all SLAM3R dependencies installed (see main README.md).

### 3. Download Model Checkpoints (Optional)

The scripts will automatically download SLAM3R models from HuggingFace if not found locally. However, you can also download checkpoints manually:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SLAM3R I2P model (optional - will auto-download if not found)
# Place in checkpoints/ directory

# Download SLAM3R L2W model (optional - will auto-download if not found)
# Place in checkpoints/ directory

# Download DUSt3R pretrained weights (for training, not needed for inference)
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

The scripts will automatically check the `checkpoints/` directory for model weights before downloading from HuggingFace.

## Quick Start

### Example 1: Replica Dataset

```bash
bash pyoctomap_integration/example_replica.sh
```

Or run directly:

```bash
python pyoctomap_integration/recon_with_octomap_incremental.py \
    --test_name replica_octomap \
    --dataset data/Replica_demo/room0 \
    --octree_resolution 0.05 \
    --save_octree_binary
```

### Example 2: Custom Image Directory

```bash
python pyoctomap_integration/recon_with_octomap_incremental.py \
    --test_name my_scene \
    --img_dir path/to/images \
    --octree_resolution 0.05 \
    --octree_conf_thres 3.0 \
    --save_octree_binary
```

## Usage

### Basic Usage

The main script `recon_with_octomap_incremental.py` runs SLAM3R reconstruction and builds an octree incrementally:

```bash
python pyoctomap_integration/recon_with_octomap_incremental.py \
    --test_name <name> \
    --dataset <path> \  # or --img_dir <path>
    --octree_resolution 0.05 \
    --save_octree_binary
```

### Key Parameters

#### SLAM3R Parameters
- `--keyframe_stride`: Sampling frequency for keyframes (default: 3)
- `--win_r`: Window radius for I2P model (default: 3)
- `--conf_thres_i2p`: I2P confidence threshold (default: 1.5)
- `--conf_thres_l2w`: L2W confidence threshold (default: 12.0)
- `--num_scene_frame`: Number of reference frames (default: 10)

#### Octree Parameters
- `--octree_resolution`: Voxel resolution in meters (default: 0.05 = 5cm)
  - Smaller values = higher detail but more memory
  - Larger values = lower detail but less memory
- `--octree_insert_freq`: Insert every Nth frame (default: 1 = all frames)
- `--octree_conf_thres`: Confidence threshold for insertion (default: 3.0)
- `--octree_max_range`: Maximum range for ray casting, -1 for unlimited (default: -1.0)
- `--save_octree_binary`: Save as binary .bt file (recommended)
- `--save_octree_text`: Save as text .ot file (for debugging)

### Output

Results are saved to `results/<test_name>/`:
- `octree/<test_name>_octree.bt`: Binary octree file (recommended)
- `octree/<test_name>_octree.ot`: Text octree file (optional)
- `*_recon.ply`: Final point cloud (from SLAM3R)
- `preds/`: Per-frame predictions (if enabled)

## API Usage

You can also use the octree utilities programmatically:

```python
from pyoctomap_integration.octree_utils import (
    create_color_octree,
    insert_pointcloud_with_color,
    save_octree,
    load_octree
)
import numpy as np

# Create octree
tree = create_color_octree(resolution=0.05)

# Insert point cloud with colors
points = np.random.rand(1000, 3) * 10  # (N, 3) in meters
colors = np.random.randint(0, 255, (1000, 3))  # (N, 3) RGB [0-255]
sensor_origin = np.array([0.0, 0.0, 0.0])  # Camera position

insert_pointcloud_with_color(
    tree, points, colors,
    sensor_origin=sensor_origin,
    max_range=-1.0
)

# Save octree
save_octree(tree, "output.bt", binary=True)

# Load octree
loaded_tree = load_octree("output.bt")
```

## Integration Workflow

The integration follows this workflow:

1. **Run SLAM3R**: Extract 3D point clouds from RGB video frames
2. **Initialize Octree**: Create a ColorOcTree with specified resolution
3. **Incremental Insertion**: For each registered frame:
   - Extract point cloud in world coordinates
   - Filter by confidence threshold
   - Insert into octree with colors and sensor origin
4. **Export**: Save octree as .bt (binary) or .ot (text) file

### Example Integration Pattern

```python
import pyoctomap
from slam3r.models import Image2PointsModel, Local2WorldModel

# 1. Run SLAM3R reconstruction (get registered point clouds)
# ... SLAM3R inference code ...

# 2. Initialize octree
tree = pyoctomap.ColorOcTree(resolution=0.05)  # 5cm voxels

# 3. For each frame, insert point cloud
for frame_id in range(num_frames):
    pts3d_world = registered_pointclouds[frame_id]  # (H, W, 3)
    colors = rgb_images[frame_id]  # (H, W, 3)
    camera_pose = camera_poses[frame_id]  # (4, 4)
    
    # Reshape to (N, 3)
    points = pts3d_world.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    
    # Get sensor origin (camera position)
    sensor_origin = camera_pose[:3, 3]
    
    # Insert into octree
    tree.insertPointCloudWithColor(
        points, colors,
        sensor_origin=sensor_origin
    )

# 4. Save octree
tree.writeBinary("reconstruction.bt")
```

## Visualization

### Using ROS

The generated .bt files can be visualized in ROS:

```bash
rosrun octomap_server octomap_saver -f map.bt
rosrun octomap_server octomap_server_node map.bt
```

Then visualize in RViz with the `Octomap` display type.

### Using Python

You can also visualize using pyoctomap's built-in functions or convert to other formats:

```python
from pyoctomap_integration.octree_utils import load_octree, get_octree_stats

tree = load_octree("reconstruction.bt")
stats = get_octree_stats(tree)
print(f"Octree resolution: {stats['resolution']} m")
print(f"Number of nodes: {stats['size']}")
print(f"Memory usage: {stats['memory_usage']} bytes")
```

## Tips

1. **Resolution Selection**: 
   - Indoor scenes: 0.05m (5cm) is usually good
   - Outdoor scenes: 0.1-0.2m (10-20cm) may be better
   - Adjust based on scene scale and desired detail

2. **Confidence Thresholding**:
   - Higher thresholds = fewer but more reliable points
   - Lower thresholds = more points but may include noise
   - Start with default (3.0) and adjust based on results

3. **Memory Management**:
   - Use `--octree_insert_freq` to skip frames if memory is limited
   - Binary format (.bt) is more efficient than text (.ot)

4. **Camera Poses**:
   - If camera poses are available in the dataset, they're used automatically
   - Otherwise, sensor origin is estimated from point cloud structure

## Troubleshooting

### Import Error: pyoctomap not found
```bash
pip install pyoctomap
```

### CUDA Version Mismatch Error
If you see "CUDA error: no kernel image is available for execution on the device":

**Problem**: PyTorch was compiled for a different CUDA version than your system.

**Quick Fix** (Recommended):
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install PyTorch with CUDA 12.1 (compatible with CUDA 13.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or use the helper script
python pyoctomap_integration/fix_cuda.py
```

**Alternative Solutions**:
1. **Use CPU mode** (works but slower):
   ```bash
   python pyoctomap_integration/recon_with_octomap_incremental.py --device cpu ...
   ```

2. **Check your CUDA version**:
   ```bash
   nvcc --version  # System CUDA version
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
   ```

**Note**: 
- CUDA 13.0+ is very new - PyTorch CUDA 12.1 builds are backward compatible
- After reinstalling PyTorch, restart your Python session
- Verify CUDA works: `python -c "import torch; print(torch.cuda.is_available()); x=torch.zeros(1).cuda(); print('CUDA OK!')"`

### Out of Memory
- Increase `--octree_resolution` (larger voxels)
- Increase `--octree_insert_freq` (skip frames)
- Reduce `--num_points_save`
- Use `--device cpu` if GPU memory is limited

### Poor Quality Octree
- Lower `--octree_resolution` for finer detail
- Adjust `--octree_conf_thres` to filter noise
- Check SLAM3R reconstruction quality first

## Citation

If you use this integration, please cite both SLAM3R and OctoMap:

```bibtex
@article{slam3r,
  title={SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos},
  author={Liu, Yuzheng and Dong, Siyan and Wang, Shuzhe and Yin, Yingda and Yang, Yanchao and Fan, Qingnan and Chen, Baoquan},
  journal={arXiv preprint arXiv:2412.09401},
  year={2024}
}

@article{hornung2013octomap,
  title={OctoMap: An efficient probabilistic 3D mapping framework based on octrees},
  author={Hornung, Armin and Wurm, Kai M and Bennewitz, Maren and Stachniss, Cyrill and Burgard, Wolfram},
  journal={Autonomous robots},
  volume={34},
  number={3},
  pages={189--206},
  year={2013},
  publisher={Springer}
}
```

## License

This integration follows the same license as SLAM3R (CC BY-NC-SA 4.0).
