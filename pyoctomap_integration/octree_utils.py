"""
Utility functions for pyoctomap integration with SLAM3R.
Handles octree creation, point cloud insertion, and export.
"""

import numpy as np
import torch
from typing import Optional, Tuple
import os

try:
    import pyoctomap
    HAS_PYOCTOMAP = True
except ImportError:
    HAS_PYOCTOMAP = False
    print("Warning: pyoctomap not installed. Please install it to use octree functionality.")
    print("Install with: pip install pyoctomap")


def create_color_octree(resolution: float = 0.05) -> 'pyoctomap.ColorOcTree':
    """
    Create a new color octree with specified resolution.
    
    Args:
        resolution: Voxel resolution in meters (default: 0.05 = 5cm)
    
    Returns:
        ColorOcTree instance
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required. Install with: pip install pyoctomap")
    
    tree = pyoctomap.ColorOcTree(resolution)
    
    # ColorOcTree should have colors enabled by default
    # Some versions don't have isColorEnabled() method, so we skip the check
    # The fact that it's a ColorOcTree means it supports colors
    
    return tree


def estimate_sensor_origin_from_pointcloud(pts3d: np.ndarray) -> np.ndarray:
    """
    Estimate sensor origin from point cloud by finding the centroid of points
    near the origin (assuming camera is at origin in camera coordinates).
    
    Args:
        pts3d: Point cloud in world coordinates (N, 3)
    
    Returns:
        Estimated sensor origin (3,)
    """
    # Use the median of points as a robust estimate
    # In camera coordinates, the camera is typically at origin
    # For world coordinates, we estimate from the point cloud structure
    if len(pts3d) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    # Use the point closest to origin as sensor estimate
    distances = np.linalg.norm(pts3d, axis=1)
    closest_idx = np.argmin(distances)
    return pts3d[closest_idx]


def insert_pointcloud_with_color(
    tree: 'pyoctomap.ColorOcTree',
    points: np.ndarray,
    colors: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    max_range: float = -1.0,
    lazy_eval: bool = False
) -> int:
    """
    Insert a point cloud with colors into the octree.
    
    Args:
        tree: ColorOcTree instance
        points: Point cloud coordinates (N, 3) in world frame
        colors: Point colors (N, 3) in range [0, 255] or [0, 1]
        sensor_origin: Sensor origin (3,) in world frame. If None, estimated from points
        max_range: Maximum range for ray casting (-1 for unlimited)
        lazy_eval: Whether to use lazy evaluation
    
    Returns:
        Number of points inserted
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required")
    
    # Ensure points are numpy array
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # Ensure correct shape
    if points.ndim == 3:
        # Reshape from (H, W, 3) to (N, 3)
        H, W, _ = points.shape
        points = points.reshape(-1, 3)
        if colors.ndim == 3:
            colors = colors.reshape(-1, 3)
    
    # Filter out invalid points (NaN, Inf, or zero)
    valid_mask = np.isfinite(points).all(axis=1)
    valid_mask &= (np.linalg.norm(points, axis=1) > 1e-6)  # Remove points at origin
    
    if not valid_mask.any():
        return 0
    
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Normalize colors to [0, 1] range and convert to float64
    # pyoctomap expects colors as double (float64), not uint8
    if colors.max() > 1.0:
        # Colors are in [0, 255], normalize to [0, 1]
        colors = colors.astype(np.float64) / 255.0
    else:
        # Colors are already in [0, 1]
        colors = colors.astype(np.float64)
    
    # Ensure colors are contiguous and have correct shape
    colors = np.ascontiguousarray(colors)
    if colors.shape != (len(points), 3):
        raise ValueError(f"Colors shape mismatch: expected ({len(points)}, 3), got {colors.shape}")
    
    # Estimate sensor origin if not provided
    if sensor_origin is None:
        sensor_origin = estimate_sensor_origin_from_pointcloud(points)
    
    # Ensure sensor_origin is numpy array
    if isinstance(sensor_origin, torch.Tensor):
        sensor_origin = sensor_origin.cpu().numpy()
    sensor_origin = np.array(sensor_origin, dtype=np.float64).flatten()
    
    if len(sensor_origin) != 3:
        raise ValueError(f"sensor_origin must have shape (3,), got {sensor_origin.shape}")
    
    # Ensure points are contiguous float64
    points = np.ascontiguousarray(points.astype(np.float64))
    
    # ColorOcTree should have colors enabled by default
    # No need to check/enable - ColorOcTree always supports colors
    
    # Insert point cloud
    try:
        # Try different parameter names for pyoctomap API compatibility
        # Some versions use maxrange, others use max_range
        try:
            tree.insertPointCloudWithColor(
                points,
                colors,
                sensor_origin=sensor_origin,
                max_range=max_range if max_range > 0 else -1.0,
                lazy_eval=lazy_eval
            )
        except TypeError:
            # Try with maxrange (no underscore)
            try:
                tree.insertPointCloudWithColor(
                    points,
                    colors,
                    sensor_origin=sensor_origin,
                    maxrange=max_range if max_range > 0 else -1.0,
                    lazy_eval=lazy_eval
                )
            except TypeError:
                # Try without max_range parameter
                tree.insertPointCloudWithColor(
                    points,
                    colors,
                    sensor_origin=sensor_origin,
                    lazy_eval=lazy_eval
                )
        return len(points)
    except Exception as e:
        print(f"Warning: Error inserting point cloud: {e}")
        import traceback
        traceback.print_exc()
        return 0


def save_octree(tree: 'pyoctomap.ColorOcTree', filepath: str, binary: bool = True):
    """
    Save octree to file.
    
    Args:
        tree: ColorOcTree instance
        filepath: Output file path (.bt for binary, .ot for text)
        binary: Whether to save in binary format
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required")
    
    # Verify it's a ColorOcTree
    tree_type = type(tree).__name__
    if 'Color' not in tree_type:
        raise ValueError(f"Tree must be a ColorOcTree, got {tree_type}! Colors cannot be saved.")
    
    # ColorOcTree should have colors enabled by default
    # Some pyoctomap versions don't have isColorEnabled() method
    # We assume ColorOcTree always supports colors
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    if binary:
        if not filepath.endswith('.bt'):
            filepath = filepath.replace('.ot', '.bt')
        # Try writeBinary - if it fails, try alternative methods
        try:
            tree.writeBinary(filepath)
        except Exception as e:
            print(f"   ⚠️  writeBinary() failed: {e}")
            # Try write() as fallback (might work for ColorOcTree)
            try:
                filepath_ot = filepath.replace('.bt', '.ot')
                tree.write(filepath_ot)
                print(f"   Saved as text format instead: {filepath_ot}")
                filepath = filepath_ot
            except Exception as e2:
                raise RuntimeError(f"Failed to save octree: {e2}")
    else:
        if not filepath.endswith('.ot'):
            filepath = filepath.replace('.bt', '.ot')
        tree.write(filepath)
    
    print(f">> Saved octree to {filepath}")
    print(f"   Resolution: {tree.getResolution():.4f} m")
    print(f"   Size: {tree.size()} nodes")
    print(f"   Tree type: {tree_type}")
    
    # Verify the saved file can be loaded as ColorOcTree
    try:
        test_tree = pyoctomap.ColorOcTree(filepath)
        print(f"   ✅ Verified: File can be loaded as ColorOcTree")
        del test_tree
    except Exception as e:
        error_str = str(e)
        print(f"   ⚠️  WARNING: Saved file cannot be loaded as ColorOcTree")
        print(f"   Error: {error_str}")
        # Try to load as simple OcTree to see if it saved at all
        try:
            test_tree = pyoctomap.OcTree(filepath)
            print(f"   ⚠️  File was saved as simple OcTree (no colors)")
            print(f"   This may be a pyoctomap version issue - ColorOcTree.writeBinary() may not work correctly")
            del test_tree
        except:
            pass


def load_octree(filepath: str) -> 'pyoctomap.ColorOcTree':
    """
    Load octree from file.
    
    Args:
        filepath: Path to .bt or .ot file
    
    Returns:
        ColorOcTree instance
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required")
    
    tree = pyoctomap.ColorOcTree(filepath)
    return tree


def get_octree_stats(tree: 'pyoctomap.ColorOcTree') -> dict:
    """
    Get statistics about the octree.
    
    Args:
        tree: ColorOcTree instance
    
    Returns:
        Dictionary with octree statistics
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required")
    
    return {
        'resolution': tree.getResolution(),
        'size': tree.size(),
        'volume': tree.volume(),
        'memory_usage': tree.memoryUsage(),
    }


def extract_camera_pose_from_view(view: dict) -> Optional[np.ndarray]:
    """
    Extract camera pose (sensor origin) from SLAM3R view dictionary.
    
    Args:
        view: View dictionary from SLAM3R
    
    Returns:
        Camera position (3,) in world coordinates, or None if not available
    """
    # Try to get camera pose from view
    if 'camera_pose' in view:
        pose = view['camera_pose']
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        
        # Extract translation (camera position in world)
        if pose.shape == (4, 4):
            return pose[:3, 3]
        elif pose.shape == (4, 3):
            return pose[:3, 2]  # Last column
        elif pose.shape == (3,):
            return pose
    
    # If camera pose not available, estimate from point cloud
    if 'pts3d_world' in view:
        pts3d = view['pts3d_world']
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()
        
        if pts3d.ndim == 4:  # (B, H, W, 3)
            pts3d = pts3d[0]
        if pts3d.ndim == 3:  # (H, W, 3)
            pts3d = pts3d.reshape(-1, 3)
        
        return estimate_sensor_origin_from_pointcloud(pts3d)
    
    return None
