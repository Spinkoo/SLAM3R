"""
Main script for running SLAM3R reconstruction with pyoctomap integration.
This script runs SLAM3R inference and incrementally builds an octree representation.
"""

import argparse
import os
import glob
import math
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import cv2

from slam3r.pipeline.recon_offline_pipeline import scene_recon_pipeline_offline
from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import transform_img, unsqueeze_view

from pyoctomap_integration.octree_utils import (
    create_color_octree,
    insert_pointcloud_with_color,
    save_octree,
    get_octree_stats,
    extract_camera_pose_from_view,
    HAS_PYOCTOMAP
)


def find_checkpoint(checkpoint_name, default_path=None):
    """Find checkpoint file, checking local checkpoints/ directory first."""
    import os
    # Check if explicitly provided path exists
    if default_path and os.path.exists(default_path):
        return default_path
    
    # Check common locations
    check_dirs = ['checkpoints', 'checkpoints/i2p', 'checkpoints/l2w']
    for check_dir in check_dirs:
        if os.path.exists(check_dir):
            # Look for files matching the name pattern
            for file in os.listdir(check_dir):
                if checkpoint_name in file.lower() and file.endswith('.pth'):
                    return os.path.join(check_dir, file)
    return None


def load_model(model_name, weights, device='cuda'):
    """Load model from checkpoint or HuggingFace."""
    import os
    import torch
    
    print(f'Loading model: {model_name}')
    model = eval(model_name)
    model.to(device)
    
    if weights:
        print(f'Loading pretrained: {weights}')
        ckpt = torch.load(weights, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt
    else:
        # Try to find local checkpoint first
        checkpoint_path = None
        if 'Image2Points' in model_name:
            checkpoint_path = find_checkpoint('slam3r_i2p') or find_checkpoint('i2p')
            if checkpoint_path:
                print(f'Found local I2P checkpoint: {checkpoint_path}')
                ckpt = torch.load(checkpoint_path, map_location=device)
                print(model.load_state_dict(ckpt['model'], strict=False))
                del ckpt
            else:
                print('Loading I2P from HuggingFace (siyan824/slam3r_i2p)...')
                model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        elif 'Local2World' in model_name:
            checkpoint_path = find_checkpoint('slam3r_l2w') or find_checkpoint('l2w')
            if checkpoint_path:
                print(f'Found local L2W checkpoint: {checkpoint_path}')
                ckpt = torch.load(checkpoint_path, map_location=device)
                print(model.load_state_dict(ckpt['model'], strict=False))
                del ckpt
            else:
                print('Loading L2W from HuggingFace (siyan824/slam3r_l2w)...')
                model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
        model.to(device)
    
    return model


def build_octree_from_reconstruction(
    input_views: list,
    per_frame_res: dict,
    rgb_imgs: list,
    conf_thres: float = 3.0,
    octree_resolution: float = 0.05,
    insert_frequency: int = 1,
    max_range: float = -1.0,
    use_camera_poses: bool = True
):
    """
    Build octree incrementally from SLAM3R reconstruction results.
    
    Args:
        input_views: List of view dictionaries with registered point clouds
        per_frame_res: Dictionary with per-frame results from SLAM3R
        rgb_imgs: List of RGB images for coloring
        conf_thres: Confidence threshold for filtering points
        octree_resolution: Octree voxel resolution in meters
        insert_frequency: Insert every Nth frame (1 = all frames)
        max_range: Maximum range for ray casting (-1 for unlimited)
        use_camera_poses: Whether to use camera poses as sensor origins
    
    Returns:
        ColorOcTree instance
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required. Install with: pip install pyoctomap")
    
    print(f"\n>> Building octree with resolution {octree_resolution:.3f} m")
    tree = create_color_octree(resolution=octree_resolution)
    
    num_frames = len(input_views)
    num_inserted = 0
    
    for frame_id in tqdm(range(num_frames), desc="Inserting into octree"):
        if frame_id % insert_frequency != 0:
            continue
        
        # Get registered point cloud
        if 'pts3d_world' not in input_views[frame_id]:
            continue
        
        pts3d_world = input_views[frame_id]['pts3d_world']
        if isinstance(pts3d_world, torch.Tensor):
            pts3d_world = to_numpy(pts3d_world)
        
        # Handle different tensor shapes and preserve spatial correspondence
        original_shape = None
        if pts3d_world.ndim == 4:  # (B, H, W, 3)
            pts3d_world = pts3d_world[0]
        if pts3d_world.ndim == 3:  # (H, W, 3)
            H, W, _ = pts3d_world.shape
            original_shape = (H, W)
            pts3d_world = pts3d_world.reshape(-1, 3)
        else:
            continue
        
        # Get confidence map
        if 'l2w_confs' in per_frame_res and per_frame_res['l2w_confs'][frame_id] is not None:
            conf = per_frame_res['l2w_confs'][frame_id]
            if isinstance(conf, torch.Tensor):
                conf = conf.cpu().numpy()
            if conf.ndim == 2:  # (H, W)
                conf = conf.reshape(-1)
            elif conf.ndim == 3:  # (1, H, W)
                conf = conf[0].reshape(-1)
            
            # Filter by confidence
            valid_mask = conf > conf_thres
        else:
            # No confidence filtering
            valid_mask = np.ones(len(pts3d_world), dtype=bool)
        
        # Get colors from RGB images BEFORE applying mask - ensure pixel-perfect correspondence
        if frame_id < len(rgb_imgs):
            rgb = rgb_imgs[frame_id]
            if isinstance(rgb, torch.Tensor):
                rgb = to_numpy(rgb)
            
            # Ensure RGB is in correct format (H, W, 3) matching point cloud dimensions
            if rgb.ndim == 3:
                # Check if dimensions match point cloud
                if original_shape and rgb.shape[:2] != original_shape:
                    # Resize RGB to match point cloud dimensions for pixel-perfect correspondence
                    rgb = cv2.resize(rgb, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Reshape to match point cloud layout (row-major order, same as pts3d_world)
                rgb = rgb.reshape(-1, 3)
            elif rgb.ndim == 4:  # (B, H, W, 3)
                rgb = rgb[0]
                if original_shape and rgb.shape[:2] != original_shape:
                    rgb = cv2.resize(rgb, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                rgb = rgb.reshape(-1, 3)
            else:
                # If already flattened, check size
                if len(rgb) != original_shape[0] * original_shape[1] if original_shape else len(pts3d_world):
                    print(f"   Warning: RGB size {len(rgb)} doesn't match point cloud size, resizing...")
                    if original_shape:
                        # Reshape to 2D, resize, then flatten
                        rgb_2d = rgb.reshape(-1, 3) if rgb.ndim == 1 else rgb
                        if rgb_2d.shape[0] != original_shape[0] * original_shape[1]:
                            # Need to reshape to 2D first
                            h = int(math.sqrt(len(rgb_2d)))
                            w = len(rgb_2d) // h
                            rgb_2d = rgb_2d.reshape(h, w, 3)
                            rgb_2d = cv2.resize(rgb_2d, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                            rgb = rgb_2d.reshape(-1, 3)
                        else:
                            rgb = rgb_2d
            
            # Ensure RGB is in [0, 255] range (uint8) for proper texture mapping
            if rgb.max() <= 1.0:
                # Colors are normalized [0, 1], convert to [0, 255]
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # Clip to valid range and ensure uint8
                rgb = rgb.clip(0, 255).astype(np.uint8)
            
            # Apply the same confidence mask to colors as points (pixel-perfect correspondence)
            if len(valid_mask) == len(rgb):
                rgb = rgb[valid_mask]
            else:
                print(f"   Warning: RGB size {len(rgb)} doesn't match mask size {len(valid_mask)}")
                # Use default colors if size mismatch
                rgb = np.ones((len(pts3d_world[valid_mask]), 3), dtype=np.uint8) * 128
        else:
            # Default color if no image available
            rgb = np.ones((len(pts3d_world[valid_mask]) if valid_mask is not None else len(pts3d_world), 3), dtype=np.uint8) * 128
        
        # Apply mask to points AFTER getting colors
        pts3d_world = pts3d_world[valid_mask]
        
        if len(pts3d_world) == 0:
            continue
        
        # Final check: ensure colors match points
        if len(rgb) != len(pts3d_world):
            print(f"   Warning: Final RGB size {len(rgb)} doesn't match points {len(pts3d_world)}, adjusting...")
            if len(rgb) > len(pts3d_world):
                rgb = rgb[:len(pts3d_world)]
            else:
                # Pad with last color
                padding = np.tile(rgb[-1:] if len(rgb) > 0 else [[128, 128, 128]], (len(pts3d_world) - len(rgb), 1))
                rgb = np.vstack([rgb, padding]) if len(rgb) > 0 else padding
        
        # Get sensor origin (camera position)
        sensor_origin = None
        if use_camera_poses:
            sensor_origin = extract_camera_pose_from_view(input_views[frame_id])
        
        # Insert into octree
        n_inserted = insert_pointcloud_with_color(
            tree,
            pts3d_world,
            rgb,
            sensor_origin=sensor_origin,
            max_range=max_range
        )
        
        num_inserted += n_inserted
    
    print(f"\n>> Inserted {num_inserted} points into octree")
    stats = get_octree_stats(tree)
    print(f">> Octree stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return tree


def main():
    parser = argparse.ArgumentParser(
        description="Run SLAM3R reconstruction with pyoctomap integration"
    )
    
    # Model arguments
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument('--i2p_model', type=str, default=None, help='I2P model string')
    parser.add_argument("--l2w_model", type=str, default=None, help='L2W model string')
    parser.add_argument('--i2p_weights', type=str, default=None, help='path to I2P weights')
    parser.add_argument("--l2w_weights", type=str, default=None, help="path to L2W weights")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", type=str, help="dataset path or identifier")
    input_group.add_argument("--img_dir", type=str, help="directory of input images")
    parser.add_argument("--video_path", type=str, help="path to input video file")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str, default="results", help="directory to save results")
    parser.add_argument("--test_name", type=str, required=True, help="name of the test")
    
    # SLAM3R reconstruction arguments
    parser.add_argument("--keyframe_stride", type=int, default=3, help="keyframe stride")
    parser.add_argument("--initial_winsize", type=int, default=5, help="initial window size")
    parser.add_argument("--win_r", type=int, default=3, help="window radius for I2P")
    parser.add_argument("--conf_thres_i2p", type=float, default=1.5, help="I2P confidence threshold")
    parser.add_argument("--conf_thres_l2w", type=float, default=12.0, help="L2W confidence threshold")
    parser.add_argument("--num_scene_frame", type=int, default=10, help="number of scene frames")
    parser.add_argument("--num_points_save", type=int, default=2000000, help="points to save in PLY")
    parser.add_argument("--norm_input", action="store_true", help="normalize input for L2W")
    
    # Octree arguments
    parser.add_argument("--octree_resolution", type=float, default=0.05, 
                       help="octree voxel resolution in meters (default: 0.05 = 5cm)")
    parser.add_argument("--octree_insert_freq", type=int, default=1,
                       help="insert every Nth frame into octree (default: 1 = all frames)")
    parser.add_argument("--octree_max_range", type=float, default=-1.0,
                       help="maximum range for ray casting (-1 for unlimited)")
    parser.add_argument("--octree_conf_thres", type=float, default=3.0,
                       help="confidence threshold for octree insertion")
    parser.add_argument("--use_camera_poses", action="store_true", default=True,
                       help="use camera poses as sensor origins")
    parser.add_argument("--save_octree_binary", action="store_true", default=True,
                       help="save octree in binary format (.bt)")
    parser.add_argument("--save_octree_text", action="store_true", default=False,
                       help="save octree in text format (.ot)")
    
    args = parser.parse_args()
    
    if not HAS_PYOCTOMAP:
        print("ERROR: pyoctomap is not installed.")
        print("Please install it with: pip install pyoctomap")
        return
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load models
    print("\n>> Loading models...")
    if args.i2p_weights:
        i2p_model = load_model(args.i2p_model, args.i2p_weights, device)
    else:
        # Try to find local checkpoint first
        i2p_checkpoint = find_checkpoint('slam3r_i2p') or find_checkpoint('i2p')
        if i2p_checkpoint:
            print(f"   Found local I2P checkpoint: {i2p_checkpoint}")
            i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
            import torch
            ckpt = torch.load(i2p_checkpoint, map_location=device)
            i2p_model.load_state_dict(ckpt['model'], strict=False)
            i2p_model.to(device)
        else:
            print("   Loading I2P from HuggingFace (siyan824/slam3r_i2p)...")
            i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
            i2p_model.to(device)
    
    if args.l2w_weights:
        l2w_model = load_model(args.l2w_model, args.l2w_weights, device)
    else:
        # Try to find local checkpoint first
        l2w_checkpoint = find_checkpoint('slam3r_l2w') or find_checkpoint('l2w')
        if l2w_checkpoint:
            print(f"   Found local L2W checkpoint: {l2w_checkpoint}")
            l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
            import torch
            ckpt = torch.load(l2w_checkpoint, map_location=device)
            l2w_model.load_state_dict(ckpt['model'], strict=False)
            l2w_model.to(device)
        else:
            print("   Loading L2W from HuggingFace (siyan824/slam3r_l2w)...")
            l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
            l2w_model.to(device)
    
    i2p_model.eval()
    l2w_model.eval()
    
    # Prepare dataset
    print("\n>> Preparing dataset...")
    
    def find_image_directory(path):
        """Find the directory containing image files (frame*.jpg or numbered images)."""
        import glob
        
        if not os.path.isdir(path):
            return None
        
        # Check if current directory has images
        jpg_files = glob.glob(os.path.join(path, '*.jpg'))
        png_files = glob.glob(os.path.join(path, '*.png'))
        if jpg_files or png_files:
            # Check if they're numbered frames (frame*.jpg or similar)
            frame_files = glob.glob(os.path.join(path, 'frame*.jpg')) + glob.glob(os.path.join(path, 'frame*.png'))
            if frame_files:
                return path
        
        # Check subdirectories for frame*.jpg pattern
        for item in os.listdir(path):
            subdir = os.path.join(path, item)
            if os.path.isdir(subdir):
                frame_files = glob.glob(os.path.join(subdir, 'frame*.jpg')) + glob.glob(os.path.join(subdir, 'frame*.png'))
                if frame_files:
                    print(f"   Found images in subdirectory: {subdir}")
                    return subdir
        
        return None
    
    if args.dataset:
        # Try to find the actual image directory
        img_dir = find_image_directory(args.dataset)
        
        if img_dir is None:
            # Fall back to using the path directly (might be a single image directory)
            img_dir = args.dataset
        
        print(f"   Loading images from: {img_dir}")
        dataset = Seq_Data(
            img_dir=img_dir,
            img_size=224,
            silent=False,
            sample_freq=1,
            start_idx=0,
            num_views=-1,
            start_freq=1,
            to_tensor=True
        )
        
        # Set scene_names for pipeline compatibility
        if not hasattr(dataset, 'scene_names') or not dataset.scene_names:
            scene_name = os.path.basename(img_dir.rstrip('/'))
            dataset.scene_names = [scene_name]
            
    elif args.img_dir:
        dataset = Seq_Data(img_dir=args.img_dir, img_size=224, to_tensor=True)
    else:
        raise ValueError("Must provide either --dataset or --img_dir")
    
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.test_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Run SLAM3R reconstruction
    print("\n>> Running SLAM3R reconstruction...")
    scene_recon_pipeline_offline(
        i2p_model, l2w_model, dataset, args, save_dir=save_dir
    )
    
    # Load reconstruction results
    print("\n>> Loading reconstruction results...")
    preds_dir = os.path.join(save_dir, 'preds')
    
    if os.path.exists(preds_dir):
        # Load from saved predictions
        registered_pcds = np.load(os.path.join(preds_dir, 'registered_pcds.npy'))
        registered_confs = np.load(os.path.join(preds_dir, 'registered_confs.npy'))
        rgb_imgs = np.load(os.path.join(preds_dir, 'input_imgs.npy'))
        
        # Reconstruct input_views structure
        input_views = []
        for i in range(len(registered_pcds)):
            input_views.append({
                'pts3d_world': torch.from_numpy(registered_pcds[i:i+1])
            })
        
        per_frame_res = {
            'l2w_confs': [torch.from_numpy(conf) for conf in registered_confs]
        }
    else:
        # Need to re-run or extract from pipeline
        print("Warning: Prediction directory not found. Re-running with save_preds=True")
        args.save_preds = True
        scene_recon_pipeline_offline(
            i2p_model, l2w_model, dataset, args, save_dir=save_dir
        )
        # Reload
        registered_pcds = np.load(os.path.join(preds_dir, 'registered_pcds.npy'))
        registered_confs = np.load(os.path.join(preds_dir, 'registered_confs.npy'))
        rgb_imgs = np.load(os.path.join(preds_dir, 'input_imgs.npy'))
        
        input_views = []
        for i in range(len(registered_pcds)):
            input_views.append({
                'pts3d_world': torch.from_numpy(registered_pcds[i:i+1])
            })
        
        per_frame_res = {
            'l2w_confs': [torch.from_numpy(conf) for conf in registered_confs]
        }
    
    # Build octree
    print("\n>> Building octree...")
    tree = build_octree_from_reconstruction(
        input_views,
        per_frame_res,
        rgb_imgs,
        conf_thres=args.octree_conf_thres,
        octree_resolution=args.octree_resolution,
        insert_frequency=args.octree_insert_freq,
        max_range=args.octree_max_range,
        use_camera_poses=args.use_camera_poses
    )
    
    # Save octree
    print("\n>> Saving octree...")
    octree_dir = os.path.join(save_dir, 'octree')
    os.makedirs(octree_dir, exist_ok=True)
    
    if args.save_octree_binary:
        octree_path = os.path.join(octree_dir, f"{args.test_name}.bt")
        save_octree(tree, octree_path, binary=True)
    
    if args.save_octree_text:
        octree_path = os.path.join(octree_dir, f"{args.test_name}.ot")
        save_octree(tree, octree_path, binary=False)
    
    print(f"\n>> Done! Octree saved to {octree_dir}")


if __name__ == "__main__":
    main()
