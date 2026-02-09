"""
Incremental octree building during SLAM3R reconstruction.
This version hooks into the reconstruction pipeline to build the octree incrementally
as frames are processed, which is more memory efficient.
"""

import argparse
import os
import glob
import numpy as np
import torch
from tqdm import tqdm

from slam3r.pipeline.recon_offline_pipeline import (
    scene_recon_pipeline_offline,
    get_img_tokens,
    initialize_scene,
    i2p_inference_batch,
    l2w_inference,
    scene_frame_retrieve,
    normalize_views,
    to_device
)
from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Image2PointsModel, Local2WorldModel
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import transform_img, unsqueeze_view, get_multiview_scale

from pyoctomap_integration.octree_utils import (
    create_color_octree,
    insert_pointcloud_with_color,
    save_octree,
    get_octree_stats,
    extract_camera_pose_from_view,
    HAS_PYOCTOMAP
)


def scene_recon_pipeline_with_octree(
    i2p_model, l2w_model, dataset, args, save_dir="results"
):
    """
    Modified SLAM3R reconstruction pipeline that builds octree incrementally.
    Based on scene_recon_pipeline_offline but with octree integration.
    """
    if not HAS_PYOCTOMAP:
        raise ImportError("pyoctomap is required. Install with: pip install pyoctomap")
    
    # Create octree
    print(f"\n>> Creating octree with resolution {args.octree_resolution:.3f} m")
    tree = create_color_octree(resolution=args.octree_resolution)
    
    # Import original pipeline logic
    from slam3r.pipeline.recon_offline_pipeline import (
        adapt_keyframe_stride,
        sel_ids_by_score
    )
    
    win_r = args.win_r
    num_scene_frame = args.num_scene_frame
    initial_winsize = args.initial_winsize
    conf_thres_l2w = args.conf_thres_l2w
    conf_thres_i2p = args.conf_thres_i2p
    num_points_save = args.num_points_save
    
    scene_id = dataset.scene_names[0]
    data_views = dataset[0][:]
    num_views = len(data_views)
    
    # Additional limit check (shouldn't be needed if dataset respected max_frames, but just in case)
    if args.max_frames > 0 and args.max_frames < num_views:
        print(f"\n>> Warning: Dataset loaded {num_views} frames, limiting to {args.max_frames} for debugging")
        data_views = data_views[:args.max_frames]
        num_views = args.max_frames
    elif args.max_frames > 0:
        print(f"\n>> Processed {num_views} frames (limited from full dataset)")
    
    # Pre-save RGB images
    rgb_imgs = []
    for i in range(len(data_views)):
        if data_views[i]['img'].shape[0] == 1:
            data_views[i]['img'] = data_views[i]['img'][0]
        rgb_imgs.append(transform_img(dict(img=data_views[i]['img'][None]))[..., ::-1])
    
    if 'valid_mask' not in data_views[0]:
        valid_masks = None
    else:
        valid_masks = [view['valid_mask'] for view in data_views]
    
    # Preprocess data
    for view in data_views:
        view['img'] = torch.tensor(view['img'][None])
        view['true_shape'] = torch.tensor(view['true_shape'][None])
        for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
            if key in view:
                del view[key]
        to_device(view, device=args.device)
    
    # Pre-extract img tokens
    try:
        res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)
        print('finish pre-extracting img tokens')
    except RuntimeError as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower() or "no kernel image" in error_msg.lower():
            print(f"\n⚠️  CUDA Error during inference: {error_msg}")
            print("\n" + "="*60)
            print("   CUDA FAILED DURING INFERENCE!")
            print("="*60)
            
            # Check GPU compute capability
            gpu_capability = None
            gpu_name = None
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_capability = torch.cuda.get_device_capability(0)
                    print(f"   GPU: {gpu_name}")
                    print(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]} (sm_{gpu_capability[0]}{gpu_capability[1]})")
                    print(f"   PyTorch CUDA version: {torch.version.cuda}")
                    
                    # Check if it's a Blackwell GPU (compute capability 12.0+)
                    if gpu_capability[0] >= 12:
                        print("\n   ⚠️  BLACKWELL GPU DETECTED!")
                        print("   Your GPU (Blackwell, compute capability 12.0) is very new.")
                        print("   PyTorch official builds currently support up to sm_90 (compute capability 9.0).")
                        print("\n   🔧 SOLUTIONS:")
                        print("   1. Use CPU mode (works now, slower):")
                        print("      Add --device cpu to your command")
                        print("\n   2. Try PyTorch nightly build (may have experimental support):")
                        print("      pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121")
                        print("\n   3. Wait for official PyTorch release with Blackwell support")
                        print("      Check: https://pytorch.org/get-started/locally/")
                        print("\n   ⚠️  Falling back to CPU mode...")
                    elif gpu_capability[0] >= 10:
                        print("\n   ⚠️  HOOPER GPU DETECTED!")
                        print("   Your GPU (Hopper, compute capability 10.0+) may need newer PyTorch.")
                        print("   PyTorch CUDA 12.1 supports up to sm_90.")
                        print("\n   🔧 SOLUTIONS:")
                        print("   1. Try PyTorch nightly build:")
                        print("      pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121")
                        print("\n   2. Use CPU mode:")
                        print("      Add --device cpu to your command")
                        print("\n   ⚠️  Falling back to CPU mode...")
                    elif gpu_capability[0] < 7:
                        print("\n   ⚠️  OLDER GPU DETECTED!")
                        print(f"   Your GPU (compute capability {gpu_capability[0]}.{gpu_capability[1]}) may be too old.")
                        print("   PyTorch CUDA 12.1 supports compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper).")
                        print("\n   🔧 SOLUTION:")
                        print("   Use CPU mode or find a PyTorch build for older GPUs")
                        print("   Add --device cpu to your command")
                        print("\n   ⚠️  Falling back to CPU mode...")
                    else:
                        print("\n   This usually means PyTorch doesn't have kernels for your GPU.")
                        print("   Possible causes:")
                        print("   1. GPU compute capability not fully supported")
                        print("   2. PyTorch was compiled without support for your GPU architecture")
                        print("\n   🔧 SOLUTIONS:")
                        print("   1. Reinstall PyTorch:")
                        print("      pip uninstall torch torchvision -y")
                        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                        print("\n   2. Use CPU mode:")
                        print("      Add --device cpu to your command")
                        print("\n   ⚠️  Falling back to CPU mode...")
            except Exception as ex:
                print("   Could not detect GPU details. Error:", ex)
                print("\n   🔧 SOLUTION:")
                print("   Use CPU mode: Add --device cpu to your command")
                print("\n   ⚠️  Falling back to CPU mode...")
            
            print("="*60)
            
            # Move models and data to CPU
            args.device = 'cpu'
            i2p_model = i2p_model.cpu()
            l2w_model = l2w_model.cpu()
            for view in data_views:
                to_device(view, device='cpu')
            
            # Retry with CPU
            res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)
            print('finish pre-extracting img tokens (CPU mode)')
        else:
            # Re-raise if it's not a CUDA error
            raise
    
    # Re-organize input views
    input_views = []
    for i in range(num_views):
        input_views.append(dict(
            label=data_views[i]['label'],
            img_tokens=res_feats[i],
            true_shape=data_views[i]['true_shape'],
            img_pos=res_poses[i]
        ))
        # Preserve camera pose if available
        if 'camera_pose' in data_views[i]:
            input_views[i]['camera_pose'] = data_views[i]['camera_pose']
    
    # Determine keyframe stride
    if args.keyframe_stride == -1:
        from slam3r.pipeline.recon_offline_pipeline import adapt_keyframe_stride
        kf_stride = adapt_keyframe_stride(
            input_views, i2p_model,
            win_r=3,
            adapt_min=args.keyframe_adapt_min,
            adapt_max=args.keyframe_adapt_max,
            adapt_stride=args.keyframe_adapt_stride
        )
    else:
        kf_stride = args.keyframe_stride
    
    # Initialize scene
    initial_winsize = min(initial_winsize, num_views // kf_stride)
    assert initial_winsize >= 2, "not enough views for initialization"
    
    initial_pcds, initial_confs, init_ref_id = initialize_scene(
        input_views[:initial_winsize * kf_stride:kf_stride],
        i2p_model,
        winsize=initial_winsize,
        return_ref_id=True
    )
    
    # Setup per-frame results
    init_num = len(initial_pcds)
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    for key in per_frame_res:
        per_frame_res[key] = [None for _ in range(num_views)]
    
    registered_confs_mean = [_ for _ in range(num_views)]
    
    # Setup world coordinates with initial window
    for i in range(init_num):
        per_frame_res['l2w_confs'][i * kf_stride] = initial_confs[i][0].to(args.device)
        registered_confs_mean[i * kf_stride] = per_frame_res['l2w_confs'][i * kf_stride].mean().cpu()
    
    buffering_set_ids = [i * kf_stride for i in range(init_num)]
    
    # Setup world coordinates
    for i in range(init_num):
        input_views[i * kf_stride]['pts3d_world'] = initial_pcds[i]
    
    initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs]
    normed_pts = normalize_views(
        [view['pts3d_world'] for view in input_views[:init_num * kf_stride:kf_stride]],
        initial_valid_masks
    )
    
    # Batch accumulation for octree insertion
    batch_points = []
    batch_colors = []
    batch_sensor_origins = []
    
    for i in range(init_num):
        input_views[i * kf_stride]['pts3d_world'] = normed_pts[i]
        input_views[i * kf_stride]['pts3d_world'][~initial_valid_masks[i]] = 0
        per_frame_res['l2w_pcds'][i * kf_stride] = normed_pts[i]
        
        # Accumulate for batch insertion
        if (i * kf_stride) % args.octree_insert_freq == 0:
            pts3d = to_numpy(normed_pts[i][0])
            if pts3d.ndim == 3:
                pts3d = pts3d.reshape(-1, 3)
            
            conf = initial_confs[i][0].cpu().numpy()
            if conf.ndim == 2:
                conf = conf.reshape(-1)
            
            valid_mask = conf > args.octree_conf_thres
            pts3d = pts3d[valid_mask]
            
            if len(pts3d) > 0:
                rgb = rgb_imgs[i * kf_stride].reshape(-1, 3)
                rgb = rgb[valid_mask]
                
                sensor_origin = extract_camera_pose_from_view(input_views[i * kf_stride])
                
                batch_points.append(pts3d)
                batch_colors.append(rgb)
                batch_sensor_origins.append(sensor_origin)
                
                # Insert batch when it reaches batch_size
                if len(batch_points) >= args.octree_batch_size:
                    # Concatenate all points and colors
                    all_points = np.concatenate(batch_points, axis=0)
                    all_colors = np.concatenate(batch_colors, axis=0)
                    # Use median sensor origin for the batch
                    all_sensor_origins = np.stack(batch_sensor_origins, axis=0)
                    median_sensor_origin = np.median(all_sensor_origins, axis=0)
                    
                    insert_pointcloud_with_color(
                        tree, all_points, all_colors,
                        sensor_origin=median_sensor_origin,
                        max_range=args.octree_max_range,
                        lazy_eval=args.octree_lazy_eval
                    )
                    
                    batch_points.clear()
                    batch_colors.clear()
                    batch_sensor_origins.clear()
    
    # I2P reconstruction for all frames
    local_confs_mean = []
    adj_distance = kf_stride
    
    for view_id in tqdm(range(num_views), desc="I2P reconstruction"):
        if view_id in buffering_set_ids:
            if view_id // kf_stride == init_ref_id:
                per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
            else:
                per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(
                    per_frame_res['l2w_pcds'][view_id], device="cpu"
                )
            per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
            continue
        
        # Construct local window
        sel_ids = [view_id]
        for i in range(1, win_r + 1):
            if view_id - i * adj_distance >= 0:
                sel_ids.append(view_id - i * adj_distance)
            if view_id + i * adj_distance < num_views:
                sel_ids.append(view_id + i * adj_distance)
        
        local_views = [input_views[id] for id in sel_ids]
        ref_id = 0
        
        output = i2p_inference_batch(
            [local_views], i2p_model, ref_id=ref_id,
            tocpu=False, unsqueeze=False
        )['preds']
        
        per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu()
        per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu()
        
        input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d']
        valid_mask = output[ref_id]['conf'] > conf_thres_i2p
        input_views[view_id]['pts3d_cam'] = normalize_views(
            [input_views[view_id]['pts3d_cam']], [valid_mask]
        )[0]
        input_views[view_id]['pts3d_cam'][~valid_mask] = 0
    
    local_confs_mean = [conf.mean() for conf in per_frame_res['i2p_confs']]
    print(f'finish recovering pcds of {len(local_confs_mean)} frames')
    
    # Register remaining frames with L2W
    next_register_id = (init_num - 1) * kf_stride + 1
    milestone = (init_num - 1) * kf_stride + 1
    num_register = 1
    update_buffer_intv = kf_stride * args.update_buffer_intv
    max_buffer_size = args.buffer_size
    strategy = args.buffer_strategy
    candi_frame_id = len(buffering_set_ids)
    
    pbar = tqdm(total=num_views, desc="Registering & building octree")
    pbar.update(next_register_id - 1)
    
    while next_register_id < num_views:
        ni = next_register_id
        max_id = min(ni + num_register, num_views) - 1
        
        cand_ref_ids = buffering_set_ids
        ref_views, sel_pool_ids = scene_frame_retrieve(
            [input_views[i] for i in cand_ref_ids],
            input_views[ni:ni + num_register:2],
            i2p_model, sel_num=num_scene_frame, depth=2
        )
        
        l2w_input_views = ref_views + input_views[ni:max_id + 1]
        output = l2w_inference(
            l2w_input_views, l2w_model,
            ref_ids=list(range(len(ref_views))),
            device=args.device,
            normalize=args.norm_input
        )
        
        src_ids_local = [id + len(ref_views) for id in range(max_id - ni + 1)]
        src_ids_global = [id for id in range(ni, max_id + 1)]
        
        for id in range(len(src_ids_global)):
            output_id = src_ids_local[id]
            view_id = src_ids_global[id]
            conf_map = output[output_id]['conf']
            input_views[view_id]['pts3d_world'] = output[output_id]['pts3d_in_other_view']
            per_frame_res['l2w_confs'][view_id] = conf_map[0]
            registered_confs_mean[view_id] = conf_map[0].mean().cpu()
            per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
            
            # Accumulate for batch insertion
            if view_id % args.octree_insert_freq == 0:
                pts3d = to_numpy(input_views[view_id]['pts3d_world'][0])
                if pts3d.ndim == 3:
                    pts3d = pts3d.reshape(-1, 3)
                
                conf = conf_map[0].cpu().numpy()
                if conf.ndim == 2:
                    conf = conf.reshape(-1)
                
                valid_mask = conf > args.octree_conf_thres
                pts3d = pts3d[valid_mask]
                
                if len(pts3d) > 0:
                    rgb = rgb_imgs[view_id].reshape(-1, 3)
                    rgb = rgb[valid_mask]
                    
                    sensor_origin = extract_camera_pose_from_view(input_views[view_id])
                    
                    batch_points.append(pts3d)
                    batch_colors.append(rgb)
                    batch_sensor_origins.append(sensor_origin)
                    
                    # Insert batch when it reaches batch_size
                    if len(batch_points) >= args.octree_batch_size:
                        # Concatenate all points and colors
                        all_points = np.concatenate(batch_points, axis=0)
                        all_colors = np.concatenate(batch_colors, axis=0)
                        # Use median sensor origin for the batch
                        all_sensor_origins = np.stack(batch_sensor_origins, axis=0)
                        median_sensor_origin = np.median(all_sensor_origins, axis=0)
                        
                        insert_pointcloud_with_color(
                            tree, all_points, all_colors,
                            sensor_origin=median_sensor_origin,
                            max_range=args.octree_max_range,
                            lazy_eval=args.octree_lazy_eval
                        )
                        
                        batch_points.clear()
                        batch_colors.clear()
                        batch_sensor_origins.clear()
        
        next_register_id += len(src_ids_global)
        pbar.update(len(src_ids_global))
        
        # Update buffering set (simplified version)
        if next_register_id - milestone >= update_buffer_intv:
            while next_register_id - milestone >= kf_stride:
                candi_frame_id += 1
                full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
                insert_flag = (not full_flag) or (
                    (strategy == 'fifo') or
                    (strategy == 'reservoir' and np.random.rand() < max_buffer_size / candi_frame_id)
                )
                if not insert_flag:
                    milestone += kf_stride
                    continue
                
                start_ids_offset = max(0, buffering_set_ids[-1] + kf_stride * 3 // 4 - milestone)
                mean_cand_recon_confs = torch.stack([
                    registered_confs_mean[i]
                    for i in range(milestone + start_ids_offset, milestone + kf_stride)
                ])
                mean_cand_local_confs = torch.stack([
                    local_confs_mean[i]
                    for i in range(milestone + start_ids_offset, milestone + kf_stride)
                ])
                mean_cand_recon_confs = (mean_cand_recon_confs - 1) / mean_cand_recon_confs
                mean_cand_local_confs = (mean_cand_local_confs - 1) / mean_cand_local_confs
                mean_cand_confs = mean_cand_recon_confs * mean_cand_local_confs
                
                most_conf_id = mean_cand_confs.argmax().item() + start_ids_offset
                id_to_buffer = milestone + most_conf_id
                buffering_set_ids.append(id_to_buffer)
                
                if full_flag:
                    if strategy == 'reservoir':
                        buffering_set_ids.pop(np.random.randint(max_buffer_size))
                    elif strategy == 'fifo':
                        buffering_set_ids.pop(0)
                
                milestone += kf_stride
        
        for i in range(next_register_id):
            to_device(input_views[i], device=args.device if i in buffering_set_ids else 'cpu')
    
    pbar.close()
    
    # Insert remaining points in batch
    if len(batch_points) > 0:
        all_points = np.concatenate(batch_points, axis=0)
        all_colors = np.concatenate(batch_colors, axis=0)
        all_sensor_origins = np.stack(batch_sensor_origins, axis=0)
        median_sensor_origin = np.median(all_sensor_origins, axis=0)
        
        print(f"\n>> Inserting final batch of {len(batch_points)} frames into octree...")
        insert_pointcloud_with_color(
            tree, all_points, all_colors,
            sensor_origin=median_sensor_origin,
            max_range=args.octree_max_range,
            lazy_eval=args.octree_lazy_eval
        )
    
    # Update octree if lazy evaluation was used
    if args.octree_lazy_eval:
        print(">> Updating octree (lazy evaluation)...")
        try:
            tree.updateInnerOccupancy()
        except AttributeError:
            # Some pyoctomap versions might not have this method
            pass
    
    # Save final point cloud (using original save function)
    # Filter to only views that have been registered (have pts3d_world)
    registered_views = []
    registered_rgb_imgs = []
    registered_confs_list = []
    for i in range(num_views):
        if 'pts3d_world' in input_views[i]:
            registered_views.append(input_views[i])
            registered_rgb_imgs.append(rgb_imgs[i])
            if per_frame_res['l2w_confs'][i] is not None:
                registered_confs_list.append(per_frame_res['l2w_confs'][i])
            else:
                registered_confs_list.append(None)
    
    if len(registered_views) > 0:
        from slam3r.pipeline.recon_offline_pipeline import save_recon
        save_recon(
            registered_views, len(registered_views), save_dir, scene_id,
            args.save_all_views, registered_rgb_imgs,
            registered_confs=registered_confs_list if all(c is not None for c in registered_confs_list) else None,
            num_points_save=num_points_save,
            conf_thres_res=conf_thres_l2w,
            valid_masks=valid_masks[:len(registered_views)] if valid_masks else None
        )
    else:
        print("Warning: No registered views to save")
    
    # Save octree
    print("\n>> Finalizing octree...")
    stats = get_octree_stats(tree)
    print(f">> Octree stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    octree_dir = os.path.join(save_dir, 'octree')
    os.makedirs(octree_dir, exist_ok=True)
    
    if args.save_octree_binary:
        octree_path = os.path.join(octree_dir, f"{scene_id}_octree.bt")
        save_octree(tree, octree_path, binary=True)
    
    if args.save_octree_text:
        octree_path = os.path.join(octree_dir, f"{scene_id}_octree.ot")
        save_octree(tree, octree_path, binary=False)
    
    return tree


def main():
    parser = argparse.ArgumentParser(
        description="Run SLAM3R with incremental octree building"
    )
    
    # Copy arguments from recon.py
    parser.add_argument("--device", type=str, default='cuda', 
                       help="Device to use: 'cuda' or 'cpu'. Will auto-fallback to CPU if CUDA unavailable")
    parser.add_argument('--i2p_weights', type=str, default=None)
    parser.add_argument("--l2w_weights", type=str, default=None)
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", type=str)
    input_group.add_argument("--img_dir", type=str)
    
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--test_name", type=str, required=True)
    
    # SLAM3R args
    parser.add_argument("--keyframe_stride", type=int, default=3)
    parser.add_argument("--initial_winsize", type=int, default=5)
    parser.add_argument("--win_r", type=int, default=3)
    parser.add_argument("--conf_thres_i2p", type=float, default=1.5)
    parser.add_argument("--conf_thres_l2w", type=float, default=12.0)
    parser.add_argument("--num_scene_frame", type=int, default=10)
    parser.add_argument("--num_points_save", type=int, default=2000000)
    parser.add_argument("--norm_input", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--buffer_strategy", type=str, default='reservoir', choices=['reservoir', 'fifo'])
    parser.add_argument("--update_buffer_intv", type=int, default=1)
    parser.add_argument("--keyframe_adapt_min", type=int, default=1)
    parser.add_argument("--keyframe_adapt_max", type=int, default=20)
    parser.add_argument("--keyframe_adapt_stride", type=int, default=1)
    parser.add_argument("--save_all_views", action="store_true")
    parser.add_argument("--max_frames", type=int, default=-1,
                       help="Limit number of frames to process (for debugging, -1 for all)")
    
    # Octree args
    parser.add_argument("--octree_resolution", type=float, default=0.1)
    parser.add_argument("--octree_insert_freq", type=int, default=1,
                       help="Insert every Nth frame into octree (default: 1 = all frames)")
    parser.add_argument("--octree_batch_size", type=int, default=10,
                       help="Batch size for octree insertion (accumulate N frames before inserting, default: 10)")
    parser.add_argument("--octree_max_range", type=float, default=-1.0)
    parser.add_argument("--octree_conf_thres", type=float, default=3.0)
    parser.add_argument("--octree_lazy_eval", action="store_true", default=False,
                       help="Use lazy evaluation for faster insertion (updates deferred)")
    parser.add_argument("--save_octree_binary", action="store_true", default=True)
    parser.add_argument("--save_octree_text", action="store_true", default=False)
    
    args = parser.parse_args()
    
    if not HAS_PYOCTOMAP:
        print("ERROR: pyoctomap is not installed.")
        print("Please install it with: pip install pyoctomap")
        return
    
    # Load models
    from slam3r.models import Image2PointsModel, Local2WorldModel
    import torch
    
    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            # Test CUDA with a simple operation to catch compatibility issues early
            cuda_works = False
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1  # Simple operation to test kernel execution
                del test_tensor
                torch.cuda.empty_cache()
                print(f"   ✅ CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"   ✅ PyTorch CUDA version: {torch.version.cuda}")
                print(f"   ✅ CUDA is working correctly!")
                cuda_works = True
            except RuntimeError as e:
                error_msg = str(e)
                print(f"\n⚠️  CUDA Error: {error_msg}")
                
                # Check if it's a kernel image error (version mismatch)
                if "no kernel image" in error_msg.lower():
                    print("\n" + "="*60)
                    print("   CUDA VERSION MISMATCH DETECTED!")
                    print("="*60)
                    print(f"   - PyTorch was compiled for CUDA {torch.version.cuda}")
                    
                    # Try to detect system CUDA version
                    import subprocess
                    system_cuda = None
                    try:
                        result = subprocess.run(['nvcc', '--version'], 
                                              capture_output=True, text=True, timeout=2)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if 'release' in line.lower():
                                    system_cuda = line.split('release')[1].strip().split(',')[0]
                                    break
                    except:
                        pass
                    
                    if system_cuda:
                        print(f"   - Your system has CUDA {system_cuda}")
                        
                        # Check if versions are actually compatible
                        try:
                            pytorch_major, pytorch_minor = map(int, torch.version.cuda.split('.')[:2])
                            system_major, system_minor = map(int, system_cuda.split('.')[:2])
                            
                            # Same major version (e.g., 12.1 vs 12.3) should be compatible
                            if pytorch_major == system_major:
                                print(f"\n   ✅ CUDA {pytorch_major}.x versions are compatible!")
                                print(f"   PyTorch CUDA {torch.version.cuda} should work with system CUDA {system_cuda}")
                                print(f"   The error might be transient or GPU compute capability issue.")
                                print(f"   Continuing with CUDA anyway...")
                                print(f"\n   If you still get errors during inference, try:")
                                print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                                # Don't fall back to CPU - continue with CUDA
                                cuda_works = True  # Mark as working to continue
                            else:
                                # Different major versions - likely incompatible
                                print("\n   🔧 SOLUTION:")
                                print("   Install PyTorch with CUDA 12.1 support:")
                                print("\n   Run this command:")
                                print("   " + "="*58)
                                print("   pip uninstall torch torchvision -y")
                                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                                print("   " + "="*58)
                                print("\n   Or use the helper script:")
                                print("   python pyoctomap_integration/fix_cuda.py")
                                print("\n   After installing, restart Python and try again with --device cuda")
                                print("\n   ⚠️  Falling back to CPU mode (will be slow)...")
                                print("="*60)
                                device = 'cpu'
                        except:
                            # If version parsing fails, provide generic solution
                            print("\n   🔧 SOLUTION:")
                            print("   Install PyTorch with CUDA 12.1 support:")
                            print("\n   Run this command:")
                            print("   " + "="*58)
                            print("   pip uninstall torch torchvision -y")
                            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                            print("   " + "="*58)
                            print("\n   ⚠️  Falling back to CPU mode (will be slow)...")
                            print("="*60)
                            device = 'cpu'
                    else:
                        print(f"   - Your system CUDA version could not be detected")
                        print("\n   🔧 SOLUTION:")
                        print("   Install PyTorch with CUDA 12.1 support:")
                        print("\n   Run this command:")
                        print("   " + "="*58)
                        print("   pip uninstall torch torchvision -y")
                        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                        print("   " + "="*58)
                        print("\n   ⚠️  Falling back to CPU mode (will be slow)...")
                        print("="*60)
                        device = 'cpu'
                else:
                    print("\n   Unknown CUDA error. Falling back to CPU mode...")
                    device = 'cpu'
            
            # If CUDA test passed or versions are compatible, keep using CUDA
            if cuda_works and device != 'cpu':
                print(f"\n>> Using CUDA device (fast inference)")
    
    # Update args.device to the actual device being used
    args.device = device
    print(f"\n>> Using device: {device}")
    if device == 'cpu':
        print("   Note: CPU mode will be significantly slower than GPU")
    
    def find_checkpoint(checkpoint_name, default_path=None):
        """Find checkpoint file, checking local checkpoints/ directory first."""
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
    
    print("\n>> Loading models...")
    
    # Load I2P model
    if args.i2p_weights:
        i2p_checkpoint = args.i2p_weights
    else:
        # Try to find in checkpoints directory
        i2p_checkpoint = find_checkpoint('slam3r_i2p') or find_checkpoint('i2p')
    
    if i2p_checkpoint and os.path.exists(i2p_checkpoint):
        print(f"   Loading I2P from local checkpoint: {i2p_checkpoint}")
        i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        ckpt = torch.load(i2p_checkpoint, map_location=device)
        i2p_model.load_state_dict(ckpt['model'], strict=False)
        i2p_model.to(device)
    else:
        print("   Loading I2P from HuggingFace (siyan824/slam3r_i2p)...")
        i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        i2p_model.to(device)
    
    # Load L2W model
    if args.l2w_weights:
        l2w_checkpoint = args.l2w_weights
    else:
        # Try to find in checkpoints directory
        l2w_checkpoint = find_checkpoint('slam3r_l2w') or find_checkpoint('l2w')
    
    if l2w_checkpoint and os.path.exists(l2w_checkpoint):
        print(f"   Loading L2W from local checkpoint: {l2w_checkpoint}")
        l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
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
        # Limit num_views if max_frames is set (for debugging - limits loading)
        num_views_limit = args.max_frames if args.max_frames > 0 else -1
        if num_views_limit > 0:
            print(f"   Limiting to {num_views_limit} images for debugging")
        dataset = Seq_Data(
            img_dir=img_dir,
            img_size=224,
            silent=False,
            sample_freq=1,
            start_idx=0,
            num_views=num_views_limit,  # This limits how many images are loaded
            start_freq=1,
            to_tensor=True
        )
        
        # Set scene_names for pipeline compatibility
        if not hasattr(dataset, 'scene_names') or not dataset.scene_names:
            scene_name = os.path.basename(img_dir.rstrip('/'))
            dataset.scene_names = [scene_name]
            
    elif args.img_dir:
        num_views_limit = args.max_frames if args.max_frames > 0 else -1
        if num_views_limit > 0:
            print(f"   Limiting to {num_views_limit} frames for debugging")
        dataset = Seq_Data(
            img_dir=args.img_dir, 
            img_size=224, 
            num_views=num_views_limit,
            to_tensor=True
        )
    
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    
    save_dir = os.path.join(args.save_dir, args.test_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Run reconstruction with octree
    tree = scene_recon_pipeline_with_octree(
        i2p_model, l2w_model, dataset, args, save_dir=save_dir
    )
    
    print(f"\n>> Done! Results saved to {save_dir}")


if __name__ == "__main__":
    main()
