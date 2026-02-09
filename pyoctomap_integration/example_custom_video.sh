#!/bin/bash
# Example script for running SLAM3R with pyoctomap on custom video/images

# Option 1: Use image directory
IMG_DIR="data/my_scene/images"
TEST_NAME="my_scene_octomap"

# Option 2: Use dataset path (if using SLAM3R dataset format)
# DATASET="data/my_scene"
# Use --dataset instead of --img_dir below

# SLAM3R reconstruction parameters
KEYFRAME_STRIDE=3
UPDATE_BUFFER_INTV=1
WIN_R=3
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5
CONF_THRES_L2W=12
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=2000000

# Octree parameters
OCTREE_RESOLUTION=0.05  # 5cm voxels (adjust based on scene scale)
OCTREE_INSERT_FREQ=1    # Insert every frame
OCTREE_CONF_THRES=3.0   # Confidence threshold
OCTREE_MAX_RANGE=-1.0   # Unlimited range

GPU_ID=-1

python pyoctomap_integration/recon_with_octomap_incremental.py \
    --test_name $TEST_NAME \
    --img_dir "${IMG_DIR}" \
    --device cuda \
    --keyframe_stride $KEYFRAME_STRIDE \
    --win_r $WIN_R \
    --num_scene_frame $NUM_SCENE_FRAME \
    --initial_winsize $INITIAL_WINSIZE \
    --conf_thres_l2w $CONF_THRES_L2W \
    --conf_thres_i2p $CONF_THRES_I2P \
    --num_points_save $NUM_POINTS_SAVE \
    --update_buffer_intv $UPDATE_BUFFER_INTV \
    --octree_resolution $OCTREE_RESOLUTION \
    --octree_insert_freq $OCTREE_INSERT_FREQ \
    --octree_conf_thres $OCTREE_CONF_THRES \
    --octree_max_range $OCTREE_MAX_RANGE \
    --save_octree_binary

echo "Done! Check results in results/${TEST_NAME}/octree/"
