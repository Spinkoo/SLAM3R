#!/bin/bash
# Example script for running SLAM3R with pyoctomap on Replica dataset

TEST_DATASET="data/Replica_demo/room0"
TEST_NAME="replica_octomap"

# SLAM3R reconstruction parameters
KEYFRAME_STRIDE=20
UPDATE_BUFFER_INTV=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5
CONF_THRES_L2W=10
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=1000000

# Octree parameters
OCTREE_RESOLUTION=0.05  # 5cm voxels
OCTREE_INSERT_FREQ=1    # Insert every frame
OCTREE_CONF_THRES=3.0   # Confidence threshold for insertion
OCTREE_MAX_RANGE=-1.0   # Unlimited range

GPU_ID=-1

python pyoctomap_integration/recon_with_octomap_incremental.py \
    --test_name $TEST_NAME \
    --dataset "${TEST_DATASET}" \
    --device cuda \
    --gpu_id $GPU_ID \
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
    --save_octree_binary \
    --save_octree_text

echo "Done! Check results in results/${TEST_NAME}/octree/"
