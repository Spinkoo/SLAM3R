"""
Simple script to visualize and inspect octree files created by SLAM3R + pyoctomap.
"""

import argparse
import os

try:
    import pyoctomap
    HAS_PYOCTOMAP = True
except ImportError:
    HAS_PYOCTOMAP = False
    print("Error: pyoctomap not installed. Install with: pip install pyoctomap")
    exit(1)

from pyoctomap_integration.octree_utils import load_octree, get_octree_stats


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and inspect octree files"
    )
    parser.add_argument(
        "octree_file",
        type=str,
        help="Path to .bt or .ot octree file"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't visualize"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.octree_file):
        print(f"Error: File not found: {args.octree_file}")
        return
    
    print(f"\n>> Loading octree from {args.octree_file}")
    tree = load_octree(args.octree_file)
    
    # Print statistics
    stats = get_octree_stats(tree)
    print("\n>> Octree Statistics:")
    print(f"   Resolution: {stats['resolution']:.4f} m ({stats['resolution']*100:.2f} cm)")
    print(f"   Number of nodes: {stats['size']}")
    print(f"   Volume: {stats['volume']:.4f} m³")
    print(f"   Memory usage: {stats['memory_usage']:,} bytes ({stats['memory_usage']/1024/1024:.2f} MB)")
    
    if args.stats_only:
        return
    
    # Additional inspection
    print("\n>> Octree Properties:")
    print(f"   Is binary: {tree.isBinary()}")
    print(f"   Is colored: {tree.isColorEnabled()}")
    
    # Get bounding box
    try:
        bbox_min = tree.getBBXMin()
        bbox_max = tree.getBBXMax()
        print(f"\n>> Bounding Box:")
        print(f"   Min: ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f})")
        print(f"   Max: ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f})")
        print(f"   Size: ({bbox_max[0]-bbox_min[0]:.2f}, {bbox_max[1]-bbox_min[1]:.2f}, {bbox_max[2]-bbox_min[2]:.2f}) m")
    except Exception as e:
        print(f"   Could not get bounding box: {e}")
    
    print("\n>> Octree loaded successfully!")
    print("   To visualize in ROS:")
    print("   rosrun octomap_server octomap_server_node", args.octree_file)
    print("   Then open RViz and add Octomap display")


if __name__ == "__main__":
    main()
