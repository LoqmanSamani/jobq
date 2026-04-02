# ============================================================================
# dataset.py — Dataset class and data loading utilities
# ============================================================================
#
# PURPOSE:
#   Implements a PyTorch Dataset that loads RGB images, 3D bounding boxes,
#   point clouds, and instance masks from the provided data folder.
#   Also provides helper functions for train/val/test splitting and
#   building DataLoaders.
#
# STRUCTURE / CONTENTS:
#   1. BBox3DDataset(torch.utils.data.Dataset)
#      - __init__: receives data_dir, list of sample IDs, transform, config
#      - __len__: returns number of samples
#      - __getitem__:
#          a. Load rgb.jpg → PIL Image or numpy array
#          b. Load pc.npy → (3, H, W) point cloud
#          c. Load bbox3d.npy → (N, 8, 3) 3D bounding box corners
#          d. Load mask.npy → (N, H, W) boolean instance segmentation masks
#          e. Apply transforms (resize, normalize, augment)
#          f. Pad/truncate objects to MAX_OBJECTS for batching
#          g. Return dict: {image, point_cloud, bbox3d, masks, num_objects}
#
#   2. collate_fn(batch)
#      - Custom collate to handle variable number of objects per sample
#      - Stack images/point clouds; pad bbox3d and masks to max objects in batch
#
#   3. get_data_splits(data_dir, train_ratio, val_ratio, seed)
#      - Scan data_dir for sample UUIDs
#      - Shuffle deterministically → split into train / val / test lists
#
#   4. build_dataloaders(config)
#      - Convenience function: creates Dataset + DataLoader for each split
#      - Applies different transforms for train vs val/test
#
# NOTES:
#   - Images have varying resolutions → must resize to a common size.
#   - Number of objects per image varies (3–15) → need padding strategy.
#   - Point cloud is "organized" (image-shaped) so it resizes alongside RGB.
#   - With only 200 samples, heavy augmentation is critical.
# ============================================================================
