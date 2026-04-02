# ============================================================================
# test_dataset.py — Tests for dataset loading and data integrity
# ============================================================================
#
# PURPOSE:
#   Verify that the dataset loads correctly, returns tensors of the expected
#   shapes/types, and that the collate function handles variable-size batches.
#
# TEST CASES:
#   1. test_dataset_length
#      - Create dataset with known sample list → assert __len__ is correct
#
#   2. test_single_sample_shapes
#      - Load one sample → check:
#        • image shape: (3, H, W)
#        • point_cloud shape: (3, H, W)
#        • bbox3d shape: (MAX_OBJECTS, 8, 3) after padding
#        • masks shape: (MAX_OBJECTS, H, W) after padding
#        • num_objects: int, > 0
#
#   3. test_single_sample_dtypes
#      - image: float32, point_cloud: float32
#      - bbox3d: float32, masks: bool or float32
#
#   4. test_single_sample_value_ranges
#      - Normalized image values in roughly [-3, 3] (after ImageNet norm)
#      - num_objects in [1, MAX_OBJECTS]
#
#   5. test_collate_fn
#      - Create a mini batch of 3 samples → run collate_fn
#      - Assert batch tensors are stacked: (B, 3, H, W) for image, etc.
#
#   6. test_data_splits_no_overlap
#      - Generate train/val/test splits → assert no sample appears in
#        more than one split
#      - Assert union covers all samples
#
#   7. test_data_splits_reproducibility
#      - Same seed → same splits
#      - Different seed → different splits
#
#   8. test_dataloader_iteration
#      - Build a dataloader → iterate one full epoch without errors
# ============================================================================
