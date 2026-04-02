# ============================================================================
# test_transforms.py — Tests for preprocessing and augmentation
# ============================================================================
#
# PURPOSE:
#   Verify that transforms produce outputs of the correct shape/type and
#   that geometric transforms consistently modify ALL modalities (image,
#   point cloud, masks, bbox3d).
#
# TEST CASES:
#   1. test_resize_output_shape
#      - Resize sample to target_size → all outputs have target dimensions
#      - Image: (3, H_new, W_new), PC: (3, H_new, W_new), etc.
#
#   2. test_normalize_image_range
#      - After normalization, image channels roughly zero-mean unit-variance
#
#   3. test_normalize_point_cloud_range
#      - After normalization, XYZ channels are centered/scaled
#
#   4. test_horizontal_flip_consistency
#      - Flip image → verify point cloud X coords are mirrored
#      - Verify bbox3d X coordinates are flipped correspondingly
#      - Flip twice → should recover original
#
#   5. test_color_jitter_preserves_geometry
#      - Apply color jitter → bbox3d and point cloud should be unchanged
#      - Only image pixel values should differ
#
#   6. test_train_transform_output_valid
#      - Run TrainTransform on a real sample → all outputs have valid
#        shapes and no NaN/Inf values
#
#   7. test_val_transform_deterministic
#      - Run ValTransform twice on the same sample → identical outputs
#
#   8. test_augmentation_does_not_lose_objects
#      - After augmentation, num_objects should be preserved (we don't
#        drop objects that go partially out of frame in this design)
# ============================================================================
