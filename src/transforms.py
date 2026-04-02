# ============================================================================
# transforms.py — Preprocessing and data augmentation
# ============================================================================
#
# PURPOSE:
#   Defines all image/point-cloud/bbox transforms used during training and
#   inference. Keeps transform logic separate from the Dataset so it can be
#   tested and swapped independently.
#
# STRUCTURE / CONTENTS:
#   1. Resize and normalization
#      - resize_sample(image, pc, masks, bbox3d, target_size)
#        → Resize image & point cloud (bilinear), masks (nearest), and
#          adjust bbox3d if the point cloud coordinates are in pixel space.
#      - normalize_image(image, mean, std)
#        → Standard ImageNet normalization or dataset-specific stats.
#      - normalize_point_cloud(pc)
#        → Normalize XYZ channels (zero-center, unit-scale or per-axis).
#
#   2. Training augmentations
#      - random_horizontal_flip(image, pc, masks, bbox3d, p=0.5)
#        → Flip image + mirror X coordinates in bbox3d and point cloud.
#      - random_color_jitter(image, brightness, contrast, saturation, hue)
#        → Standard photometric augmentation (does NOT affect geometry).
#      - random_crop_and_resize(image, pc, masks, bbox3d, scale_range)
#        → Crop a sub-region, adjust all modalities accordingly.
#      - random_rotation(image, pc, masks, bbox3d, angle_range)
#        → Small in-plane rotation; rotate bbox3d corners accordingly.
#      - gaussian_noise(pc, sigma)
#        → Add noise to point cloud to improve robustness.
#
#   3. Compose helpers
#      - TrainTransform(config) — chains augmentations + normalization
#      - ValTransform(config)  — only resize + normalization (no augmentation)
#
# NOTES:
#   - Every geometric augmentation MUST consistently transform image, pc,
#     masks, AND bbox3d together. This is the trickiest part.
#   - With only 200 samples, aggressive augmentation is essential to avoid
#     overfitting. Consider mixup or copy-paste augmentation as well.
#   - Use albumentations or kornia where convenient, but keep the bbox3d
#     adjustments manual since they are 3D coordinates.
# ============================================================================
