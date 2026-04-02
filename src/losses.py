# ============================================================================
# losses.py — Loss functions for 3D bounding box prediction
# ============================================================================
#
# PURPOSE:
#   Defines all loss components used to train the 3D bbox prediction model.
#   Keeps losses modular so they can be weighted, swapped, or ablated easily.
#
# STRUCTURE / CONTENTS:
#   1. HeatmapFocalLoss
#      - Modified focal loss for the center heatmap (as in CenterNet).
#      - Handles class imbalance: most spatial locations are background.
#      - Args: pred_heatmap, gt_heatmap, alpha, beta
#
#   2. OffsetL1Loss
#      - L1 loss for the sub-pixel center offset regression.
#      - Only computed at ground-truth center locations.
#
#   3. BBox3DCornerLoss
#      - L1 or Smooth-L1 loss on the 8 predicted 3D corner coordinates
#        vs ground-truth corners.
#      - Only computed at ground-truth center locations.
#      - Consider Chamfer distance or IoU-based losses as alternatives.
#
#   4. BBox3DCenterSizeLoss (alternative parameterization)
#      - If bbox is parameterized as center(3) + size(3) + rotation:
#          a. L1 on center offset from the GT center
#          b. L1 on log-dimensions (stabilizes gradient for small objects)
#          c. Angular loss on rotation (e.g., sin/cos parameterization)
#
#   5. MaskAuxLoss (optional)
#      - Binary cross-entropy on predicted instance masks vs GT masks.
#      - Used as an auxiliary supervision signal to improve features.
#
#   6. CombinedLoss
#      - Aggregates all active loss components with configurable weights.
#      - forward(predictions, targets) → scalar total_loss, loss_dict
#      - loss_dict contains each component for logging.
#
# NOTES:
#   - Start with the simplest loss (L1 on corners + focal on heatmap).
#   - Add more components incrementally — measure impact on validation.
#   - Log individual loss components separately for debugging.
# ============================================================================
