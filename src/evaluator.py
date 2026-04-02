# ============================================================================
# evaluator.py — Evaluation metrics and test loop
# ============================================================================
#
# PURPOSE:
#   Computes quantitative metrics for 3D bounding box predictions and runs
#   the full test-set evaluation. Metrics are chosen to be meaningful for
#   this specific task.
#
# STRUCTURE / CONTENTS:
#   1. Metric functions
#      - corner_distance_error(pred_corners, gt_corners)
#        → Mean L2 distance across all 8 corners (per object, per sample).
#      - center_distance_error(pred_corners, gt_corners)
#        → L2 distance between bbox centers (mean of 8 corners).
#      - iou_3d(pred_corners, gt_corners)
#        → 3D Intersection-over-Union (approximate or exact).
#          Exact 3D IoU is complex; can use a convex-hull approach or
#          approximate with axis-aligned IoU after rotation alignment.
#      - size_error(pred_corners, gt_corners)
#        → Error in estimated bbox dimensions (L, W, H).
#      - rotation_error(pred_corners, gt_corners)
#        → Angular error in estimated orientation.
#
#   2. Matching / assignment
#      - match_predictions_to_gt(pred_boxes, gt_boxes, threshold)
#        → Hungarian matching or greedy assignment based on center distance.
#        → Returns matched pairs, false positives, false negatives.
#
#   3. Aggregate metrics
#      - mean_corner_error(dataset) → averaged over all matched objects
#      - mean_ap(dataset, iou_thresholds) → mAP-style metric at various IoU
#      - precision_recall(dataset, threshold)
#
#   4. Evaluator class
#      - __init__(model, test_loader, config, device)
#      - evaluate()
#          a. Run model on all test samples
#          b. Decode predictions (peak extraction from heatmap, etc.)
#          c. Match predictions to ground truth
#          d. Compute and aggregate all metrics
#          e. Return results dict + per-sample details
#
# NOTES:
#   - 3D IoU can be expensive to compute exactly; start with corner error
#     and center error, then add IoU if time allows.
#   - Report metrics as a summary table for documentation.
# ============================================================================
