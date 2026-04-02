# ============================================================================
# test_evaluator.py — Tests for evaluation metrics
# ============================================================================
#
# PURPOSE:
#   Verify metric functions compute correctly on known inputs.
#
# TEST CASES:
#   1. test_corner_error_zero_on_identical
#      - pred == gt → corner error == 0
#
#   2. test_corner_error_known_value
#      - Shift all corners by (1,0,0) → mean error should be 1.0
#
#   3. test_center_error
#      - Known center offset → exact expected value
#
#   4. test_matching_perfect_assignment
#      - One pred per GT, close enough → all matched
#
#   5. test_matching_with_false_positives
#      - Extra predictions → counted as false positives
#
#   6. test_matching_with_false_negatives
#      - Missing predictions → counted as false negatives
#
#   7. test_iou_3d_identical_boxes
#      - Same box → IoU == 1.0
#
#   8. test_iou_3d_no_overlap
#      - Distant boxes → IoU == 0.0
#
#   9. test_evaluator_end_to_end
#      - Run Evaluator on a tiny dataset → returns results dict with
#        all expected keys (mean_corner_error, etc.)
# ============================================================================
