# ============================================================================
# test_losses.py — Tests for loss functions
# ============================================================================
#
# PURPOSE:
#   Verify that each loss function computes correctly, returns valid
#   gradients, and handles edge cases.
#
# TEST CASES:
#   1. test_heatmap_focal_loss_zero_on_perfect
#      - pred == gt heatmap → loss should be zero (or near-zero)
#
#   2. test_heatmap_focal_loss_positive
#      - pred != gt heatmap → loss should be positive
#
#   3. test_heatmap_focal_loss_gradient
#      - Compute loss → backward → verify gradients are non-None and finite
#
#   4. test_corner_loss_zero_on_perfect
#      - Predicted corners == GT corners → loss should be zero
#
#   5. test_corner_loss_increases_with_error
#      - Perturb predictions increasingly → loss should increase monotonically
#
#   6. test_offset_loss_basic
#      - Known offset error → verify L1 value matches expected
#
#   7. test_combined_loss_returns_dict
#      - Run CombinedLoss → returns total scalar + dict of component losses
#      - All components are non-negative
#
#   8. test_combined_loss_weights
#      - Set weight=0 for one component → that component should not
#        contribute to total loss
#
#   9. test_loss_on_batch
#      - Create a mini-batch of dummy predictions and targets
#      - Compute loss → verify shape is scalar, value is finite
# ============================================================================
