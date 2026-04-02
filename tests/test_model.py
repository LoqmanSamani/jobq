# ============================================================================
# test_model.py — Tests for the neural network architecture
# ============================================================================
#
# PURPOSE:
#   Verify the model can be instantiated, produces outputs of expected shapes,
#   and that gradients flow correctly. These are fast smoke tests, not
#   convergence tests.
#
# TEST CASES:
#   1. test_model_instantiation
#      - Create BBox3DNet(config) → no errors
#      - Print parameter count for sanity check
#
#   2. test_forward_output_shapes
#      - Feed a dummy batch (B=2, 3, H, W) → check prediction dict:
#        • heatmap: (B, 1, H', W')
#        • offset: (B, 2, H', W')
#        • regression: (B, 24, H', W')  — or however corners are encoded
#
#   3. test_forward_with_point_cloud
#      - Feed dummy (image + point cloud) → same output structure
#      - Verify point cloud branch doesn't break shapes
#
#   4. test_forward_no_nan
#      - Forward pass → no NaN or Inf in any output tensor
#
#   5. test_backward_pass
#      - Forward → compute dummy loss (sum of outputs) → backward
#      - Verify all parameter gradients are non-None
#
#   6. test_model_on_cpu
#      - Run forward pass on CPU → should work (important for testing
#        without GPU access)
#
#   7. test_different_input_sizes
#      - Forward with (256, 256) and (320, 480) → both should work
#        (model should handle arbitrary sizes thanks to fully convolutional
#        design; or test the fixed size specified in config)
#
#   8. test_backbone_pretrained_weights_loaded
#      - Check that backbone layers have non-zero weights (pretrained
#        loaded correctly, not random init)
# ============================================================================
