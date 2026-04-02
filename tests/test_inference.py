# ============================================================================
# test_inference.py — Tests for inference pipeline and export
# ============================================================================
#
# PURPOSE:
#   Verify end-to-end inference produces valid predictions and that model
#   export (ONNX) works correctly.
#
# TEST CASES:
#   1. test_predict_single_image
#      - Load a real sample → predict → returns list of detections
#      - Each detection has 'corners' (8,3) and 'score' (float)
#
#   2. test_predict_batch
#      - Predict on 3 images → returns list of 3 detection lists
#
#   3. test_heatmap_decoding
#      - Create a synthetic heatmap with known peaks → decode
#      - Verify detected positions match the planted peaks
#
#   4. test_nms_removes_duplicates
#      - Create overlapping detections → NMS should reduce count
#
#   5. test_onnx_export_creates_file
#      - Export model to ONNX → file exists and is non-empty
#
#   6. test_onnx_output_matches_pytorch
#      - Run same input through PyTorch model and ONNX model
#      - Outputs should be close (atol=1e-4)
#
#   7. test_prediction_scores_in_range
#      - All detection scores should be in [0, 1]
#
#   8. test_no_predictions_on_blank_image
#      - Feed a blank (zero) image → should return zero or near-zero
#        detections (sanity check)
# ============================================================================
