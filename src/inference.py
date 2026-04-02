# ============================================================================
# inference.py — Inference pipeline and model export (ONNX / TensorRT)
# ============================================================================
#
# PURPOSE:
#   Provides a clean inference API for running predictions on new images,
#   and optionally exports the trained model to ONNX / FP16 / TensorRT
#   for deployment (bonus points in the challenge).
#
# STRUCTURE / CONTENTS:
#   1. Predictor class
#      - __init__(checkpoint_path, config, device)
#          → Load model weights, set to eval mode
#      - preprocess(image_path, pc_path=None)
#          → Load and transform a single sample (same as val transforms)
#      - predict(image, point_cloud=None)
#          → Run forward pass, decode raw outputs into 3D bounding boxes
#          → Peak extraction from heatmap (top-K or threshold-based)
#          → Assemble output: list of {corners: (8,3), score: float}
#      - predict_batch(image_paths)
#          → Batch inference for efficiency
#
#   2. Post-processing
#      - decode_heatmap(heatmap, offset, regression, config)
#        → Extract peaks from heatmap, apply offsets, recover 3D corners
#      - nms_3d(detections, iou_threshold)
#        → Non-maximum suppression in 3D (if needed; CenterNet often
#          produces near-unique peaks, so NMS may be light)
#
#   3. ONNX export
#      - export_to_onnx(model, dummy_input, output_path)
#        → torch.onnx.export with dynamic axes for batch dimension
#      - validate_onnx(onnx_path, dummy_input, torch_output)
#        → Load with onnxruntime, compare outputs for correctness
#
#   4. FP16 / TensorRT export (optional, bonus)
#      - export_to_fp16_onnx(model, dummy_input, output_path)
#      - export_to_tensorrt(onnx_path, output_path, fp16=True)
#        → Use trtexec or tensorrt Python API
#
# NOTES:
#   - The inference pipeline should be usable standalone (no training deps).
#   - ONNX export is a strong bonus signal — prioritize it over TensorRT.
#   - Test that ONNX output matches PyTorch output within tolerance.
# ============================================================================
