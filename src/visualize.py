# ============================================================================
# visualize.py — Visualization utilities
# ============================================================================
#
# PURPOSE:
#   Provides functions to visualize predictions, ground truth, and training
#   progress. Essential for verifying the pipeline works and for the final
#   submission documentation.
#
# STRUCTURE / CONTENTS:
#   1. 2D overlays
#      - draw_projected_bbox3d(image, corners_3d, camera_matrix=None, color)
#        → Project 8 corners onto 2D image plane, draw 12 edges of the cuboid.
#        → If no camera matrix, project by dropping Z or using the point cloud
#          mapping.
#      - draw_masks(image, masks, alpha=0.4)
#        → Overlay instance segmentation masks with transparency.
#      - draw_gt_vs_pred(image, gt_corners, pred_corners)
#        → Side-by-side or overlaid GT (green) vs prediction (red).
#
#   2. 3D visualization
#      - plot_point_cloud_with_bboxes(pc, bboxes, title)
#        → 3D scatter plot of point cloud with wireframe bounding boxes.
#        → Use matplotlib 3D or open3d for interactive viewing.
#
#   3. Training progress plots
#      - plot_training_curves(log_path)
#        → Plot train/val loss curves, learning rate schedule.
#      - plot_metric_curves(log_path)
#        → Plot evaluation metrics over epochs.
#
#   4. Batch visualization
#      - visualize_batch(batch, predictions=None, num_samples=4)
#        → Grid of images from a batch with GT and (optionally) predictions.
#        → Useful as a sanity check during training.
#
#   5. Save helpers
#      - save_figure(fig, path)
#      - save_prediction_gallery(results, output_dir, num_samples=20)
#        → Generate a gallery of prediction visualizations for the report.
#
# NOTES:
#   - Keep matplotlib as the default backend (no GPU needed for viz).
#   - For 3D viz, open3d is nice but optional — matplotlib 3D suffices.
#   - Ensure all viz functions work headless (Agg backend) for server use.
# ============================================================================
