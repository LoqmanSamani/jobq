# ============================================================================
# test_visualize.py — Tests for visualization utilities
# ============================================================================
#
# PURPOSE:
#   Verify that visualization functions run without errors and produce
#   valid output files/figures. These are mostly smoke tests.
#
# TEST CASES:
#   1. test_draw_projected_bbox3d_no_error
#      - Pass a real image + bbox corners → function completes
#      - Returned image has same shape as input
#
#   2. test_draw_masks_overlay
#      - Pass image + masks → overlay image is produced
#      - Output shape matches input
#
#   3. test_draw_gt_vs_pred
#      - Pass GT and pred corners → image with both drawn
#
#   4. test_plot_point_cloud_creates_figure
#      - Pass point cloud + bboxes → matplotlib figure is returned
#      - Figure is valid (not None)
#
#   5. test_plot_training_curves
#      - Create a dummy CSV log file → plot_training_curves runs
#      - Figure is produced
#
#   6. test_save_figure_creates_file
#      - Save figure to tmp path → file exists on disk
#
#   7. test_visualize_batch
#      - Pass a dummy batch → grid image is produced without error
#
#   8. test_headless_mode
#      - Force Agg backend → all viz functions still work
#        (important for running on a server without display)
# ============================================================================
