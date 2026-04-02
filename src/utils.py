# ============================================================================
# utils.py — General-purpose utility functions
# ============================================================================
#
# PURPOSE:
#   Small helper functions shared across the pipeline. Keeps other modules
#   clean by centralizing common operations.
#
# STRUCTURE / CONTENTS:
#   1. Reproducibility
#      - set_seed(seed) → set random, numpy, torch, cuda seeds
#      - set_deterministic(flag) → torch.backends.cudnn.deterministic, etc.
#
#   2. Device management
#      - get_device(config) → torch.device (cuda if available, else cpu)
#      - print_gpu_memory() → log current GPU memory usage
#
#   3. Logging
#      - setup_logger(name, log_file, level) → Python logger with file + console
#      - CSVLogger(path, fieldnames) → simple CSV writer for training metrics
#
#   4. Checkpoint I/O
#      - save_checkpoint(state_dict, path)
#      - load_checkpoint(path) → state_dict
#      - find_latest_checkpoint(checkpoint_dir) → path to most recent
#
#   5. Geometry helpers
#      - corners_to_center_size(corners) → center (3,), size (3,), orientation
#      - center_size_to_corners(center, size, rotation) → (8, 3) corners
#      - compute_bbox3d_iou(box_a, box_b) → approximate 3D IoU
#
#   6. Timing
#      - Timer context manager for profiling sections of code
#
# NOTES:
#   - This file should have NO heavy dependencies (no model imports).
#   - Keep it lightweight — if a helper grows complex, move it to its own module.
# ============================================================================
