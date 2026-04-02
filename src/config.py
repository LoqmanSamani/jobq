# ============================================================================
# config.py — Central configuration and hyperparameters
# ============================================================================
#
# PURPOSE:
#   Single source of truth for all tunable settings in the pipeline.
#   Uses a dataclass (or dict) so every experiment is fully reproducible.
#
# STRUCTURE / CONTENTS:
#   1. Data paths
#      - DATA_DIR: path to the raw data folder
#      - OUTPUT_DIR: path for checkpoints, logs, exports
#
#   2. Data settings
#      - IMAGE_SIZE: target (H, W) after resizing — must be fixed for batching
#      - TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT: ratios or explicit sample lists
#      - NUM_WORKERS: dataloader workers (keep low on limited hardware)
#
#   3. Model settings
#      - BACKBONE: which backbone to use (e.g., resnet18, efficientnet-b0)
#      - USE_POINT_CLOUD: whether to fuse the point cloud branch
#      - MAX_OBJECTS: maximum detections per image (for fixed-size output heads)
#
#   4. Training hyperparameters
#      - BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
#      - EPOCHS, SCHEDULER settings
#      - MIXED_PRECISION: bool — enable AMP to save memory
#      - GRADIENT_ACCUMULATION_STEPS: simulate larger batches on limited GPU
#
#   5. Loss weights
#      - Weights for each loss component (corner loss, center loss, size loss, etc.)
#
#   6. Inference / export settings
#      - ONNX_EXPORT: bool
#      - FP16_EXPORT: bool
#      - CONFIDENCE_THRESHOLD
#
#   7. Reproducibility
#      - SEED
#      - DETERMINISTIC: bool
#
# NOTES:
#   - Keep defaults conservative for a machine with limited GPU memory.
#   - Consider using Hydra or a YAML file later; start with a plain dataclass.
# ============================================================================
