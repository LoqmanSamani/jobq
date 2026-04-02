# ============================================================================
# test_config.py — Tests for configuration module
# ============================================================================
#
# PURPOSE:
#   Verify that the configuration system works correctly and produces
#   valid, consistent settings.
#
# TEST CASES:
#   1. test_default_config_valid
#      - Instantiate default config → assert all required fields exist
#      - Assert types are correct (e.g., BATCH_SIZE is int, LR is float)
#
#   2. test_config_override
#      - Override specific fields → verify changes propagate correctly
#      - Ensure overriding one field doesn't break others
#
#   3. test_config_paths_exist
#      - If DATA_DIR is set, verify the path actually exists on disk
#
#   4. test_config_value_ranges
#      - BATCH_SIZE > 0, LEARNING_RATE > 0, EPOCHS > 0
#      - TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT ≈ 1.0
#      - IMAGE_SIZE elements are positive integers
# ============================================================================
