# ============================================================================
# test_utils.py — Tests for utility functions
# ============================================================================
#
# PURPOSE:
#   Verify helper functions in utils.py work correctly.
#
# TEST CASES:
#   1. test_set_seed_reproducibility
#      - Set seed → generate random numbers → reset seed → same numbers
#
#   2. test_corners_to_center_size_roundtrip
#      - corners → center, size → back to corners → match original
#
#   3. test_center_size_to_corners_known
#      - Known center (0,0,0), size (2,2,2) → 8 corners at (±1, ±1, ±1)
#
#   4. test_get_device
#      - Returns 'cpu' when CUDA unavailable, 'cuda' otherwise
#
#   5. test_csv_logger
#      - Log a few rows → read CSV → values match
#
#   6. test_checkpoint_save_load_roundtrip
#      - Save a dummy state_dict → load → matches original
#
#   7. test_timer_context_manager
#      - Use Timer → elapsed time is positive and reasonable
# ============================================================================
