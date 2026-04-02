# ============================================================================
# test_trainer.py — Tests for the training loop
# ============================================================================
#
# PURPOSE:
#   Verify the training loop runs without errors and that the model
#   parameters actually update. These are integration-level smoke tests.
#
# TEST CASES:
#   1. test_single_training_step
#      - Run one batch through the trainer → loss is returned and finite
#      - Verify model parameters changed from their initial values
#
#   2. test_overfitting_one_sample
#      - Train on a single sample for many iterations (e.g., 50)
#      - Loss should decrease significantly (proves gradient flow works)
#
#   3. test_gradient_accumulation
#      - Train with accumulation_steps=4 and batch_size=1
#      - Compare to accumulation_steps=1 and batch_size=4
#      - Effective gradient should be similar (not exact due to BN)
#
#   4. test_mixed_precision_no_error
#      - Run a few training steps with AMP enabled → no errors or NaN loss
#
#   5. test_checkpoint_save_and_load
#      - Train a few steps → save checkpoint → load into fresh model
#      - Forward pass on same input → outputs should match
#
#   6. test_validation_runs
#      - Call validate() → returns metrics dict with expected keys
#
#   7. test_early_stopping
#      - Feed fake val losses that plateau → verify training stops early
# ============================================================================
