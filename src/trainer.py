# ============================================================================
# trainer.py — Training loop
# ============================================================================
#
# PURPOSE:
#   Implements the full training procedure: forward pass, loss computation,
#   backpropagation, logging, checkpointing, and learning rate scheduling.
#   Designed to be resource-aware (AMP, gradient accumulation).
#
# STRUCTURE / CONTENTS:
#   1. Trainer class
#      - __init__(model, train_loader, val_loader, loss_fn, optimizer,
#                 scheduler, config, device)
#
#      - train_one_epoch(epoch)
#          a. Set model to train mode
#          b. Iterate over train_loader
#          c. Forward pass → predictions
#          d. Compute combined loss
#          e. Backward pass with AMP scaler (if mixed precision enabled)
#          f. Gradient accumulation: step optimizer every N mini-batches
#          g. Log batch loss to console / TensorBoard / CSV
#          h. Return epoch-level train metrics
#
#      - validate(epoch)
#          a. Set model to eval mode, torch.no_grad()
#          b. Iterate over val_loader
#          c. Forward + loss computation
#          d. Accumulate evaluation metrics (see evaluator.py)
#          e. Return epoch-level val metrics
#
#      - fit()
#          a. Main loop over epochs
#          b. Call train_one_epoch → validate → scheduler.step
#          c. Early stopping if val loss plateaus
#          d. Save best checkpoint (lowest val loss / highest metric)
#          e. Save periodic checkpoints
#          f. Print epoch summary
#
#   2. Helper functions
#      - build_optimizer(model, config) → Adam / AdamW with weight decay
#      - build_scheduler(optimizer, config) → CosineAnnealing / ReduceOnPlateau
#      - save_checkpoint(model, optimizer, epoch, path)
#      - load_checkpoint(path, model, optimizer) → resume training
#
# NOTES:
#   - Mixed precision (torch.cuda.amp) is essential for limited GPU memory.
#   - Gradient accumulation lets us simulate batch_size=16 even if only
#     batch_size=2 fits in memory.
#   - With 200 samples and small batch size, one epoch is very fast — will
#     need many epochs.  Watch for overfitting via val loss.
#   - Log to both console and a CSV/TensorBoard for later visualization.
# ============================================================================
