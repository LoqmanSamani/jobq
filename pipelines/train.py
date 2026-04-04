import os
import json
import time

from src.config import Config
from src.model import BBox3DNet
from src.losses import CombinedLoss
from src.dataset import build_dataloaders
from src.trainer import Trainer
from src.utils import set_seed, get_device, count_parameters, load_checkpoint


def setup_dirs(base_dir):
    """create training output directories"""
    dirs = {
        "root": base_dir,
        "checkpoints": os.path.join(base_dir, "checkpoints"),
        "train": os.path.join(base_dir, "train"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def train(config, device, dirs):
    """train model, save history and best checkpoint, return model + metadata"""

    model = BBox3DNet(config, pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    loss_fn = CombinedLoss(
        w_heatmap=config.w_heatmap,
        w_offset=config.w_offset,
        w_corners=config.w_corners,
        w_center=config.w_center,
        w_scale=config.w_scale,
        focal_alpha=config.focal_alpha,
        focal_beta=config.focal_beta,
    )

    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"Data splits: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Batch size: {config.batch_size}, Grad accum: {config.grad_accum_steps}, "
          f"Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"Epochs: {config.epochs}, LR: {config.learning_rate}, "
          f"Warmup: {config.warmup_epochs}, Patience: {config.early_stop_patience}")

    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)

    start_time = time.time()
    history = trainer.fit()
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time / 60:.1f} minutes "
          f"({len(history)} epochs)")

    # save history
    history_path = os.path.join(dirs["train"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # reload best checkpoint
    ckpt_path = os.path.join(dirs["checkpoints"], "best.pt")
    if os.path.exists(ckpt_path):
        epoch, val_loss = load_checkpoint(ckpt_path, model)
        print(f"Best checkpoint: epoch {epoch}, val_loss {val_loss:.4f}")

    # save training summary
    train_summary = {
        "config": {
            "image_size": f"{config.image_height}x{config.image_width}",
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "effective_batch_size": config.batch_size * config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "warmup_epochs": config.warmup_epochs,
            "early_stop_patience": config.early_stop_patience,
            "use_amp": config.use_amp,
            "loss_weights": {
                "heatmap": config.w_heatmap,
                "offset": config.w_offset,
                "corners": config.w_corners,
                "center": config.w_center,
            },
        },
        "training": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "actual_epochs": len(history),
            "best_val_loss": min(h["val"]["total"] for h in history),
            "final_train_loss": history[-1]["train"]["total"],
            "training_time_minutes": round(train_time / 60, 1),
        },
    }
    summary_path = os.path.join(dirs["root"], "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(train_summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: epochs={config.epochs}, lr={config.learning_rate}, "
          f"batch={config.batch_size}, grad_accum={config.grad_accum_steps}")

    dirs = setup_dirs(config.output_dir)
    train(config, device, dirs)
    print("\nDone!")


if __name__ == "__main__":
    main()
