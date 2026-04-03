import os
import math
import torch
import torch.nn as nn



def build_optimizer(model, config):
    """optimizer with selective weight decay (applied only to non-bias, 
    non-normalization params, which stabilizes training)
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.learning_rate)


def build_scheduler(optimizer, config):
    """applies two-phase lr scheduling:
          - linear warmup for the fisrst warmup epochs (0 -> init lr)
          - cosine decay for the remaining epochs (init lr -> 0)
    """
    warmup = config.warmup_epochs
    total = config.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    """save training state to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """restore model and optimizer/scheduler from a saved checkpoint and return epoch and val_loss"""
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))



class Trainer:
    """
    training:
        - train one epoch (forward + loss + backward)
        - validate one epoch
        - fit loop with scheduling, early stopping, checkpointing
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        
        self.grad_accum_steps = config.grad_accum_steps
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []

    def train_one_epoch(self, epoch):
        """run one full pass over the training set"""
        
        self.model.train()
        running = {} 
        n_batches = 0
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device)
            targets = {
                "heatmap": batch["heatmap"].to(self.device),
                "bbox3d": batch["bbox3d"].to(self.device),
                "centers_2d": batch["centers_2d"].to(self.device),
                "num_objects": batch["num_objects"].to(self.device),
            }
            # forward + loss
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                preds = self.model(image)
                loss, loss_dict = self.loss_fn(preds, targets)
                loss = loss / self.grad_accum_steps
            # backward
            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for key, val in loss_dict.items():
                running[key] = running.get(key, 0.0) + val.item()
            n_batches += 1

        avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self, epoch):
        """run one full pass over val-set """
        
        self.model.eval()
        running = {}
        n_batches = 0
        for batch in self.val_loader:
            image = batch["image"].to(self.device)
            targets = {
                "heatmap": batch["heatmap"].to(self.device),
                "bbox3d": batch["bbox3d"].to(self.device),
                "centers_2d": batch["centers_2d"].to(self.device),
                "num_objects": batch["num_objects"].to(self.device),
            }

            preds = self.model(image)
            _, loss_dict = self.loss_fn(preds, targets)

            for key, val in loss_dict.items():
                running[key] = running.get(key, 0.0) + val.item()
            n_batches += 1

        avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
        return avg_losses

    def fit(self):
        """main training loop"""
        
        for epoch in range(self.config.epochs):
            train_losses = self.train_one_epoch(epoch)
            val_losses = self.validate(epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train": train_losses,
                "val": val_losses,
            }
            self.history.append(record)
            print(
                f"Epoch {epoch:03d} | "
                f"LR {current_lr:.2e} | "
                f"train_loss {train_losses.get('total', 0):.4f} | "
                f"val_loss {val_losses.get('total', 0):.4f}"
            )

            val_total = val_losses.get("total", float("inf"))
            ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")

            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_total,
                    os.path.join(ckpt_dir, "best.pt"),
                )
            else:
                self.patience_counter += 1

            # early stopping
            if self.patience_counter >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={self.config.early_stop_patience})")
                break

        return self.history
