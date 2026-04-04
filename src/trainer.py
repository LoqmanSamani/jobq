import os
import torch
from src.utils import build_optimizer, build_scheduler, save_checkpoint, load_checkpoint





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
            point_cloud = batch["point_cloud"].to(self.device) if "point_cloud" in batch else None
            targets = {
                "heatmap": batch["heatmap"].to(self.device),
                "half_edges": batch["half_edges"].to(self.device),
                "center_3d": batch["center_3d"].to(self.device),
                "centers_2d": batch["centers_2d"].to(self.device),
                "num_objects": batch["num_objects"].to(self.device),
            }
            # forward + loss
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                preds = self.model(image, point_cloud=point_cloud)
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
                "half_edges": batch["half_edges"].to(self.device),
                "center_3d": batch["center_3d"].to(self.device),
                "centers_2d": batch["centers_2d"].to(self.device),
                "num_objects": batch["num_objects"].to(self.device),
            }

            point_cloud = batch["point_cloud"].to(self.device) if "point_cloud" in batch else None
            preds = self.model(image, point_cloud=point_cloud)
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
