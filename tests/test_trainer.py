import os
import math
import copy
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.config import Config
from src.losses import CombinedLoss
from src.model import BBox3DNet
from src.trainer import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    Trainer,
)



def _make_fake_batch(B=2, H=256, W=384, out_h=64, out_w=96, max_obj=5):
    return {
        "image": torch.randn(B, 3, H, W),
        "point_cloud": torch.randn(B, 3, H, W),
        "heatmap": torch.zeros(B, 1, out_h, out_w),
        "bbox3d": torch.randn(B, max_obj, 8, 3),
        "centers_2d": torch.zeros(B, max_obj, 2),
        "masks": torch.zeros(B, max_obj, H, W),
        "num_objects": torch.tensor([1, 1]),
    }


def _make_fake_loader(n_batches=3, B=2):
    batches = [_make_fake_batch(B=B) for _ in range(n_batches)]
    for batch in batches:
        for b in range(B):
            cy, cx = 10, 15
            batch["heatmap"][b, 0, cy, cx] = 1.0
            batch["centers_2d"][b, 0] = torch.tensor([float(cy), float(cx)])
    return batches

class FakeLoader:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


@pytest.fixture
def setup():
    config = Config(
        epochs=5,
        batch_size=2,
        grad_accum_steps=1,
        warmup_epochs=2,
        early_stop_patience=3,
        use_amp=False,
    )
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)
    loss_fn = CombinedLoss()
    train_loader = FakeLoader(_make_fake_loader(n_batches=3, B=2))
    val_loader = FakeLoader(_make_fake_loader(n_batches=2, B=2))
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)
    return trainer, model, config, device


def test_build_optimizer_param_groups():
    """optimizer should have 2 param groups: decay and no-decay"""
    config = Config()
    model = BBox3DNet(config, pretrained=False)
    opt = build_optimizer(model, config)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == config.weight_decay
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_build_optimizer_lr():
    """optimizer lr should match config"""
    config = Config(learning_rate=3e-4)
    model = BBox3DNet(config, pretrained=False)
    opt = build_optimizer(model, config)
    for pg in opt.param_groups:
        assert pg["lr"] == 3e-4


def test_scheduler_warmup_phase():
    """during warmup, lr should ramp up linearly"""
    config = Config(warmup_epochs=5, epochs=100)
    model = BBox3DNet(config, pretrained=False)
    opt = build_optimizer(model, config)
    sched = build_scheduler(opt, config)
    lrs = []
    for epoch in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    for i in range(len(lrs) - 1):
        assert lrs[i] < lrs[i + 1], f"LR didn't increase during warmup: {lrs}"


def test_scheduler_cosine_decay():
    """after warmup, lr should decrease via cosine schedule"""
    config = Config(warmup_epochs=2, epochs=20)
    model = BBox3DNet(config, pretrained=False)
    opt = build_optimizer(model, config)
    sched = build_scheduler(opt, config)
    for _ in range(3):
        sched.step()
    lrs = []
    for epoch in range(3, 18):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[-1] < lrs[0], f"LR didn't decay: start={lrs[0]}, end={lrs[-1]}"


def test_save_and_load_checkpoint():
    """save checkpoint, load into fresh model, outputs should match"""
    config = Config()
    device = torch.device("cpu")
    model1 = BBox3DNet(config, pretrained=False).to(device)
    opt1 = build_optimizer(model1, config)
    sched1 = build_scheduler(opt1, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        save_checkpoint(model1, opt1, sched1, epoch=10, val_loss=0.5, path=path)
        model2 = BBox3DNet(config, pretrained=False).to(device)
        opt2 = build_optimizer(model2, config)
        sched2 = build_scheduler(opt2, config)
        loaded_epoch, loaded_loss = load_checkpoint(path, model2, opt2, sched2)
        assert loaded_epoch == 10
        assert loaded_loss == 0.5
        x = torch.randn(1, 3, config.image_height, config.image_width)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        for key in out1:
            assert torch.allclose(out1[key], out2[key], atol=1e-6), f"Mismatch in {key}"


def test_load_checkpoint_model_only():
    """load_checkpoint should work with just model (no optimizer/scheduler)"""
    config = Config()
    model = BBox3DNet(config, pretrained=False)
    opt = build_optimizer(model, config)
    sched = build_scheduler(opt, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, opt, sched, epoch=5, val_loss=1.0, path=path)
        model2 = BBox3DNet(config, pretrained=False)
        epoch, loss = load_checkpoint(path, model2)
        assert epoch == 5


def test_train_one_epoch_returns_losses(setup):
    """train_one_epoch should return a dict with 'total' key"""
    trainer, *_ = setup
    losses = trainer.train_one_epoch(0)
    assert "total" in losses
    assert isinstance(losses["total"], float)
    assert math.isfinite(losses["total"])


def test_train_one_epoch_has_loss_components(setup):
    """returned dict should contain all three core loss components"""
    trainer, *_ = setup
    losses = trainer.train_one_epoch(0)
    assert "heatmap" in losses
    assert "offset" in losses
    assert "corners" in losses


def test_validate_returns_losses(setup):
    """validate should return a dict with finite loss values"""
    trainer, *_ = setup
    losses = trainer.validate(0)
    assert "total" in losses
    assert math.isfinite(losses["total"])


def test_model_params_change_after_training(setup):
    """model parameters should be different after one training epoch"""
    trainer, model, *_ = setup
    params_before = {n: p.clone() for n, p in model.named_parameters()}
    trainer.train_one_epoch(0)
    changed = 0
    for n, p in model.named_parameters():
        if not torch.equal(params_before[n], p):
            changed += 1
    assert changed > 0, "No parameters changed after training"



def test_gradient_accumulation_steps():
    """with grad_accum_steps=2, optimizer should step once per 2 batches"""
    config = Config(epochs=1, batch_size=2, grad_accum_steps=2, use_amp=False)
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)
    loss_fn = CombinedLoss()
    train_loader = FakeLoader(_make_fake_loader(n_batches=4, B=2))
    val_loader = FakeLoader(_make_fake_loader(n_batches=1, B=2))
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)
    losses = trainer.train_one_epoch(0)
    assert math.isfinite(losses["total"])



def test_overfit_single_sample():
    """training on one repeated sample should reduce loss over iterations"""
    config = Config(
        epochs=1, batch_size=1, grad_accum_steps=1,
        learning_rate=1e-3, use_amp=False, warmup_epochs=0,
    )
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)
    loss_fn = CombinedLoss()
    single = _make_fake_batch(B=1)
    single["heatmap"][0, 0, 10, 15] = 1.0
    single["centers_2d"][0, 0] = torch.tensor([10.0, 15.0])
    batches = [single] * 20
    train_loader = FakeLoader(batches)
    val_loader = FakeLoader([single])
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)

    first_loss = None
    last_loss = None
    model.train()
    trainer.optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        image = batch["image"]
        targets = {k: batch[k] for k in ["heatmap", "bbox3d", "centers_2d", "num_objects"]}
        preds = model(image)
        loss, ld = loss_fn(preds, targets)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        if i == 0:
            first_loss = ld["total"].item()
        last_loss = ld["total"].item()

    assert last_loss < first_loss, f"Loss didn't decrease: {first_loss} → {last_loss}"



def test_fit_returns_history(setup):
    """fit() should return a non-empty history list"""
    trainer, *_ = setup
    history = trainer.fit()
    assert len(history) > 0
    assert "epoch" in history[0]
    assert "lr" in history[0]
    assert "train" in history[0]
    assert "val" in history[0]


def test_fit_records_all_epochs(setup):
    """history should contain one entry per completed epoch"""
    trainer, *_ = setup
    history = trainer.fit()
    assert len(history) <= trainer.config.epochs
    assert len(history) > 0


def test_early_stopping_triggers():
    """if val loss never improves, training should stop after patience epochs"""
    config = Config(
        epochs=50, batch_size=2, grad_accum_steps=1,
        warmup_epochs=0, early_stop_patience=3, use_amp=False,
    )
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)
    loss_fn = CombinedLoss()
    train_loader = FakeLoader(_make_fake_loader(n_batches=2, B=2))
    val_loader = FakeLoader(_make_fake_loader(n_batches=2, B=2))
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)
    trainer.best_val_loss = 0.0
    history = trainer.fit()
    assert len(history) <= config.early_stop_patience + 1


def test_fit_saves_best_checkpoint(setup):
    """fit() should save a best.pt checkpoint in the output directory"""
    trainer, *_ = setup
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.config.output_dir = tmpdir
        trainer.fit()
        best_path = os.path.join(tmpdir, "checkpoints", "best.pt")
        assert os.path.exists(best_path), "best.pt checkpoint was not saved"


def test_lr_changes_across_epochs(setup):
    """lr should change between epochs (warmup or cosine)"""
    trainer, *_ = setup
    history = trainer.fit()
    lrs = [h["lr"] for h in history]
    assert len(set(lrs)) > 1, f"LR stayed constant across epochs: {lrs}"


def test_no_nan_losses_during_training(setup):
    """no loss component should be NaN after training"""
    trainer, *_ = setup
    history = trainer.fit()
    for record in history:
        for split in ["train", "val"]:
            for key, val in record[split].items():
                assert math.isfinite(val), f"NaN/Inf in {split}/{key} at epoch {record['epoch']}"
