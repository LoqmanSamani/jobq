import torch
import pytest
from src.losses import (
    HeatmapFocalLoss,
    OffsetL1Loss,
    BBox3DCornerLoss,
    CornerVarianceLoss,
    SizeConsistencyLoss,
    CombinedLoss,
)
from src.utils import _gather_at_centers




@pytest.fixture
def batch():
    """minimal batch with b=2, 3 and 2 objects, output resolution 8×12"""
    B, H, W = 2, 8, 12
    max_obj = 5
    torch.manual_seed(0)
    return {
        "pred_heatmap": torch.randn(B, 1, H, W, requires_grad=True),
        "gt_heatmap": torch.zeros(B, 1, H, W),
        "pred_offset": torch.randn(B, 2, H, W, requires_grad=True),
        "pred_reg": torch.randn(B, 24, H, W, requires_grad=True),
        "bbox3d": torch.randn(B, max_obj, 8, 3),
        "centers_2d": torch.tensor([
            [[3.5, 5.2], [6.1, 9.8], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[4.0, 7.0], [2.3, 3.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ], dtype=torch.float32),
        "num_objects": torch.tensor([3, 2]),
    }



def test_gather_returns_correct_count(batch):
    """should return n_i elements per sample, not max_obj"""
    result = _gather_at_centers(
        batch["pred_reg"], batch["centers_2d"], batch["num_objects"]
    )
    assert len(result) == 2
    assert result[0].shape[0] == 3 
    assert result[1].shape[0] == 2


def test_gather_channel_dim(batch):
    """gathered values should have C channels"""
    result = _gather_at_centers(
        batch["pred_reg"], batch["centers_2d"], batch["num_objects"]
    )
    assert result[0].shape[1] == 24
    assert result[1].shape[1] == 24


def test_gather_zero_objects():
    """sample with 0 objects should return empty tensor"""
    pred = torch.randn(1, 4, 8, 12)
    centers = torch.zeros(1, 5, 2)
    num_obj = torch.tensor([0])
    result = _gather_at_centers(pred, centers, num_obj)
    assert result[0].shape == (0, 4)



def test_focal_loss_perfect_prediction():
    """when prediction matches GT perfectly, loss should be near zero"""
    loss_fn = HeatmapFocalLoss()
    gt = torch.zeros(1, 1, 8, 12)
    gt[0, 0, 4, 6] = 1.0
    pred = torch.full((1, 1, 8, 12), -10.0)
    pred[0, 0, 4, 6] = 10.0 
    loss = loss_fn(pred, gt)
    assert loss.item() < 0.01


def test_focal_loss_positive():
    """random predictions should give a positive loss"""
    loss_fn = HeatmapFocalLoss()
    gt = torch.zeros(1, 1, 8, 12)
    gt[0, 0, 4, 6] = 1.0
    pred = torch.randn(1, 1, 8, 12)
    loss = loss_fn(pred, gt)
    assert loss.item() > 0


def test_focal_loss_gradient():
    """gradients should flow through focal loss"""
    loss_fn = HeatmapFocalLoss()
    gt = torch.zeros(1, 1, 8, 12)
    gt[0, 0, 4, 6] = 1.0
    pred = torch.randn(1, 1, 8, 12, requires_grad=True)
    loss = loss_fn(pred, gt)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_focal_loss_all_background():
    """all-zero GT (no objects) should still produce a valid loss"""
    loss_fn = HeatmapFocalLoss()
    gt = torch.zeros(1, 1, 8, 12)
    pred = torch.randn(1, 1, 8, 12)
    loss = loss_fn(pred, gt)
    assert torch.isfinite(loss)


def test_focal_loss_batch_invariance():
    """loss should scale linearly with number of positive centers, not batch size"""
    loss_fn = HeatmapFocalLoss()
    gt1 = torch.zeros(1, 1, 8, 12)
    gt1[0, 0, 3, 5] = 1.0
    pred1 = torch.randn(1, 1, 8, 12)
    gt2 = gt1.repeat(2, 1, 1, 1)
    pred2 = pred1.repeat(2, 1, 1, 1)
    loss1 = loss_fn(pred1, gt1)
    loss2 = loss_fn(pred2, gt2)
    assert abs(loss1.item() - loss2.item()) / (loss1.item() + 1e-8) < 0.5



def test_offset_loss_perfect():
    """when predicted offset matches GT fractional part, loss should be ~0."""
    loss_fn = OffsetL1Loss()
    B, H, W = 1, 8, 12
    pred = torch.zeros(B, 2, H, W)
    centers = torch.tensor([[[3.5, 5.2], [0.0, 0.0]]])
    num_obj = torch.tensor([1])
    pred[0, 0, 3, 5] = 0.5
    pred[0, 1, 3, 5] = 0.2
    loss = loss_fn(pred, centers, num_obj)
    assert loss.item() < 1e-5


def test_offset_loss_positive(batch):
    """random predictions should give positive offset loss"""
    loss_fn = OffsetL1Loss()
    loss = loss_fn(batch["pred_offset"], batch["centers_2d"], batch["num_objects"])
    assert loss.item() > 0


def test_offset_loss_zero_objects():
    """zero objects should return zero loss without crashing"""
    loss_fn = OffsetL1Loss()
    pred = torch.randn(1, 2, 8, 12)
    centers = torch.zeros(1, 5, 2)
    num_obj = torch.tensor([0])
    loss = loss_fn(pred, centers, num_obj)
    assert loss.item() == 0.0



def test_corner_loss_perfect():
    """perfect prediction should yield ~0 loss"""
    loss_fn = BBox3DCornerLoss()
    B, H, W = 1, 8, 12
    max_obj = 3
    gt_corners = torch.randn(B, max_obj, 8, 3)
    centers = torch.tensor([[[4.0, 6.0], [2.0, 3.0], [0.0, 0.0]]])
    num_obj = torch.tensor([2])
    pred_reg = torch.zeros(B, 24, H, W)
    pred_reg[0, :, 4, 6] = gt_corners[0, 0].reshape(24)
    pred_reg[0, :, 2, 3] = gt_corners[0, 1].reshape(24)
    loss = loss_fn(pred_reg, gt_corners, centers, num_obj)
    assert loss.item() < 1e-5


def test_corner_loss_increases_with_error():
    """larger corner errors should produce larger loss"""
    loss_fn = BBox3DCornerLoss()
    B, H, W = 1, 8, 12
    gt_corners = torch.zeros(B, 3, 8, 3)
    centers = torch.tensor([[[4.0, 6.0], [0.0, 0.0], [0.0, 0.0]]])
    num_obj = torch.tensor([1])
    
    losses = []
    for offset in [0.1, 0.5, 1.0, 2.0]:
        pred_reg = torch.zeros(B, 24, H, W)
        pred_reg[0, :, 4, 6] = offset
        loss = loss_fn(pred_reg, gt_corners, centers, num_obj)
        losses.append(loss.item())
    
    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], f"loss[{i}]={losses[i]} >= loss[{i+1}]={losses[i+1]}"


def test_corner_loss_gradient(batch):
    """gradients should flow to pred_reg"""
    loss_fn = BBox3DCornerLoss()
    loss = loss_fn(
        batch["pred_reg"], batch["bbox3d"], batch["centers_2d"], batch["num_objects"]
    )
    loss.backward()
    assert batch["pred_reg"].grad is not None
    assert torch.isfinite(batch["pred_reg"].grad).all()



def test_variance_loss_identical_corners():
    """if all 8 corners are the same, variance should be 0"""
    loss_fn = CornerVarianceLoss()
    B, H, W = 1, 8, 12
    centers = torch.tensor([[[4.0, 6.0], [0.0, 0.0]]])
    num_obj = torch.tensor([1])
    pred_reg = torch.zeros(B, 24, H, W)
    corner_val = torch.tensor([1.0, 2.0, 3.0]).repeat(8)
    pred_reg[0, :, 4, 6] = corner_val
    loss = loss_fn(pred_reg, centers, num_obj)
    assert loss.item() < 1e-5


def test_variance_loss_positive(batch):
    """random corners should have positive variance"""
    loss_fn = CornerVarianceLoss()
    loss = loss_fn(batch["pred_reg"], batch["centers_2d"], batch["num_objects"])
    assert loss.item() > 0



def test_size_loss_perfect_match():
    """identical pred and GT corners should give 0 size loss"""
    loss_fn = SizeConsistencyLoss()
    B, H, W = 1, 8, 12
    gt_corners = torch.randn(B, 3, 8, 3)
    centers = torch.tensor([[[4.0, 6.0], [0.0, 0.0], [0.0, 0.0]]])
    num_obj = torch.tensor([1])
    pred_reg = torch.zeros(B, 24, H, W)
    pred_reg[0, :, 4, 6] = gt_corners[0, 0].reshape(24)
    loss = loss_fn(pred_reg, gt_corners, centers, num_obj)
    assert loss.item() < 1e-5


def test_size_loss_positive(batch):
    """mismatched corners should give positive size loss"""
    loss_fn = SizeConsistencyLoss()
    loss = loss_fn(
        batch["pred_reg"], batch["bbox3d"], batch["centers_2d"], batch["num_objects"]
    )
    assert loss.item() > 0


def test_combined_loss_returns_dict(batch):
    """combinedLoss should return (scalar, dict) with expected keys"""
    loss_fn = CombinedLoss()
    preds = {
        "heatmap": batch["pred_heatmap"],
        "offset": batch["pred_offset"],
        "regression": batch["pred_reg"],
    }
    targets = {
        "heatmap": batch["gt_heatmap"],
        "bbox3d": batch["bbox3d"],
        "centers_2d": batch["centers_2d"],
        "num_objects": batch["num_objects"],
    }
    total, loss_dict = loss_fn(preds, targets)
    assert total.dim() == 0 
    assert "heatmap" in loss_dict
    assert "offset" in loss_dict
    assert "corners" in loss_dict
    assert "total" in loss_dict


def test_combined_loss_all_nonnegative(batch):
    """all individual loss components should be non-negative"""
    loss_fn = CombinedLoss()
    preds = {
        "heatmap": batch["pred_heatmap"],
        "offset": batch["pred_offset"],
        "regression": batch["pred_reg"],
    }
    targets = {
        "heatmap": batch["gt_heatmap"],
        "bbox3d": batch["bbox3d"],
        "centers_2d": batch["centers_2d"],
        "num_objects": batch["num_objects"],
    }
    _, loss_dict = loss_fn(preds, targets)
    for key, val in loss_dict.items():
        assert val.item() >= 0, f"{key} loss is negative: {val.item()}"


def test_combined_loss_gradient_flows(batch):
    """backward through combined loss should produce gradients on all predictions"""
    loss_fn = CombinedLoss()
    preds = {
        "heatmap": batch["pred_heatmap"],
        "offset": batch["pred_offset"],
        "regression": batch["pred_reg"],
    }
    targets = {
        "heatmap": batch["gt_heatmap"],
        "bbox3d": batch["bbox3d"],
        "centers_2d": batch["centers_2d"],
        "num_objects": batch["num_objects"],
    }
    total, _ = loss_fn(preds, targets)
    total.backward()
    assert batch["pred_heatmap"].grad is not None
    assert batch["pred_offset"].grad is not None
    assert batch["pred_reg"].grad is not None


def test_combined_loss_zero_weight_disables():
    """setting a weight to 0 should make that component = 0 contribution"""
    loss_fn = CombinedLoss(w_heatmap=0.0, w_offset=1.0, w_corners=1.0)
    B, H, W = 1, 8, 12
    torch.manual_seed(0)
    preds = {
        "heatmap": torch.randn(B, 1, H, W),
        "offset": torch.randn(B, 2, H, W),
        "regression": torch.randn(B, 24, H, W),
    }
    gt = torch.zeros(B, 1, H, W)
    gt[0, 0, 4, 6] = 1.0
    targets = {
        "heatmap": gt,
        "bbox3d": torch.randn(B, 5, 8, 3),
        "centers_2d": torch.tensor([[[4.0, 6.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        "num_objects": torch.tensor([1]),
    }
    total_with_heatmap_off, _ = loss_fn(preds, targets)
    
    loss_fn2 = CombinedLoss(w_heatmap=1.0, w_offset=1.0, w_corners=1.0)
    total_with_heatmap_on, _ = loss_fn2(preds, targets)
    
    assert total_with_heatmap_off.item() != total_with_heatmap_on.item()


def test_combined_loss_with_optional_losses(batch):
    """enabling optional losses should add extra keys to loss_dict"""
    loss_fn = CombinedLoss(w_variance=0.01, w_size=0.01)
    preds = {
        "heatmap": batch["pred_heatmap"],
        "offset": batch["pred_offset"],
        "regression": batch["pred_reg"],
    }
    targets = {
        "heatmap": batch["gt_heatmap"],
        "bbox3d": batch["bbox3d"],
        "centers_2d": batch["centers_2d"],
        "num_objects": batch["num_objects"],
    }
    total, loss_dict = loss_fn(preds, targets)
    assert "variance" in loss_dict
    assert "size" in loss_dict
    assert torch.isfinite(total)


def test_combined_loss_finite_on_random(batch):
    """combined loss should never produce NaN or Inf on random inputs"""
    loss_fn = CombinedLoss(w_variance=0.1, w_size=0.1)
    preds = {
        "heatmap": batch["pred_heatmap"],
        "offset": batch["pred_offset"],
        "regression": batch["pred_reg"],
    }
    targets = {
        "heatmap": batch["gt_heatmap"],
        "bbox3d": batch["bbox3d"],
        "centers_2d": batch["centers_2d"],
        "num_objects": batch["num_objects"],
    }
    total, loss_dict = loss_fn(preds, targets)
    assert torch.isfinite(total), f"total loss is not finite: {total.item()}"
    for key, val in loss_dict.items():
        assert torch.isfinite(val), f"{key} is not finite: {val.item()}"
