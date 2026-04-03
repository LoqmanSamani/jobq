import numpy as np
import torch
import pytest
from src.config import Config
from src.model import BBox3DNet
from src.evaluator import Evaluator
from src.utils import (
    corner_error,
    center_error,
    iou_3d,
    size_error,
    _box_dimensions,
    match_predictions_to_gt,
    precision_recall,
    evaluate_sample,
)




def _make_box(center=(0, 0, 0), half=(0.5, 0.5, 0.5)):
    """standard cuboid with 8 corners"""
    cx, cy, cz = center
    hx, hy, hz = half
    return np.array([
        [cx - hx, cy - hy, cz - hz],  
        [cx + hx, cy - hy, cz - hz],  
        [cx + hx, cy + hy, cz - hz],  
        [cx - hx, cy + hy, cz - hz],  
        [cx - hx, cy - hy, cz + hz],  
        [cx + hx, cy - hy, cz + hz],  
        [cx + hx, cy + hy, cz + hz],  
        [cx - hx, cy + hy, cz + hz],  
    ], dtype=np.float32)


def test_corner_error_identical():
    """identical boxes should have zero corner error"""
    box = _make_box()
    assert corner_error(box, box) == pytest.approx(0.0, abs=1e-6)


def test_corner_error_known_shift():
    """shifting all corners by (1,0,0) → mean corner error = 1.0"""
    gt = _make_box()
    pred = gt + np.array([1.0, 0.0, 0.0])
    assert corner_error(pred, gt) == pytest.approx(1.0, abs=1e-5)


def test_corner_error_positive():
    """different boxes should have positive error"""
    pred = _make_box(center=(0.3, 0.2, 0.1))
    gt = _make_box(center=(0, 0, 0))
    assert corner_error(pred, gt) > 0



def test_center_error_identical():
    """identical boxes → center error = 0"""
    box = _make_box()
    assert center_error(box, box) == pytest.approx(0.0, abs=1e-6)


def test_center_error_known_offset():
    """box shifted by (3, 4, 0) → center error = 5.0 (3-4-5 triangle)"""
    gt = _make_box(center=(0, 0, 0))
    pred = _make_box(center=(3, 4, 0))
    assert center_error(pred, gt) == pytest.approx(5.0, abs=1e-5)


def test_center_error_less_than_corner_error():
    """center error should be ≤ corner error when box is translated"""
    gt = _make_box()
    pred = _make_box(center=(0.5, 0.5, 0.5), half=(0.3, 0.3, 0.3)) 
    ce = center_error(pred, gt)
    coe = corner_error(pred, gt)
    assert ce <= coe


def test_iou_identical_boxes():
    """identical boxes should have IoU = 1.0"""
    box = _make_box()
    assert iou_3d(box, box) == pytest.approx(1.0, abs=1e-2)


def test_iou_no_overlap():
    """distant non-overlapping boxes should have iou ≈ 0"""
    box1 = _make_box(center=(0, 0, 0), half=(0.5, 0.5, 0.5))
    box2 = _make_box(center=(10, 10, 10), half=(0.5, 0.5, 0.5))
    assert iou_3d(box1, box2) == pytest.approx(0.0, abs=1e-3)


def test_iou_partial_overlap():
    """partially overlapping boxes should have iou between 0 and 1"""
    box1 = _make_box(center=(0, 0, 0), half=(0.5, 0.5, 0.5))
    box2 = _make_box(center=(0.5, 0, 0), half=(0.5, 0.5, 0.5))
    iou = iou_3d(box1, box2)
    assert 0.0 < iou < 1.0


def test_iou_contained_box():
    """a smaller box fully inside a larger one should have iou > 0"""
    big = _make_box(center=(0, 0, 0), half=(1.0, 1.0, 1.0))
    small = _make_box(center=(0, 0, 0), half=(0.3, 0.3, 0.3))
    iou = iou_3d(big, small)
    assert iou > 0


def test_iou_symmetric():
    """iou(a, b) should equal iou(b, a)"""
    box1 = _make_box(center=(0, 0, 0), half=(0.5, 0.5, 0.5))
    box2 = _make_box(center=(0.3, 0.2, 0.1), half=(0.4, 0.6, 0.5))
    assert iou_3d(box1, box2) == pytest.approx(iou_3d(box2, box1), abs=1e-6)



def test_size_error_identical():
    """identical boxes → size error = 0"""
    box = _make_box(half=(0.5, 0.3, 0.7))
    assert size_error(box, box) == pytest.approx(0.0, abs=1e-5)


def test_size_error_known_diff():
    """boxes with different half-extents should have known size error"""
    gt = _make_box(half=(0.5, 0.5, 0.5))  
    pred = _make_box(half=(1.0, 0.5, 0.5)) 
    assert size_error(pred, gt) == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_box_dimensions():
    """_box_dimensions should give correct l, w, h"""
    box = _make_box(half=(1.0, 0.5, 0.3))
    dims = _box_dimensions(box)
    np.testing.assert_allclose(sorted(dims), sorted([2.0, 1.0, 0.6]), atol=1e-5)



def test_matching_perfect():
    """one prediction close to one GT → matched with no fp/fn"""
    pred = np.array([_make_box(center=(0.01, 0, 0))])
    gt = np.array([_make_box(center=(0, 0, 0))])
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 1
    assert len(fp) == 0
    assert len(fn) == 0


def test_matching_false_positive():
    """extra prediction with no nearby gt -> false positive"""
    pred = np.array([
        _make_box(center=(0, 0, 0)),
        _make_box(center=(10, 10, 10)),
    ])
    gt = np.array([_make_box(center=(0, 0, 0))])
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 1
    assert len(fp) == 1
    assert len(fn) == 0


def test_matching_false_negative():
    """gt with no nearby prediction -> false negative"""
    pred = np.array([_make_box(center=(0, 0, 0))])
    gt = np.array([
        _make_box(center=(0, 0, 0)),
        _make_box(center=(10, 10, 10)),
    ])
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 1
    assert len(fp) == 0
    assert len(fn) == 1


def test_matching_empty_preds():
    """no predictions → all GT are false negatives"""
    pred = np.zeros((0, 8, 3))
    gt = np.array([_make_box()])
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 0
    assert len(fp) == 0
    assert len(fn) == 1


def test_matching_empty_gt():
    """no gt -> all predictions are false positives"""
    pred = np.array([_make_box()])
    gt = np.zeros((0, 8, 3))
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 0
    assert len(fp) == 1
    assert len(fn) == 0


def test_matching_both_empty():
    """no preds, no gt -> everything empty"""
    pred = np.zeros((0, 8, 3))
    gt = np.zeros((0, 8, 3))
    matches, fp, fn = match_predictions_to_gt(pred, gt, max_dist=0.5)
    assert len(matches) == 0
    assert len(fp) == 0
    assert len(fn) == 0


def test_precision_recall_perfect():
    """perfect prediction -> precision = recall = 1 at all thresholds"""
    pred = np.array([_make_box(center=(0, 0, 0))])
    gt = np.array([_make_box(center=(0.01, 0, 0))])
    pr = precision_recall(pred, gt, thresholds=(0.1, 0.5))
    for t, vals in pr.items():
        assert vals["precision"] == pytest.approx(1.0)
        assert vals["recall"] == pytest.approx(1.0)


def test_precision_recall_tight_threshold():
    """tight threshold with imperfect match -> lower recall"""
    pred = np.array([_make_box(center=(0.3, 0, 0))])
    gt = np.array([_make_box(center=(0, 0, 0))])
    pr = precision_recall(pred, gt, thresholds=(0.1,))
    assert pr[0.1]["recall"] == 0.0


def test_evaluate_sample_keys():
    """evaluate_sample should return dict with all expected keys"""
    pred = np.array([_make_box(center=(0.01, 0, 0))])
    gt = np.array([_make_box(center=(0, 0, 0))])
    result = evaluate_sample(pred, gt)
    expected_keys = {
        "corner_errors", "center_errors", "ious", "size_errors",
        "n_matches", "n_false_positives", "n_false_negatives", "precision_recall",
    }
    assert expected_keys.issubset(result.keys())


def test_evaluate_sample_matched_counts():
    """one pred, one GT, close enough -> 1 match"""
    pred = np.array([_make_box(center=(0.1, 0, 0))])
    gt = np.array([_make_box(center=(0, 0, 0))])
    result = evaluate_sample(pred, gt, max_dist=0.5)
    assert result["n_matches"] == 1
    assert result["n_false_positives"] == 0
    assert result["n_false_negatives"] == 0
    assert len(result["corner_errors"]) == 1
    assert len(result["ious"]) == 1


class FakeEvalLoader:
    """yields pre-built batches for Evaluator testing"""
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def test_evaluator_end_to_end():
    """Evaluator should run on a fake batch and return summary with expected keys"""
    config = Config()
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)

    B = 2
    H, W = config.image_height, config.image_width
    oH = H // config.output_stride
    oW = W // config.output_stride
    max_obj = config.max_objects

    batch = {
        "image": torch.randn(B, 3, H, W),
        "heatmap": torch.zeros(B, 1, oH, oW),
        "bbox3d": torch.randn(B, max_obj, 8, 3),
        "centers_2d": torch.zeros(B, max_obj, 2),
        "num_objects": torch.tensor([2, 1]),
        "masks": torch.zeros(B, max_obj, H, W),
        "point_cloud": torch.randn(B, 3, H, W),
    }

    loader = FakeEvalLoader([batch])
    evaluator = Evaluator(model, loader, config, device, top_k=10, conf_thresh=0.1)
    summary, per_sample = evaluator.evaluate()

    assert "mean_corner_error" in summary
    assert "mean_center_error" in summary
    assert "mean_iou_3d" in summary
    assert "mean_size_error" in summary
    assert "precision" in summary
    assert "recall" in summary
    assert summary["n_samples"] == B
    assert len(per_sample) == B


def test_evaluator_no_detections():
    """with very high conf_thresh, no detections should be made -> all fn"""
    config = Config()
    device = torch.device("cpu")
    model = BBox3DNet(config, pretrained=False)

    B = 1
    H, W = config.image_height, config.image_width
    oH = H // config.output_stride
    oW = W // config.output_stride

    batch = {
        "image": torch.randn(B, 3, H, W),
        "heatmap": torch.zeros(B, 1, oH, oW),
        "bbox3d": torch.randn(B, config.max_objects, 8, 3),
        "centers_2d": torch.zeros(B, config.max_objects, 2),
        "num_objects": torch.tensor([3]),
        "masks": torch.zeros(B, config.max_objects, H, W),
        "point_cloud": torch.randn(B, 3, H, W),
    }

    loader = FakeEvalLoader([batch])
    evaluator = Evaluator(model, loader, config, device, top_k=5, conf_thresh=0.99)
    summary, _ = evaluator.evaluate()
    assert summary["total_false_negatives"] == 3
    assert summary["total_true_positives"] == 0
