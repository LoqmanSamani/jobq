import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.visualize import (
    draw_projected_bbox3d,
    draw_masks,
    draw_gt_vs_pred,
    plot_point_cloud_with_bboxes,
    plot_training_curves,
    plot_metric_curves,
    visualize_batch,
    save_figure,
    save_prediction_gallery,
    BBOX_EDGES,
)


@pytest.fixture
def dummy_image():
    """random 256x384 RGB image"""
    return np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)


@pytest.fixture
def dummy_corners():
    """single cuboid with 8 corners inside image bounds"""
    return np.array([
        [50, 50, 0], [150, 50, 0], [150, 150, 0], [50, 150, 0],
        [50, 50, 1], [150, 50, 1], [150, 150, 1], [50, 150, 1],
    ], dtype=np.float32)


@pytest.fixture
def dummy_masks():
    """2 binary masks on a 256x384 image"""
    masks = np.zeros((2, 256, 384), dtype=bool)
    masks[0, 50:100, 50:100] = True
    masks[1, 150:200, 200:300] = True
    return masks


@pytest.fixture
def dummy_point_cloud():
    """(3, 64, 96) point cloud"""
    return np.random.randn(3, 64, 96).astype(np.float32)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestDrawProjectedBbox3D:
    
    def test_returns_same_shape(self, dummy_image, dummy_corners):
        result = draw_projected_bbox3d(dummy_image, dummy_corners)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8

    def test_does_not_modify_original(self, dummy_image, dummy_corners):
        original = dummy_image.copy()
        draw_projected_bbox3d(dummy_image, dummy_corners)
        np.testing.assert_array_equal(dummy_image, original)

    def test_custom_color(self, dummy_image, dummy_corners):
        result = draw_projected_bbox3d(dummy_image, dummy_corners, color=(255, 0, 0))
        assert result.shape == dummy_image.shape

    def test_corners_out_of_bounds(self, dummy_image):
        """corners outside image should be clamped, not crash"""
        corners = np.array([
            [-100, -100, 0], [1000, -100, 0], [1000, 1000, 0], [-100, 1000, 0],
            [-100, -100, 5], [1000, -100, 5], [1000, 1000, 5], [-100, 1000, 5],
        ], dtype=np.float32)
        result = draw_projected_bbox3d(dummy_image, corners)
        assert result.shape == dummy_image.shape



class TestDrawMasks:
    
    def test_returns_same_shape(self, dummy_image, dummy_masks):
        result = draw_masks(dummy_image, dummy_masks)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8

    def test_no_masks(self, dummy_image):
        masks = np.zeros((0, 256, 384), dtype=bool)
        result = draw_masks(dummy_image, masks)
        # with no masks the image should be unchanged
        np.testing.assert_array_equal(result, dummy_image)

    def test_alpha_effect(self, dummy_image, dummy_masks):
        """different alpha should produce different results"""
        r1 = draw_masks(dummy_image, dummy_masks, alpha=0.0)
        r2 = draw_masks(dummy_image, dummy_masks, alpha=0.9)
        # alpha=0 means no mask overlay, should be very close to original
        np.testing.assert_array_equal(r1, dummy_image)
        assert not np.array_equal(r2, dummy_image)


class TestDrawGtVsPred:

    def test_returns_image(self, dummy_image, dummy_corners):
        gt = [dummy_corners]
        pred = [dummy_corners + 10]
        result = draw_gt_vs_pred(dummy_image, gt, pred)
        assert result.shape == dummy_image.shape

    def test_empty_lists(self, dummy_image):
        result = draw_gt_vs_pred(dummy_image, [], [])
        np.testing.assert_array_equal(result, dummy_image)


class TestPlotPointCloud:

    def test_returns_figure(self, dummy_point_cloud):
        fig = plot_point_cloud_with_bboxes(dummy_point_cloud)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_bboxes(self, dummy_point_cloud, dummy_corners):
        bboxes = dummy_corners.reshape(1, 8, 3)
        fig = plot_point_cloud_with_bboxes(dummy_point_cloud, bboxes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_2d_input(self):
        """(N, 3) shaped point cloud should also work"""
        pc = np.random.randn(1000, 3).astype(np.float32)
        fig = plot_point_cloud_with_bboxes(pc)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_subsampling(self, dummy_point_cloud):
        """should not crash with small subsample"""
        fig = plot_point_cloud_with_bboxes(dummy_point_cloud, subsample=100)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTrainingCurves:

    def test_returns_figure(self):
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.5],
            "val_loss": [1.1, 0.9, 0.7, 0.6],
            "lr": [1e-4, 1e-4, 5e-5, 2e-5],
        }
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_lr(self):
        history = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}
        fig = plot_training_curves(history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_history(self):
        fig = plot_training_curves({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMetricCurves:

    def test_returns_figure(self):
        metrics = {
            "mean_corner_error": [2.0, 1.5, 1.2],
            "mean_iou_3d": [0.1, 0.2, 0.3],
        }
        fig = plot_metric_curves(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_metrics(self):
        fig = plot_metric_curves({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_metric(self):
        metrics = {"loss": [1.0, 0.5]}
        fig = plot_metric_curves(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizeBatch:

    def test_returns_figure(self):
        # normalized batch (B, 3, H, W)
        images = np.random.randn(4, 3, 256, 384).astype(np.float32)
        fig = visualize_batch(images, num_samples=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_gt_boxes(self):
        images = np.random.randn(2, 3, 256, 384).astype(np.float32)
        gt = np.random.randn(2, 5, 8, 3).astype(np.float32) * 50 + 100
        num_obj = np.array([3, 2])
        fig = visualize_batch(images, gt_bboxes=gt, num_objects=num_obj, num_samples=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_predictions(self):
        images = np.random.randn(2, 3, 256, 384).astype(np.float32)
        pred = [np.random.randn(3, 8, 3).astype(np.float32) * 50 + 100,
                np.random.randn(2, 8, 3).astype(np.float32) * 50 + 100]
        fig = visualize_batch(images, pred_bboxes=pred, num_samples=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_sample(self):
        images = np.random.randn(1, 3, 256, 384).astype(np.float32)
        fig = visualize_batch(images, num_samples=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSaveHelpers:

    def test_save_figure(self, tmp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = os.path.join(tmp_dir, "test_fig.png")
        save_figure(fig, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_figure_creates_subdir(self, tmp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        path = os.path.join(tmp_dir, "subdir", "nested", "fig.png")
        save_figure(fig, path)
        assert os.path.exists(path)

    def test_save_prediction_gallery(self, tmp_dir, dummy_image, dummy_corners):
        images = [dummy_image, dummy_image, dummy_image]
        gt = [dummy_corners.reshape(1, 8, 3)] * 3
        pred = [dummy_corners.reshape(1, 8, 3) + 5] * 3
        scores = [np.array([0.85])] * 3

        out_dir = os.path.join(tmp_dir, "gallery")
        paths = save_prediction_gallery(images, gt, pred, scores, out_dir, num_samples=3)
        assert len(paths) == 3
        for p in paths:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0


class TestConstants:

    def test_bbox_edges_count(self):
        assert len(BBOX_EDGES) == 12

    def test_bbox_edges_valid_indices(self):
        for i, j in BBOX_EDGES:
            assert 0 <= i < 8
            assert 0 <= j < 8
