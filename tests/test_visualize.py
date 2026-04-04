import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from src.visualize import (
    plot_loss_curves,
    plot_wireframe_overlay,
    plot_bev,
    plot_heatmap_comparison,
    save_figure,
    estimate_intrinsics,
    project_3d_to_2d,
    BBOX_EDGES,
)


@pytest.fixture
def dummy_image():
    """random 256x384 RGB image"""
    return np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)


@pytest.fixture
def dummy_corners():
    """single cuboid with 8 corners in realistic 3D world coordinates"""
    return np.array([
        [-0.05, -0.05, 0.95], [0.05, -0.05, 0.95], [0.05, 0.05, 0.95], [-0.05, 0.05, 0.95],
        [-0.05, -0.05, 1.05], [0.05, -0.05, 1.05], [0.05, 0.05, 1.05], [-0.05, 0.05, 1.05],
    ], dtype=np.float32)


@pytest.fixture
def dummy_intrinsics():
    """camera intrinsics for a 384x256 image"""
    return (500.0, 500.0, 192.0, 128.0)


@pytest.fixture
def dummy_history():
    """3 epochs of training history dicts"""
    return [
        {
            "train": {"heatmap": 1.0, "offset": 0.5, "corners": 0.8, "center": 0.6, "scale": 0.4, "total": 2.9},
            "val":   {"heatmap": 1.1, "offset": 0.6, "corners": 0.9, "center": 0.7, "scale": 0.5, "total": 3.3},
            "lr": 1e-4,
        },
        {
            "train": {"heatmap": 0.7, "offset": 0.3, "corners": 0.5, "center": 0.4, "scale": 0.3, "total": 1.9},
            "val":   {"heatmap": 0.8, "offset": 0.4, "corners": 0.6, "center": 0.5, "scale": 0.35, "total": 2.3},
            "lr": 5e-5,
        },
        {
            "train": {"heatmap": 0.5, "offset": 0.2, "corners": 0.3, "center": 0.25, "scale": 0.2, "total": 1.25},
            "val":   {"heatmap": 0.6, "offset": 0.25, "corners": 0.35, "center": 0.3, "scale": 0.25, "total": 1.5},
            "lr": 2e-5,
        },
    ]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d



class TestPlotLossCurves:

    def test_returns_figure(self, dummy_history):
        fig = plot_loss_curves(dummy_history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_seven_subplots(self, dummy_history):
        fig = plot_loss_curves(dummy_history)
        assert len(fig.axes) == 7
        plt.close(fig)

    def test_single_epoch(self):
        history = [{
            "train": {"heatmap": 1.0, "offset": 0.5, "corners": 0.8, "center": 0.6, "scale": 0.4, "total": 2.9},
            "val":   {"heatmap": 1.1, "offset": 0.6, "corners": 0.9, "center": 0.7, "scale": 0.5, "total": 3.3},
            "lr": 1e-4,
        }]
        fig = plot_loss_curves(history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)



class TestPlotWireframeOverlay:

    def test_returns_figure(self, dummy_image, dummy_corners, dummy_intrinsics):
        fig = plot_wireframe_overlay(dummy_image, [dummy_corners], [dummy_corners + 0.01], dummy_intrinsics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_three_panels(self, dummy_image, dummy_corners, dummy_intrinsics):
        fig = plot_wireframe_overlay(dummy_image, [dummy_corners], [dummy_corners + 0.01], dummy_intrinsics)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_empty_lists(self, dummy_image, dummy_intrinsics):
        fig = plot_wireframe_overlay(dummy_image, [], [], dummy_intrinsics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_boxes(self, dummy_image, dummy_corners, dummy_intrinsics):
        gt = [dummy_corners, dummy_corners + 0.02]
        pred = [dummy_corners + 0.005, dummy_corners + 0.025]
        fig = plot_wireframe_overlay(dummy_image, gt, pred, dummy_intrinsics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)



class TestPlotBev:

    def test_returns_figure(self, dummy_corners):
        fig = plot_bev([dummy_corners], [dummy_corners + 10])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_lists(self):
        fig = plot_bev([], [])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_boxes(self, dummy_corners):
        gt = [dummy_corners, dummy_corners + 20]
        pred = [dummy_corners + 5]
        fig = plot_bev(gt, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)



class TestPlotHeatmapComparison:

    def test_returns_figure(self):
        gt = np.random.rand(64, 96).astype(np.float32)
        pred = np.random.randn(64, 96).astype(np.float32)
        fig = plot_heatmap_comparison(gt, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_panels(self):
        gt = np.random.rand(64, 96).astype(np.float32)
        pred = np.random.randn(64, 96).astype(np.float32)
        fig = plot_heatmap_comparison(gt, pred)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_zeros(self):
        gt = np.zeros((32, 48), dtype=np.float32)
        pred = np.zeros((32, 48), dtype=np.float32)
        fig = plot_heatmap_comparison(gt, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)




class TestSaveFigure:

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




class TestConstants:

    def test_bbox_edges_count(self):
        assert len(BBOX_EDGES) == 12

    def test_bbox_edges_valid_indices(self):
        for i, j in BBOX_EDGES:
            assert 0 <= i < 8
            assert 0 <= j < 8
