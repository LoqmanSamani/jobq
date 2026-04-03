import os
import numpy as np
import torch
import pytest
from src.config import Config
from src.dataset import BBox3DDataset, build_dataloaders
from src.utils import get_data_splits, gaussian_2d, generate_heatmap_target
from src.transforms import ValTransform




@pytest.fixture
def config():
    return Config()


def test_splits_no_overlap(config):
    """train, val, test splits must not share any sample"""
    train, val, test = get_data_splits(config.data_dir, seed=42)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_splits_cover_all_samples(config):
    """union of splits should equal the full dataset"""
    train, val, test = get_data_splits(config.data_dir, seed=42)
    all_ids = sorted(train + val + test)
    expected = sorted([
        d for d in os.listdir(config.data_dir)
        if os.path.isdir(os.path.join(config.data_dir, d))
    ])
    assert all_ids == expected


def test_splits_reproducible(config):
    """same seed must produce identical splits"""
    t1, v1, te1 = get_data_splits(config.data_dir, seed=42)
    t2, v2, te2 = get_data_splits(config.data_dir, seed=42)
    assert t1 == t2
    assert v1 == v2
    assert te1 == te2


def test_splits_different_seed(config):
    """different seeds should (very likely) produce different splits"""
    t1, _, _ = get_data_splits(config.data_dir, seed=42)
    t2, _, _ = get_data_splits(config.data_dir, seed=99)
    assert t1 != t2


def test_split_sizes(config):
    """check that split sizes match the configured ratios"""
    train, val, test = get_data_splits(
        config.data_dir, config.train_ratio, config.val_ratio, seed=42
    )
    total = len(train) + len(val) + len(test)
    assert total == 200
    assert len(train) == 140   # 0.7 * 200
    assert len(val) == 30      # 0.15 * 200
    assert len(test) == 30     # 0.15 * 200


def test_gaussian_2d_peak_at_center():
    """gaussian kernel should peak at its center"""
    kernel = gaussian_2d((11, 11), sigma=2.0)
    assert kernel.shape == (11, 11)
    cy, cx = 5, 5
    assert kernel[cy, cx] == kernel.max()
    assert abs(kernel[cy, cx] - 1.0) < 1e-6 


def test_gaussian_2d_symmetric():
    """kernel should be symmetric around its center"""
    kernel = gaussian_2d((9, 9), sigma=1.5)
    np.testing.assert_allclose(kernel, kernel[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(kernel, kernel[::-1, :], atol=1e-6)


def test_heatmap_shape():
    """heatmap should have shape (1, output_h, output_w)"""
    masks = np.zeros((3, 256, 384), dtype=bool)
    masks[0, 50:60, 100:110] = True
    masks[1, 150:160, 200:210] = True
    masks[2, 30:40, 300:310] = True
    heatmap, centers = generate_heatmap_target(masks, 64, 96)
    assert heatmap.shape == (1, 64, 96)
    assert heatmap.dtype == np.float32


def test_heatmap_values_in_range():
    """all heatmap values must be in [0, 1]"""
    masks = np.zeros((5, 256, 384), dtype=bool)
    for i in range(5):
        r, c = np.random.randint(10, 240), np.random.randint(10, 370)
        masks[i, r:r+15, c:c+15] = True
    heatmap, _ = generate_heatmap_target(masks, 64, 96)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_heatmap_has_peaks():
    """heatmap should have non-zero values where objects are"""
    masks = np.zeros((2, 256, 384), dtype=bool)
    masks[0, 100:130, 150:180] = True
    masks[1, 200:220, 300:320] = True
    heatmap, centers = generate_heatmap_target(masks, 64, 96)
    assert (heatmap > 0.5).sum() >= 2


def test_heatmap_centers_2d_shape():
    """centers_2d should have shape (n, 2)"""
    masks = np.zeros((4, 256, 384), dtype=bool)
    for i in range(4):
        masks[i, 50+i*40:60+i*40, 100+i*50:110+i*50] = True
    _, centers = generate_heatmap_target(masks, 64, 96)
    assert centers.shape == (4, 2)
    assert centers.dtype == np.float32


def test_heatmap_empty_mask():
    """an all-zero mask should produce a zero heatmap"""
    masks = np.zeros((1, 256, 384), dtype=bool)
    heatmap, centers = generate_heatmap_target(masks, 64, 96)
    assert heatmap.max() == 0.0



def test_dataset_length(config):
    """dataset length should match the number of sample IDs provided"""
    train, _, _ = get_data_splits(config.data_dir, seed=42)
    ds = BBox3DDataset(config.data_dir, train, config, ValTransform(config))
    assert len(ds) == len(train)


def test_dataset_single_sample_shapes(config):
    """every output tensor from __getitem__ should have the correct shape"""
    train, _, _ = get_data_splits(config.data_dir, seed=42)
    ds = BBox3DDataset(config.data_dir, train, config, ValTransform(config))
    sample = ds[0]

    H, W = config.image_height, config.image_width
    oH = H // config.output_stride
    oW = W // config.output_stride
    M = config.max_objects

    assert sample["image"].shape == (3, H, W)
    assert sample["point_cloud"].shape == (3, H, W)
    assert sample["heatmap"].shape == (1, oH, oW)
    assert sample["bbox3d"].shape == (M, 8, 3)
    assert sample["centers_2d"].shape == (M, 2)
    assert sample["masks"].shape == (M, H, W)
    assert sample["num_objects"].dim() == 0  # scalar


def test_dataset_single_sample_dtypes(config):
    """output tensors should have expected dtypes"""
    train, _, _ = get_data_splits(config.data_dir, seed=42)
    ds = BBox3DDataset(config.data_dir, train, config, ValTransform(config))
    sample = ds[0]

    assert sample["image"].dtype == torch.float32
    assert sample["point_cloud"].dtype == torch.float32
    assert sample["heatmap"].dtype == torch.float32
    assert sample["bbox3d"].dtype == torch.float32
    assert sample["num_objects"].dtype == torch.long


def test_dataset_num_objects_valid(config):
    """num_objects should be between 1 and max_objects"""
    train, _, _ = get_data_splits(config.data_dir, seed=42)
    ds = BBox3DDataset(config.data_dir, train, config, ValTransform(config))
    for i in range(min(10, len(ds))):
        n = ds[i]["num_objects"].item()
        assert 1 <= n <= config.max_objects


def test_dataset_no_nan(config):
    """no NaN or Inf in any output tensor"""
    train, _, _ = get_data_splits(config.data_dir, seed=42)
    ds = BBox3DDataset(config.data_dir, train, config, ValTransform(config))
    sample = ds[0]
    for key in ["image", "point_cloud", "heatmap", "bbox3d", "centers_2d", "masks"]:
        assert not torch.isnan(sample[key]).any(), f"NaN in {key}"
        assert not torch.isinf(sample[key]).any(), f"Inf in {key}"



def test_dataloader_one_batch(config):
    """we should be able to iterate one batch from each DataLoader"""
    config.num_workers = 0
    train_loader, val_loader, test_loader = build_dataloaders(config)

    batch = next(iter(train_loader))
    assert batch["image"].shape[0] == config.batch_size
    assert batch["image"].shape[1:] == (3, config.image_height, config.image_width)

    batch = next(iter(val_loader))
    assert batch["image"].dim() == 4

    batch = next(iter(test_loader))
    assert batch["image"].dim() == 4


def test_dataloader_full_epoch(config):
    """iterate through the full training set without errors"""
    config.num_workers = 0
    train_loader, _, _ = build_dataloaders(config)
    count = 0
    for batch in train_loader:
        count += batch["image"].shape[0]
    assert count > 0
    assert count <= 140
