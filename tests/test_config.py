import os
import pytest
from src.config import Config




def test_default_config_creates():
    cfg = Config()
    assert isinstance(cfg.batch_size, int)
    assert isinstance(cfg.learning_rate, float)
    assert isinstance(cfg.image_height, int)
    assert isinstance(cfg.image_width, int)
    assert isinstance(cfg.train_ratio, float)
    assert isinstance(cfg.val_ratio, float)
    assert isinstance(cfg.test_ratio, float)
    assert isinstance(cfg.data_dir, str)
    assert isinstance(cfg.output_dir, str)  

def test_default_paths_filled():
    """auto-filled of paths from project_dir"""
    cfg = Config()
    assert cfg.data_dir != ""
    assert cfg.output_dir != ""
    assert cfg.data_dir.endswith("data")
    assert cfg.output_dir.endswith("outputs")


def test_split_ratios_sum_to_one():
    cfg = Config()
    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    assert abs(total - 1.0) < 1e-6


def test_positive_values():
    cfg = Config()
    assert cfg.batch_size > 0
    assert cfg.learning_rate > 0
    assert cfg.epochs > 0
    assert cfg.image_height > 0
    assert cfg.image_width > 0
    assert cfg.max_objects > 0
    assert cfg.output_stride > 0


def test_override():
    cfg = Config(batch_size=8, learning_rate=0.001)
    assert cfg.batch_size == 8
    assert cfg.learning_rate == 0.001
