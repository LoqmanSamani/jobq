import numpy as np
import pytest
from src.config import Config
from src.transforms import (
    resize_sample,
    normalize_image,
    normalize_point_cloud,
    random_horizontal_flip,
    random_color_jitter,
    TrainTransform,
    ValTransform,
)





@pytest.fixture
def dummy_sample():
    
    H, W = 451, 706
    N = 5
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    pc = np.random.randn(3, H, W).astype(np.float64)
    masks = np.random.rand(N, H, W) > 0.8
    
    return image, pc, masks


@pytest.fixture
def config():
    return Config()


def test_resize_output_shapes(dummy_sample, config):
    """all outputs should have the target resolution"""
    image, pc, masks = dummy_sample
    img_r, pc_r, masks_r = resize_sample(
        image, pc, masks, config.image_height, config.image_width
    )
    assert img_r.shape == (config.image_height, config.image_width, 3)
    assert pc_r.shape == (3, config.image_height, config.image_width)
    assert masks_r.shape == (5, config.image_height, config.image_width)


def test_resize_preserves_dtypes(dummy_sample, config):
    """resize should keep image as uint8, pc as float, masks as bool"""
    image, pc, masks = dummy_sample
    img_r, pc_r, masks_r = resize_sample(
        image, pc, masks, config.image_height, config.image_width
    )
    assert img_r.dtype == np.uint8
    assert pc_r.dtype in (np.float32, np.float64)
    assert masks_r.dtype == bool


def test_resize_empty_masks(config):
    """resize should handle zero-object masks (n=0)"""
    H, W = 400, 600
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    pc = np.random.randn(3, H, W).astype(np.float64)
    masks = np.zeros((0, H, W), dtype=bool)
    _, _, masks_r = resize_sample(
        image, pc, masks, config.image_height, config.image_width
    )
    assert masks_r.shape == (0, config.image_height, config.image_width)



def test_normalize_image_shape_and_dtype():
    """output should be (3, h, w) float32"""
    image = np.random.randint(0, 256, (256, 384, 3), dtype=np.uint8)
    result = normalize_image(image)
    assert result.shape == (3, 256, 384)
    assert result.dtype == np.float32


def test_normalize_image_range():
    """normalized image values should be roughly in [-3, 3]"""
    image = np.random.randint(0, 256, (256, 384, 3), dtype=np.uint8)
    result = normalize_image(image)
    assert result.min() > -4.0
    assert result.max() < 4.0


def test_normalize_point_cloud_zero_mean():
    pc = np.random.randn(3, 256, 384).astype(np.float64) + 5.0 
    result = normalize_point_cloud(pc)
    # each channel mean should be near zero
    for ch in range(3):
        assert abs(result[ch].mean()) < 0.1


def test_normalize_point_cloud_dtype():
    pc = np.random.randn(3, 100, 100).astype(np.float64)
    result = normalize_point_cloud(pc)
    assert result.dtype == np.float32


def test_horizontal_flip_shape_preserved(dummy_sample):
    """flip should not change tensor shapes"""
    image, pc, masks = dummy_sample
    img_f, pc_f, masks_f = random_horizontal_flip(image, pc, masks, p=1.0)
    assert img_f.shape == image.shape
    assert pc_f.shape == pc.shape
    assert masks_f.shape == masks.shape


def test_horizontal_flip_double_recovers_original(dummy_sample):
    """flipping twice should recover the original (except x negation)"""
    image, pc, masks = dummy_sample
    img1, pc1, masks1 = random_horizontal_flip(image, pc, masks, p=1.0)
    img2, pc2, masks2 = random_horizontal_flip(img1, pc1, masks1, p=1.0)
    np.testing.assert_array_equal(img2, image)
    np.testing.assert_array_equal(masks2, masks)
    np.testing.assert_allclose(pc2, pc, atol=1e-6)


def test_horizontal_flip_p_zero_no_change(dummy_sample):
    """p=0 means never flip"""
    image, pc, masks = dummy_sample
    img_f, pc_f, masks_f = random_horizontal_flip(image, pc, masks, p=0.0)
    np.testing.assert_array_equal(img_f, image)
    np.testing.assert_array_equal(pc_f, pc)


def test_color_jitter_preserves_shape(dummy_sample):
    """color jitter should not change image shape or dtype"""
    image = dummy_sample[0]
    result = random_color_jitter(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_color_jitter_values_valid(dummy_sample):
    """color jitter output should stay in [0, 255]"""
    image = dummy_sample[0]
    result = random_color_jitter(image)
    assert result.min() >= 0
    assert result.max() <= 255


def test_train_transform_output_shapes(dummy_sample, config):
    """trainTransform should produce correctly shaped and typed outputs"""
    image, pc, masks = dummy_sample
    transform = TrainTransform(config)
    img_t, pc_t, masks_t = transform(image, pc, masks)
    assert img_t.shape == (3, config.image_height, config.image_width)
    assert img_t.dtype == np.float32
    assert pc_t.shape == (3, config.image_height, config.image_width)
    assert pc_t.dtype == np.float32
    assert masks_t.shape == (5, config.image_height, config.image_width)


def test_val_transform_deterministic(dummy_sample, config):
    """val transform applied twice to the same input should give identical output"""
    image, pc, masks = dummy_sample
    transform = ValTransform(config)
    img1, pc1, masks1 = transform(image.copy(), pc.copy(), masks.copy())
    img2, pc2, masks2 = transform(image.copy(), pc.copy(), masks.copy())
    np.testing.assert_array_equal(img1, img2)
    np.testing.assert_array_equal(pc1, pc2)
    np.testing.assert_array_equal(masks1, masks2)


def test_train_transform_no_nan_inf(dummy_sample, config):
    """outputs of train transform must not contain nan or inf"""
    image, pc, masks = dummy_sample
    transform = TrainTransform(config)
    img_t, pc_t, masks_t = transform(image, pc, masks)
    assert not np.isnan(img_t).any()
    assert not np.isinf(img_t).any()
    assert not np.isnan(pc_t).any()
    assert not np.isinf(pc_t).any()
