import torch
import pytest
from src.config import Config
from src.model import Backbone, FPNNeck, DetectionHead, BBox3DNet
from src.utils import count_parameters





@pytest.fixture
def config():
    return Config()


@pytest.fixture
def model(config):
    return BBox3DNet(config, pretrained=False)


@pytest.fixture
def dummy_input(config):
    return torch.randn(2, 3, config.image_height, config.image_width)


def test_backbone_output_count():
    """backbone should return 4 feature levels"""
    backbone = Backbone(pretrained=False)
    x = torch.randn(1, 3, 256, 384)
    feats = backbone(x)
    assert len(feats) == 4


def test_backbone_output_shapes():
    """each backbone feature level should have the expected spatial size"""
    backbone = Backbone(pretrained=False)
    x = torch.randn(1, 3, 256, 384)
    feats = backbone(x)
    expected_shapes = [
        (1, 64, 64, 96),
        (1, 128, 32, 48),
        (1, 256, 16, 24),
        (1, 512, 8, 12),
    ]
    for feat, expected in zip(feats, expected_shapes):
        assert feat.shape == expected, f"Expected {expected}, got {feat.shape}"


def test_backbone_channels_attribute():
    """backbone.channels should list [64, 128, 256, 512] for resnet18"""
    backbone = Backbone(pretrained=False)
    assert backbone.channels == [64, 128, 256, 512]


def test_fpn_output_shape():
    """fpn should produce a single feature map at stride-4 resolution"""
    fpn = FPNNeck([64, 128, 256, 512], out_channels=64)
    features = [
        torch.randn(2, 64, 64, 96),
        torch.randn(2, 128, 32, 48),
        torch.randn(2, 256, 16, 24),
        torch.randn(2, 512, 8, 12),
    ]
    out = fpn(features)
    assert out.shape == (2, 64, 64, 96)


def test_fpn_different_out_channels():
    """fpn with non-default out_channels should produce the right channel count"""
    fpn = FPNNeck([64, 128, 256, 512], out_channels=128)
    features = [
        torch.randn(1, 64, 64, 96),
        torch.randn(1, 128, 32, 48),
        torch.randn(1, 256, 16, 24),
        torch.randn(1, 512, 8, 12),
    ]
    out = fpn(features)
    assert out.shape[1] == 128


def test_detection_head_output_shape():
    """detection head should preserve spatial dims and produce out_channels"""
    head = DetectionHead(64, out_channels=24)
    x = torch.randn(2, 64, 64, 96)
    out = head(x)
    assert out.shape == (2, 24, 64, 96)


def test_heatmap_head_init_bias():
    """heatmap head bias should be initialized to the specified value"""
    head = DetectionHead(64, out_channels=1, init_bias=-2.19)
    bias_val = head.out_conv.bias.item()
    assert abs(bias_val - (-2.19)) < 1e-4


def test_model_instantiation(config):
    """bbox3dnet should instantiate without errors"""
    model = BBox3DNet(config, pretrained=False)
    assert model is not None


def test_model_parameter_count(model):
    """bbox3dnet should have a reasonable number of parameters for resnet18 + heads"""
    total, trainable = count_parameters(model)
    assert 10_000_000 < total < 20_000_000, f"Unexpected param count: {total}"
    assert total == trainable


def test_forward_output_keys(model, dummy_input):
    """forward pass should return a dict with heatmap, offset, regression"""
    preds = model(dummy_input)
    assert set(preds.keys()) == {"heatmap", "offset", "regression"}


def test_forward_output_shapes(model, dummy_input, config):
    """all prediction tensors should have the expected shapes"""
    preds = model(dummy_input)
    B = 2
    oH = config.image_height // config.output_stride  # 64
    oW = config.image_width // config.output_stride    # 96
    assert preds["heatmap"].shape == (B, 1, oH, oW)
    assert preds["offset"].shape == (B, 2, oH, oW)
    assert preds["regression"].shape == (B, 24, oH, oW)


def test_forward_no_nan(model, dummy_input):
    """no nan or inf in any output tensor"""
    preds = model(dummy_input)
    for key, tensor in preds.items():
        assert not torch.isnan(tensor).any(), f"NaN in {key}"
        assert not torch.isinf(tensor).any(), f"Inf in {key}"


def test_forward_with_point_cloud(model, config):
    """forward pass with point_cloud arg should not break the forward pass"""
    B = 2
    image = torch.randn(B, 3, config.image_height, config.image_width)
    pc = torch.randn(B, 3, config.image_height, config.image_width)
    preds = model(image, point_cloud=pc)
    assert preds["heatmap"].shape[1] == 1
    assert preds["regression"].shape[1] == 24


def test_backward_pass(model, dummy_input):
    """gradients should flow to all parameters"""
    preds = model(dummy_input)
    loss = sum(p.sum() for p in preds.values())
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_model_on_cpu(config):
    """model should work on cpu (no cuda required for testing)"""
    model = BBox3DNet(config, pretrained=False).cpu()
    x = torch.randn(1, 3, config.image_height, config.image_width).cpu()
    preds = model(x)
    assert preds["heatmap"].device.type == "cpu"


def test_batch_size_one(model, config):
    """forward pass with batch_size=1 should work (edge case)"""
    x = torch.randn(1, 3, config.image_height, config.image_width)
    preds = model(x)
    assert preds["heatmap"].shape[0] == 1


def test_pretrained_weights_nonzero():
    """pretrained backbone should have non-zero weights (not random init)"""
    model = BBox3DNet(Config(), pretrained=True)
    first_conv = None
    for module in model.backbone.net.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    assert first_conv is not None
    assert first_conv.weight.abs().sum() > 0
    assert first_conv.weight.std() > 0.001
