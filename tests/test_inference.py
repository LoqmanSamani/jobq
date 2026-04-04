import os
import tempfile
import numpy as np
import torch
import pytest
import cv2
from src.config import Config
from src.model import BBox3DNet
from src.inference import Predictor
from src.utils import decode_heatmap, nms_3d, export_to_onnx, validate_onnx, export_to_fp16_onnx





@pytest.fixture
def config():
    return Config()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model(config, device):
    m = BBox3DNet(config, pretrained=False)
    m.to(device).eval()
    return m


@pytest.fixture
def dummy_input(config, device):
    """single-image batch tensor"""
    return torch.randn(1, 3, config.image_height, config.image_width, device=device)


@pytest.fixture
def dummy_pc(config, device):
    """single point cloud batch tensor"""
    return torch.randn(1, 3, config.image_height, config.image_width, device=device)


@pytest.fixture
def dummy_preds(model, dummy_input, dummy_pc):
    """raw model output dict"""
    with torch.no_grad():
        return model(dummy_input, point_cloud=dummy_pc)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def fake_checkpoint(model, tmp_dir, config):
    """save a minimal checkpoint so Predictor can load it"""
    path = os.path.join(tmp_dir, "test_ckpt.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


@pytest.fixture
def fake_image(tmp_dir, config):
    """create a dummy RGB image on disk"""
    path = os.path.join(tmp_dir, "test_img.jpg")
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def fake_sample_dir(tmp_dir, config):
    """create a fake sample directory with rgb.jpg and pc.npy"""
    sample_dir = os.path.join(tmp_dir, "fake_sample")
    os.makedirs(sample_dir, exist_ok=True)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(sample_dir, "rgb.jpg"), img)
    pc = np.random.randn(3, 480, 640).astype(np.float64)
    np.save(os.path.join(sample_dir, "pc.npy"), pc)
    return sample_dir


class TestDecodeHeatmap:
    def test_output_structure(self, dummy_preds):
        results = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
        )
        assert isinstance(results, list)
        assert len(results) == 1  # batch of 1
        det = results[0]
        assert "corners" in det
        assert "scores" in det
        assert "centers_hm" in det

    def test_shapes(self, dummy_preds):
        results = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
        )
        det = results[0]
        N = len(det["scores"])
        assert det["corners"].shape == (N, 8, 3)
        assert det["scores"].shape == (N,)
        assert det["centers_hm"].shape == (N, 2)

    def test_scores_in_range(self, dummy_preds):
        results = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
            conf_thresh=0.0,
        )
        det = results[0]
        assert (det["scores"] >= 0.0).all()
        assert (det["scores"] <= 1.0).all()

    def test_top_k_limits_output(self, dummy_preds):
        results = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
            top_k=3, conf_thresh=0.0,
        )
        assert len(results[0]["scores"]) <= 3

    def test_high_threshold_gives_fewer(self, dummy_preds):
        low = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
            conf_thresh=0.0,
        )
        high = decode_heatmap(
            dummy_preds["heatmap"], dummy_preds["offset"], dummy_preds["regression"],
            dummy_preds["center_3d"],
            conf_thresh=0.9,
        )
        assert len(high[0]["scores"]) <= len(low[0]["scores"])

    def test_synthetic_peaks(self):
        """plant known peaks in a heatmap and verify they are detected"""
        H, W = 64, 96
        heatmap = torch.full((1, 1, H, W), -10.0)  # all very low
        offset = torch.zeros(1, 2, H, W)
        regression = torch.randn(1, 9, H, W)
        center_3d = torch.randn(1, 3, H, W)

        # plant two high-confidence peaks at known locations
        heatmap[0, 0, 10, 20] = 5.0  # sigmoid ≈ 0.993
        heatmap[0, 0, 50, 80] = 4.0  # sigmoid ≈ 0.982

        results = decode_heatmap(heatmap, offset, regression, center_3d, top_k=10, conf_thresh=0.5)
        det = results[0]
        assert len(det["scores"]) == 2

        centers = det["centers_hm"]
        planted = np.array([[10.0, 20.0], [50.0, 80.0]])
        # sort both by y coordinate
        centers_sorted = centers[np.argsort(centers[:, 0])]
        planted_sorted = planted[np.argsort(planted[:, 0])]
        np.testing.assert_allclose(centers_sorted, planted_sorted, atol=1.0)

    def test_batch_multiple_images(self, model, config, device):
        """decode works with batch_size > 1"""
        batch = torch.randn(3, 3, config.image_height, config.image_width, device=device)
        pc = torch.randn(3, 3, config.image_height, config.image_width, device=device)
        with torch.no_grad():
            preds = model(batch, point_cloud=pc)
        results = decode_heatmap(
            preds["heatmap"], preds["offset"], preds["regression"],
            preds["center_3d"],
            conf_thresh=0.0,
        )
        assert len(results) == 3

    def test_empty_after_threshold(self):
        """all scores below threshold returns empty detections"""
        H, W = 64, 96
        heatmap = torch.full((1, 1, H, W), -10.0)  # sigmoid ≈ 0.0
        offset = torch.zeros(1, 2, H, W)
        regression = torch.zeros(1, 9, H, W)
        center_3d = torch.zeros(1, 3, H, W)
        results = decode_heatmap(heatmap, offset, regression, center_3d, conf_thresh=0.5)
        det = results[0]
        assert len(det["scores"]) == 0
        assert det["corners"].shape == (0, 8, 3)


class TestNms3D:
    def test_empty_input(self):
        keep = nms_3d(np.zeros((0, 8, 3)), np.zeros(0), iou_threshold=0.25)
        assert keep == []

    def test_single_detection(self):
        corners = np.random.randn(1, 8, 3).astype(np.float32)
        scores = np.array([0.9])
        keep = nms_3d(corners, scores, iou_threshold=0.25)
        assert keep == [0]

    def test_identical_suppressed(self):
        """two identical boxes — one should be suppressed"""
        box = np.random.randn(8, 3).astype(np.float32) * 2
        corners = np.stack([box, box])
        scores = np.array([0.9, 0.7])
        keep = nms_3d(corners, scores, iou_threshold=0.25)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept

    def test_non_overlapping_kept(self):
        """two far-apart boxes should both be kept"""
        box1 = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        box2 = box1 + 100.0  # far away
        corners = np.stack([box1, box2])
        scores = np.array([0.9, 0.8])
        keep = nms_3d(corners, scores, iou_threshold=0.25)
        assert len(keep) == 2

    def test_score_ordering(self):
        """highest score is always kept first"""
        corners = np.random.randn(5, 8, 3).astype(np.float32) * 100  # spread out
        scores = np.array([0.3, 0.9, 0.5, 0.1, 0.7])
        keep = nms_3d(corners, scores, iou_threshold=0.01)
        assert keep[0] == 1  # index of max score


class TestPredictor:
    def test_init(self, fake_checkpoint, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        assert predictor.model is not None
        assert not predictor.model.training  # eval mode

    def test_preprocess_shapes(self, fake_checkpoint, fake_sample_dir, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        image_tensor, pc_tensor = predictor.preprocess(fake_sample_dir)
        assert image_tensor.shape == (1, 3, config.image_height, config.image_width)
        assert pc_tensor.shape == (1, 3, config.image_height, config.image_width)
        assert image_tensor.dtype == torch.float32
        assert pc_tensor.dtype == torch.float32

    def test_preprocess_missing_dir(self, fake_checkpoint, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        with pytest.raises(FileNotFoundError):
            predictor.preprocess("/nonexistent/sample_dir")

    def test_predict_returns_detections(self, fake_checkpoint, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        image_tensor = torch.randn(1, 3, config.image_height, config.image_width, device=device)
        pc_tensor = torch.randn(1, 3, config.image_height, config.image_width, device=device)
        results = predictor.predict(image_tensor, pc_tensor, conf_thresh=0.0, apply_nms=False)
        assert len(results) == 1
        det = results[0]
        assert "corners" in det and "scores" in det

    def test_predict_sample(self, fake_checkpoint, fake_sample_dir, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        det = predictor.predict_sample(fake_sample_dir, conf_thresh=0.0, apply_nms=False)
        assert "corners" in det
        assert "scores" in det
        assert det["corners"].ndim == 3 and det["corners"].shape[1:] == (8, 3)

    def test_predict_batch(self, fake_checkpoint, fake_sample_dir, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        results = predictor.predict_batch(
            [fake_sample_dir, fake_sample_dir], conf_thresh=0.0, apply_nms=False,
        )
        assert len(results) == 2

    def test_predict_with_nms(self, fake_checkpoint, config, device):
        predictor = Predictor(fake_checkpoint, config, device)
        image_tensor = torch.randn(1, 3, config.image_height, config.image_width, device=device)
        pc_tensor = torch.randn(1, 3, config.image_height, config.image_width, device=device)
        # with NMS enabled (default)
        results = predictor.predict(image_tensor, pc_tensor, conf_thresh=0.0, apply_nms=True)
        assert len(results) == 1


class TestOnnxExport:
    def test_export_creates_file(self, model, config, tmp_dir):
        path = os.path.join(tmp_dir, "model.onnx")
        export_to_onnx(model, config, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_onnx_output_matches_pytorch(self, model, config, tmp_dir):
        path = os.path.join(tmp_dir, "model.onnx")
        export_to_onnx(model, config, path)
        result = validate_onnx(path, model, config, atol=1e-4)
        assert result["matches"], f"max diffs: {result['max_diff']}"

    def test_onnx_dynamic_batch(self, model, config, tmp_dir):
        """exported ONNX should accept batch_size > 1"""
        import onnxruntime as ort

        path = os.path.join(tmp_dir, "model.onnx")
        export_to_onnx(model, config, path)

        session = ort.InferenceSession(path)
        dummy_image = np.random.randn(3, 3, config.image_height, config.image_width).astype(np.float32)
        dummy_pc = np.random.randn(3, 3, config.image_height, config.image_width).astype(np.float32)
        outputs = session.run(None, {"image": dummy_image, "point_cloud": dummy_pc})
        assert outputs[0].shape[0] == 3  # heatmap batch dim
        assert outputs[1].shape[0] == 3  # offset batch dim
        assert outputs[2].shape[0] == 3  # regression batch dim
        assert outputs[3].shape[0] == 3  # center_3d batch dim


class TestFp16Export:
    def test_fp16_creates_file(self, model, config, tmp_dir):
        path = os.path.join(tmp_dir, "model_fp16.onnx")
        export_to_fp16_onnx(model, config, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_fp16_has_float16_tensors(self, model, config, tmp_dir):
        """fp16 model should contain float16 typed tensors"""
        import onnx
        fp16_path = os.path.join(tmp_dir, "model_fp16.onnx")
        export_to_fp16_onnx(model, config, fp16_path)
        onnx_model = onnx.load(fp16_path)
        # at least some initializers should be float16
        fp16_count = sum(
            1 for init in onnx_model.graph.initializer
            if init.data_type == onnx.TensorProto.FLOAT16
        )
        assert fp16_count > 0, "no float16 initializers found"

    def test_fp16_no_temp_leftovers(self, model, config, tmp_dir):
        """temp fp32 file should be cleaned up"""
        path = os.path.join(tmp_dir, "model_fp16.onnx")
        export_to_fp16_onnx(model, config, path)
        tmp_fp32 = path.replace(".onnx", "_fp32_tmp.onnx")
        assert not os.path.exists(tmp_fp32)
