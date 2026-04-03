"""
inference pipeline and model export:
    - predictor class for loading checkpoint and running inference on images
    - heatmap decoding to extract detections from model outputs
    - optional 3D NMS post-processing
    - ONNX export for deployment
    - validation of ONNX export against PyTorch outputs
    - FP16 ONNX export for faster GPU inference
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
import onnx
from onnx import numpy_helper

from src.config import Config
from src.model import BBox3DNet
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluator import iou_3d




def decode_heatmap(heatmap, offset, regression, top_k=25, conf_thresh=0.3):
    """extract detections from raw model output maps"""
    
    heatmap = heatmap.sigmoid()
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * (heatmap == hmax).float()

    B, _, H, W = heatmap.shape
    results = []
    
    for b in range(B):
        scores_flat = heatmap[b, 0].reshape(-1)
        k = min(top_k, scores_flat.numel())
        topk_scores, topk_idx = scores_flat.topk(k)
        mask = topk_scores >= conf_thresh
        topk_scores = topk_scores[mask]
        topk_idx = topk_idx[mask]
        if topk_idx.numel() == 0:
            results.append({
                "corners": np.zeros((0, 8, 3), dtype=np.float32),
                "scores": np.zeros(0, dtype=np.float32),
                "centers_hm": np.zeros((0, 2), dtype=np.float32),
            })
            continue

        ys = (topk_idx // W).float()
        xs = (topk_idx % W).float()
        # apply sub-pixel offsets
        off = offset[b, :, ys.long(), xs.long()]  # (2, N)
        ys_refined = ys + off[0]
        xs_refined = xs + off[1]
        # read regression values for detected peaks
        reg = regression[b, :, topk_idx // W, topk_idx % W] 
        corners = reg.permute(1, 0).reshape(-1, 8, 3) 
        centers_hm = torch.stack([ys_refined, xs_refined], dim=1)
        results.append({
            "corners": corners.detach().cpu().numpy(),
            "scores": topk_scores.detach().cpu().numpy(),
            "centers_hm": centers_hm.detach().cpu().numpy(),
        })

    return results



def nms_3d(corners, scores, iou_threshold=0.25):
    """greedy 3d nms using convex-hull iou from evaluator.iou_3d"""
    
    if len(scores) == 0:
        return []

    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if j == i or suppressed[j]:
                continue
            iou = iou_3d(corners[i], corners[j])
            if iou > iou_threshold:
                suppressed[j] = True

    return keep



class Predictor:
    """loads a trained checkpoint, preprocesses images, and runs prediction"""
    
    def __init__(self, checkpoint_path, config=None, device=None):
        self.config = config or Config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build model and load weights
        self.model = BBox3DNet(self.config, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_path):
        """load and preprocess a single rgb image"""
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize to model input size
        h, w = self.config.image_height, self.config.image_width
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # normalize (imagenet stats)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device) # (1, 3, h, w)
        
        return tensor

    def predict(self, image_tensor, top_k=25, conf_thresh=0.3, apply_nms=True, nms_iou=0.25):
        """run model on a preprocessed image tensor and decode detections"""
        
        with torch.no_grad():
            preds = self.model(image_tensor)

        decoded = decode_heatmap(
            preds["heatmap"], preds["offset"], preds["regression"],
            top_k=top_k, conf_thresh=conf_thresh,
        )

        if apply_nms:
            for det in decoded:
                if len(det["scores"]) > 0:
                    keep = nms_3d(det["corners"], det["scores"], iou_threshold=nms_iou)
                    det["corners"] = det["corners"][keep]
                    det["scores"] = det["scores"][keep]
                    det["centers_hm"] = det["centers_hm"][keep]

        return decoded

    def predict_image(self, image_path, top_k=25, conf_thresh=0.3, apply_nms=True, nms_iou=0.25):
        """load image from path, preprocess, and predict"""
        
        tensor = self.preprocess(image_path)
        results = self.predict(tensor, top_k, conf_thresh, apply_nms, nms_iou)
        
        return results[0]  # single image

    def predict_batch(self, image_paths, top_k=25, conf_thresh=0.3, apply_nms=True, nms_iou=0.25):
        """run inference on a list of image paths"""
        
        tensors = []
        for path in image_paths:
            tensors.append(self.preprocess(path))
        batch = torch.cat(tensors, dim=0)  # (b, 3, h, w)
        
        return self.predict(batch, top_k, conf_thresh, apply_nms, nms_iou)



def export_to_onnx(model, config, output_path, opset_version=17):
    """export model to ONNX with dynamic batch dimension"""
    
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, config.image_height, config.image_width, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["heatmap", "offset", "regression"],
        dynamic_axes={
            "image": {0: "batch"},
            "heatmap": {0: "batch"},
            "offset": {0: "batch"},
            "regression": {0: "batch"},
        },
    )
    return output_path


def validate_onnx(onnx_path, model, config, atol=1e-4):
    """
    load ONNX model with onnxruntime, run same input through both
    pytorch and onnx, and compare outputs numerically
    """
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, config.image_height, config.image_width, device=device)

    # pytorch forward
    with torch.no_grad():
        pt_out = model(dummy)

    # onnx forward
    session = ort.InferenceSession(onnx_path)
    dummy_np = dummy.cpu().numpy()
    ort_out = session.run(None, {"image": dummy_np})

    names = ["heatmap", "offset", "regression"]
    diffs = {}
    all_match = True
    for name, ort_val in zip(names, ort_out):
        pt_val = pt_out[name].cpu().numpy()
        max_diff = float(np.max(np.abs(pt_val - ort_val)))
        diffs[name] = max_diff
        if max_diff > atol:
            all_match = False

    return {"matches": all_match, "max_diff": diffs}



def export_to_fp16_onnx(model, config, output_path, opset_version=17):
    """export model to FP16 ONNX for faster inference on GPU"""
    
    # first export fp32 to a temp path
    fp32_path = output_path.replace(".onnx", "_fp32_tmp.onnx")
    export_to_onnx(model, config, fp32_path, opset_version)

    # load and convert to fp16
    onnx_model = onnx.load(fp32_path)
    for initializer in onnx_model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(initializer).astype(np.float16)
            new_init = numpy_helper.from_array(arr, name=initializer.name)
            initializer.CopyFrom(new_init)

    # update graph input/output types
    for value_info in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT:
            tensor_type.elem_type = onnx.TensorProto.FLOAT16

    onnx.save(onnx_model, output_path)

    # clean up temp file
    if os.path.exists(fp32_path):
        os.remove(fp32_path)

    return output_path
