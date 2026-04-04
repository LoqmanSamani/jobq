"""
inference pipeline and model export:
    - predictor class for loading checkpoint and running inference on samples (image + point cloud)
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
from src.config import Config
from src.model import BBox3DNet
from src.transforms import IMAGENET_MEAN, IMAGENET_STD, normalize_point_cloud
from src.utils import decode_heatmap, nms_3d, export_to_onnx, validate_onnx, export_to_fp16_onnx




class Predictor:
    """loads a trained checkpoint, preprocesses samples (image + point cloud), and runs prediction"""
    def __init__(self, checkpoint_path, config=None, device=None):
        self.config = config or Config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build model and load weights
        self.model = BBox3DNet(self.config, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, sample_dir):
        """load and preprocess image and point cloud from a sample directory"""
        h, w = self.config.image_height, self.config.image_width
        
        # load and preprocess image
        img_path = os.path.join(sample_dir, "rgb.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device) # (1, 3, h, w)

        # load and preprocess point cloud
        pc_path = os.path.join(sample_dir, "pc.npy")
        pc = np.load(pc_path)
        pc_hwc = pc.transpose(1, 2, 0)
        pc_hwc = cv2.resize(pc_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        pc = pc_hwc.transpose(2, 0, 1)
        pc = normalize_point_cloud(pc)
        pc_tensor = torch.from_numpy(pc).unsqueeze(0).to(self.device) # (1, 3, h, w)
        
        return image_tensor, pc_tensor

    def predict(self, image_tensor, pc_tensor, top_k=None, conf_thresh=None, apply_nms=True, nms_iou=None):
        """run model on preprocessed image and point cloud tensors and decode detections"""
        top_k = top_k if top_k is not None else self.config.top_k
        conf_thresh = conf_thresh if conf_thresh is not None else self.config.conf_thresh
        nms_iou = nms_iou if nms_iou is not None else self.config.nms_iou_thresh
        
        with torch.no_grad():
            preds = self.model(image_tensor, point_cloud=pc_tensor)

        decoded = decode_heatmap(
            preds["heatmap"], preds["offset"], preds["regression"],
            preds["center_3d"],
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

    def predict_sample(self, sample_dir, top_k=None, conf_thresh=None, apply_nms=True, nms_iou=None):
        """load sample from directory, preprocess, and predict"""    
        image_tensor, pc_tensor = self.preprocess(sample_dir)
        results = self.predict(image_tensor, pc_tensor, top_k, conf_thresh, apply_nms, nms_iou)
        
        return results[0]  # single sample

    def predict_batch(self, sample_dirs, top_k=None, conf_thresh=None, apply_nms=True, nms_iou=None):
        """run inference on a list of sample directories"""    
        image_tensors = []
        pc_tensors = []
        for sample_dir in sample_dirs:
            img, pc = self.preprocess(sample_dir)
            image_tensors.append(img)
            pc_tensors.append(pc)
        image_batch = torch.cat(image_tensors, dim=0)  # (b, 3, h, w)
        pc_batch = torch.cat(pc_tensors, dim=0)  # (b, 3, h, w)
        
        return self.predict(image_batch, pc_batch, top_k, conf_thresh, apply_nms, nms_iou)
