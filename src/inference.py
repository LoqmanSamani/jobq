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
from src.config import Config
from src.model import BBox3DNet
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils import decode_heatmap, nms_3d, export_to_onnx, validate_onnx, export_to_fp16_onnx




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
