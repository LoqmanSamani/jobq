"""
evaluation metrics:
    - mean corner error: average l2 distance across all 8 corners
    - mean center error: l2 distance between bbox centers (mean of 8 corners)
    - 3d iou: intersection-over-union of 3d convex hulls
    - precision / recall at center-distance thresholds
    - size error: l1 error on bbox dimensions (l, w, h)
"""
import torch
import numpy as np
from src.utils import (
    corner_error,
    center_error,
    iou_3d,
    size_error,
    match_predictions_to_gt,
    precision_recall,
    evaluate_sample,
)




class Evaluator:
    """runs the model over a DataLoader, decodes predictions,
       matches them to ground truth, and aggregates metrics.
    """
    def __init__(self, model, loader, config, device, top_k=25, conf_thresh=0.3, max_dist=0.5):
        self.model = model
        self.loader = loader
        self.config = config
        self.device = device
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.max_dist = max_dist

    @staticmethod
    def _nms_heatmap(heatmap, kernel=3):
        """simple max pooling NMS for heatmap peaks"""
        pad = (kernel - 1) // 2
        import torch.nn.functional as F
        hmax = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)
        return heatmap * (heatmap == hmax).float()
    

    def _decode_predictions(self, preds):
        """extract detections from dense prediction maps"""
        heatmap = preds["heatmap"].sigmoid()
        heatmap = self._nms_heatmap(heatmap)

        B, _, H, W = heatmap.shape
        all_boxes = []
        all_scores = []

        for b in range(B):
            scores_flat = heatmap[b, 0].reshape(-1) 
            k = min(self.top_k, scores_flat.numel())
            topk_scores, topk_idx = scores_flat.topk(k)
            mask = topk_scores >= self.conf_thresh
            topk_scores = topk_scores[mask]
            topk_idx = topk_idx[mask]

            if topk_idx.numel() == 0:
                all_boxes.append(np.zeros((0, 8, 3), dtype=np.float32))
                all_scores.append(np.zeros(0, dtype=np.float32))
                continue

            ys = (topk_idx // W).float()
            xs = (topk_idx % W).float()

            offsets = preds["offset"][b, :, ys.long(), xs.long()]  
            ys = ys + offsets[0]
            xs = xs + offsets[1]

            reg = preds["regression"][b, :, topk_idx // W, topk_idx % W] 
            corners = reg.permute(1, 0).reshape(-1, 8, 3) 

            all_boxes.append(corners.detach().cpu().numpy())
            all_scores.append(topk_scores.detach().cpu().numpy())

        return all_boxes, all_scores

    def evaluate(self):
        """run evaluation over the full loader"""
        self.model.eval()
        per_sample = []
        with torch.no_grad():
            for batch in self.loader:
                image = batch["image"].to(self.device)
                preds = self.model(image)
                pred_boxes_list, pred_scores_list = self._decode_predictions(preds)
                gt_bbox3d = batch["bbox3d"].numpy()
                num_objects = batch["num_objects"].numpy()
                B = image.shape[0]
                for b in range(B):
                    n = int(num_objects[b])
                    gt_boxes = gt_bbox3d[b, :n]  
                    pred_boxes = pred_boxes_list[b]
                    sample_result = evaluate_sample(
                        pred_boxes, gt_boxes, max_dist=self.max_dist
                    )
                    per_sample.append(sample_result)

        all_corner = [e for s in per_sample for e in s["corner_errors"]]
        all_center = [e for s in per_sample for e in s["center_errors"]]
        all_iou = [e for s in per_sample for e in s["ious"]]
        all_size = [e for s in per_sample for e in s["size_errors"]]
        total_tp = sum(s["n_matches"] for s in per_sample)
        total_fp = sum(s["n_false_positives"] for s in per_sample)
        total_fn = sum(s["n_false_negatives"] for s in per_sample)

        summary = {
            "mean_corner_error": float(np.mean(all_corner)) if all_corner else float("inf"),
            "mean_center_error": float(np.mean(all_center)) if all_center else float("inf"),
            "mean_iou_3d": float(np.mean(all_iou)) if all_iou else 0.0,
            "mean_size_error": float(np.mean(all_size)) if all_size else float("inf"),
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_tp + total_fn, 1),
            "n_samples": len(per_sample),
        }

        return summary, per_sample
