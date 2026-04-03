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
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment



def corner_error(pred_corners, gt_corners):
    """mean corner error: average l2 distance across all 8 corners"""
    dists = np.linalg.norm(pred_corners - gt_corners, axis=1)
    return float(dists.mean())



def center_error(pred_corners, gt_corners):
    """mean center error: l2 distance between bbox centers (mean of 8 corners)"""
    pred_center = pred_corners.mean(axis=0)  # (3,)
    gt_center = gt_corners.mean(axis=0)
    return float(np.linalg.norm(pred_center - gt_center))


def _convex_hull_volume(points):
    """compute volume of convex hull of a set of 3D points"""
    if len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def iou_3d(pred_corners, gt_corners):
    """intersection-over-union of 3d convex hulls"""
    vol_pred = _convex_hull_volume(pred_corners)
    vol_gt = _convex_hull_volume(gt_corners)

    if vol_pred < 1e-10 or vol_gt < 1e-10:
        return 0.0

    all_points = np.vstack([pred_corners, gt_corners]) 
    vol_union_hull = _convex_hull_volume(all_points)
    if vol_union_hull < 1e-10:
        return 0.0

    intersection = vol_pred + vol_gt - vol_union_hull
    intersection = max(0.0, intersection) 

    union = vol_pred + vol_gt - intersection
    if union < 1e-10:
        return 0.0

    return float(intersection / union)



def _box_dimensions(corners):
    """compute box dimensions (l, w, h) from 8 corners"""
    edge_groups = [
        [(0, 1), (3, 2), (4, 5), (7, 6)], 
        [(0, 3), (1, 2), (4, 7), (5, 6)],  
        [(0, 4), (1, 5), (2, 6), (3, 7)], 
    ]
    dims = []
    for group in edge_groups:
        lengths = [np.linalg.norm(corners[i] - corners[j]) for i, j in group]
        dims.append(float(np.mean(lengths)))
    return np.array(dims) 


def size_error(pred_corners, gt_corners):
    """l1 error on bbox dimensions (l, w, h)"""
    pred_dims = _box_dimensions(pred_corners)
    gt_dims = _box_dimensions(gt_corners)
    return float(np.abs(pred_dims - gt_dims).mean())



def match_predictions_to_gt(pred_boxes, gt_boxes, max_dist=0.5):
    """match predicted boxes to ground-truth using the Hungarian algorithm"""
    M = pred_boxes.shape[0]
    N = gt_boxes.shape[0]

    if M == 0 and N == 0:
        return [], [], []
    if M == 0:
        return [], [], list(range(N))
    if N == 0:
        return [], list(range(M)), []

    pred_centers = pred_boxes.mean(axis=1) 
    gt_centers = gt_boxes.mean(axis=1)     
    cost = np.linalg.norm(pred_centers[:, None] - gt_centers[None, :], axis=2)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    matched_preds = set()
    matched_gts = set()

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_dist:
            matches.append((int(r), int(c)))
            matched_preds.add(r)
            matched_gts.add(c)

    false_positives = [i for i in range(M) if i not in matched_preds]
    false_negatives = [j for j in range(N) if j not in matched_gts]

    return matches, false_positives, false_negatives



def precision_recall(pred_boxes, gt_boxes, thresholds=(0.1, 0.2, 0.3, 0.5)):
    """compute precision and recall at multiple center-distance thresholds"""
    results = {}
    for t in thresholds:
        matches, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, max_dist=t)
        tp = len(matches)
        precision = tp / (tp + len(fp)) if (tp + len(fp)) > 0 else 0.0
        recall = tp / (tp + len(fn)) if (tp + len(fn)) > 0 else 0.0
        results[t] = {"precision": precision, "recall": recall}
    return results



def evaluate_sample(pred_boxes, gt_boxes, max_dist=0.5):
    """run all five metrics on one sample"""
    matches, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, max_dist)

    corner_errors = []
    center_errors = []
    ious = []
    size_errors = []

    for pi, gi in matches:
        corner_errors.append(corner_error(pred_boxes[pi], gt_boxes[gi]))
        center_errors.append(center_error(pred_boxes[pi], gt_boxes[gi]))
        ious.append(iou_3d(pred_boxes[pi], gt_boxes[gi]))
        size_errors.append(size_error(pred_boxes[pi], gt_boxes[gi]))

    pr = precision_recall(pred_boxes, gt_boxes)

    return {
        "corner_errors": corner_errors,
        "center_errors": center_errors,
        "ious": ious,
        "size_errors": size_errors,
        "n_matches": len(matches),
        "n_false_positives": len(fp),
        "n_false_negatives": len(fn),
        "precision_recall": pr,
    }


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
