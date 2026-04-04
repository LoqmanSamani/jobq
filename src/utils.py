"""
shared helpers and utilities for the entire project
"""
import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linear_sum_assignment, linprog
import onnxruntime as ort
import onnx
from onnx import numpy_helper



# sign pattern for reconstructing 8 cuboid corners from 3 half-edge vectors
CORNER_SIGNS = np.array([
    [-1, -1, -1],  # c0
    [+1, -1, -1],  # c1
    [+1, +1, -1],  # c2
    [-1, +1, -1],  # c3
    [-1, -1, +1],  # c4
    [+1, -1, +1],  # c5
    [+1, +1, +1],  # c6
    [-1, +1, +1],  # c7
], dtype=np.float32)  # (8, 3)

CORNER_SIGNS_TORCH = torch.from_numpy(CORNER_SIGNS)  # (8, 3)


# all 6 permutations of 3 elements (for canonicalization)
_PERMS_3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]


def canonicalize_half_edges(h1, h2, h3):
    """canonicalize 3 half-edge vectors to remove the 48-fold sign/permutation ambiguity"""
    vectors = np.stack([h1, h2, h3])  # (3, 3) or (3, n, 3)
    single = vectors.ndim == 2  # single object vs batch

    if single:
        # for a single (3, 3) matrix: try all 6 permutations
        best_score = -1.0
        best_perm = (0, 1, 2)
        for perm in _PERMS_3:
            score = sum(abs(vectors[perm[i], i]) for i in range(3))
            if score > best_score:
                best_score = score
                best_perm = perm
        ordered = vectors[list(best_perm)]
        for i in range(3):
            if ordered[i, i] < 0:
                ordered[i] = -ordered[i]
        return ordered[0], ordered[1], ordered[2]
    else:
        # batch version: vectors is (3, n, 3)
        n = vectors.shape[1]
        result = np.zeros_like(vectors)
        for obj in range(n):
            v = vectors[:, obj, :]  # (3, 3)
            best_score = -1.0
            best_perm = (0, 1, 2)
            for perm in _PERMS_3:
                score = sum(abs(v[perm[i], i]) for i in range(3))
                if score > best_score:
                    best_score = score
                    best_perm = perm
            ordered = v[list(best_perm)]
            for i in range(3):
                if ordered[i, i] < 0:
                    ordered[i] = -ordered[i]
            result[:, obj, :] = ordered
        return result[0], result[1], result[2]


def reconstruct_corners_np(center, half_edges):
    """reconstruct 8 cuboid corners from center + half-edge vectors (numpy)"""
    H = half_edges.reshape(-1, 3, 3)  # (n, 3, 3): rows are h1, h2, h3
    offsets = np.einsum('ij,njk->nik', CORNER_SIGNS, H)  # (n, 8, 3)
    return center[:, None, :] + offsets


def reconstruct_corners_torch(center, half_edges):
    """reconstruct 8 cuboid corners from center + half-edge vectors (torch)"""
    H = half_edges.reshape(-1, 3, 3)  # (n, 3, 3)
    signs = CORNER_SIGNS_TORCH.to(half_edges.device)  # (8, 3)
    offsets = torch.einsum('ij,njk->nik', signs, H)  # (n, 8, 3)
    return center[:, None, :] + offsets





def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
   
   
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



def get_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """scan data_dir for sample folders and split them into train/val/test sets"""
    
    
    all_ids = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]) # all dirs names

    rng = np.random.RandomState(seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train : n_train + n_val]
    test_ids = all_ids[n_train + n_val :]

    return train_ids, val_ids, test_ids



def gaussian_2d(shape, sigma=1.0):
    """generate a 2d Gaussian kernel(used to 'stamp' object centers on the heatmap)"""
    
    h, w = shape
    y = np.arange(0, h, dtype=np.float32) - (h - 1) / 2.0
    x = np.arange(0, w, dtype=np.float32) - (w - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    # exp(-(x² + y²) / (2σ²)) 
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    
    return kernel


def generate_heatmap_target(masks, output_h, output_w, min_radius=2, sigma_divisor=6.0):
    """generate a center net-style heatmap target from instance masks"""
    
    n_objects = masks.shape[0]
    input_h, input_w = masks.shape[1], masks.shape[2]
    scale_y = output_h / input_h
    scale_x = output_w / input_w

    heatmap = np.zeros((1, output_h, output_w), dtype=np.float32)
    centers_2d = np.zeros((n_objects, 2), dtype=np.float32) 

    for i in range(n_objects):
        ys, xs = np.where(masks[i])
        if len(ys) == 0:
            continue
        cy_input = ys.mean()
        cx_input = xs.mean()

        cy = cy_input * scale_y
        cx = cx_input * scale_x
        centers_2d[i] = [cy, cx]

        obj_h = (ys.max() - ys.min()) * scale_y
        obj_w = (xs.max() - xs.min()) * scale_x
        radius = max(min_radius, int(math.sqrt(obj_h * obj_w) / 2))

        diameter = 2 * radius + 1
        sigma = diameter / sigma_divisor  # default 6σ covers the full diameter
        gaussian = gaussian_2d((diameter, diameter), sigma)

        cy_int = int(round(cy))
        cx_int = int(round(cx))

        top    = max(0, cy_int - radius)
        bottom = min(output_h, cy_int + radius + 1)
        left   = max(0, cx_int - radius)
        right  = min(output_w, cx_int + radius + 1)

        g_top    = max(0, radius - cy_int)
        g_bottom = g_top + (bottom - top)
        g_left   = max(0, radius - cx_int)
        g_right  = g_left + (right - left)

        heatmap[0, top:bottom, left:right] = np.maximum(
            heatmap[0, top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right],
        )

    return heatmap, centers_2d



def _gather_at_centers(pred, centers_2d, num_objects):
    """extract predicted values at GT center locations for each object in the batch"""
    
    B, C, H, W = pred.shape
    gathered = []
    for b in range(B):
        n = num_objects[b].item()
        if n == 0:
            gathered.append(pred.new_zeros(0, C))
            continue
        cy = centers_2d[b, :n, 0].long().clamp(0, H - 1) 
        cx = centers_2d[b, :n, 1].long().clamp(0, W - 1)
        vals = pred[b, :, cy, cx].permute(1, 0)  # (n, C)
        gathered.append(vals)
        
    return gathered



def build_optimizer(model, config):
    """optimizer with selective weight decay (applied only to non-bias, 
    non-normalization params, which stabilizes training)
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=config.learning_rate)


def build_scheduler(optimizer, config):
    """applies two-phase lr scheduling:
          - linear warmup for the fisrst warmup epochs (0 -> init lr)
          - cosine decay for the remaining epochs (init lr -> 0)
    """
    warmup = config.warmup_epochs
    total = config.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    """save training state to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """restore model and optimizer/scheduler from a saved checkpoint and return epoch and val_loss"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))



def corner_error(pred_corners, gt_corners):
    """mean corner error: average l2 distance with optimal corner assignment"""
    cost = np.linalg.norm(pred_corners[:, None] - gt_corners[None, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)
    return float(cost[row_idx, col_idx].mean())



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


def _intersection_volume(hull_a, hull_b):
    """compute the volume of the intersection of two convex hulls in 3d"""
    halfspaces = np.vstack([hull_a.equations, hull_b.equations])
    normals = halfspaces[:, :-1]
    offsets = halfspaces[:, -1]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)

    n_dim = normals.shape[1]
    c = np.zeros(n_dim + 1)
    c[-1] = -1  # maximize r

    A_lp = np.hstack([normals, norms])
    b_lp = -offsets

    bounds = [(None, None)] * n_dim + [(0, None)]
    result = linprog(c, A_ub=A_lp, b_ub=b_lp, bounds=bounds, method="highs")

    if not result.success or result.x[-1] < 1e-10:
        return 0.0  # intersection is empty or degenerate

    interior_point = result.x[:n_dim]

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        inter_hull = ConvexHull(hs.intersections)
        return max(0.0, inter_hull.volume)
    except Exception:
        return 0.0


def iou_3d(pred_corners, gt_corners):
    """intersection-over-union of 3d convex hulls using proper
    halfspace intersection for accurate volume computation
    """
    try:
        hull_pred = ConvexHull(pred_corners)
    except Exception:
        return 0.0
    try:
        hull_gt = ConvexHull(gt_corners)
    except Exception:
        return 0.0

    vol_pred = hull_pred.volume
    vol_gt = hull_gt.volume

    if vol_pred < 1e-10 or vol_gt < 1e-10:
        return 0.0

    inter_vol = _intersection_volume(hull_pred, hull_gt)
    union_vol = vol_pred + vol_gt - inter_vol

    if union_vol < 1e-10:
        return 0.0

    return float(inter_vol / union_vol)



def _box_dimensions(corners):
    """compute box dimensions (l, w, h) from 8 corners, ordering-agnostic"""
    centered = corners - corners.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return np.sort(s / np.sqrt(2.0))


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



def decode_heatmap(heatmap, offset, regression, center_3d, top_k=25, conf_thresh=0.3):
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
        off = offset[b, :, ys.long(), xs.long()]  # (2, n)
        ys_refined = ys + off[0]
        xs_refined = xs + off[1]
        # read regression (half-edge vectors) and center_3d at detected peaks
        reg = regression[b, :, topk_idx // W, topk_idx % W]  # (9, n)
        half_edges = reg.permute(1, 0)  # (n, 9)
        ctr = center_3d[b, :, topk_idx // W, topk_idx % W]  # (3, n)
        ctr = ctr.permute(1, 0)  # (n, 3)
        # reconstruct absolute corners from center + half-edge vectors
        corners = reconstruct_corners_torch(ctr, half_edges)  # (n, 8, 3)
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



def export_to_onnx(model, config, output_path, opset_version=17):
    """export model to ONNX with dynamic batch dimension""" 
    model.eval()
    device = next(model.parameters()).device
    dummy_image = torch.randn(1, 3, config.image_height, config.image_width, device=device)
    dummy_pc = torch.randn(1, 3, config.image_height, config.image_width, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_pc),
        output_path,
        opset_version=opset_version,
        input_names=["image", "point_cloud"],
        output_names=["heatmap", "offset", "regression", "center_3d"],
        dynamic_axes={
            "image": {0: "batch"},
            "point_cloud": {0: "batch"},
            "heatmap": {0: "batch"},
            "offset": {0: "batch"},
            "regression": {0: "batch"},
            "center_3d": {0: "batch"},
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
    dummy_image = torch.randn(1, 3, config.image_height, config.image_width, device=device)
    dummy_pc = torch.randn(1, 3, config.image_height, config.image_width, device=device)

    # pytorch forward
    with torch.no_grad():
        pt_out = model(dummy_image, point_cloud=dummy_pc)

    # onnx forward
    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {
        "image": dummy_image.cpu().numpy(),
        "point_cloud": dummy_pc.cpu().numpy(),
    })

    names = ["heatmap", "offset", "regression", "center_3d"]
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
    """export model to FP16 ONNX for faster inference on gpu"""
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
