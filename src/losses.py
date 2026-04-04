"""
losses:
    - heatmap focal loss (center-net style)
    - offset l1 loss (sub-pixel center refinement) 
    - half-edge L1 loss (3 half-edge vectors x 3 coords = 9 values)
    - center 3d loss (L1 on 3D object center)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import _gather_at_centers



class HeatmapFocalLoss(nn.Module):
    """modified focal loss"""
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta 

    def forward(self, pred, gt):
       
        pred = pred.sigmoid()
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)
        pos_mask = gt.eq(1.0).float() 
        neg_mask = 1.0 - pos_mask
        
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = (-((1 - gt) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask)
        
        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss


class OffsetL1Loss(nn.Module):
    """l1 loss on predicted sub-pixel offsets at GT center locations"""
    def forward(self, pred_offset, centers_2d, num_objects):
        
        gathered = _gather_at_centers(pred_offset, centers_2d, num_objects)
        total_loss = pred_offset.new_tensor(0.0)
        total_count = 0
        B = pred_offset.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_vals = gathered[b]
            gt_offset = centers_2d[b, :n, :] - centers_2d[b, :n, :].floor()
            total_loss = total_loss + F.l1_loss(pred_vals, gt_offset, reduction="sum")
            total_count += n
        
        if total_count == 0:
            return total_loss
        
        return total_loss / total_count


class BBox3DCornerLoss(nn.Module):
    """smooth-l1 loss on the predicted 3D bounding box corners"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_reg, bbox3d_gt, centers_2d, num_objects):
       
        gathered = _gather_at_centers(pred_reg, centers_2d, num_objects)
        total_loss = pred_reg.new_tensor(0.0)
        total_count = 0
        
        B = pred_reg.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_corners = gathered[b].reshape(n, 8, 3)
            gt_corners = bbox3d_gt[b, :n]  # (n, 8, 3)
            total_loss = total_loss + F.smooth_l1_loss(
                pred_corners, gt_corners, beta=self.beta, reduction="sum"
            )
            total_count += n * 8 * 3 
        
        if total_count == 0:
            return total_loss
        
        return total_loss / total_count


class CornerVarianceLoss(nn.Module):
    """penalizes high variance among predicted corners for each object"""
    def forward(self, pred_reg, centers_2d, num_objects):
        
        gathered = _gather_at_centers(pred_reg, centers_2d, num_objects)
        total_loss = pred_reg.new_tensor(0.0)
        total_count = 0
        B = pred_reg.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue

            pred_corners = gathered[b].reshape(n, 8, 3)
            var = pred_corners.var(dim=1)
            total_loss = total_loss + var.sum()
            total_count += n
        
        if total_count == 0:
            return total_loss
        
        return total_loss / total_count


class SizeConsistencyLoss(nn.Module):
    """l1 loss on bounding box edge lengths derived from predicted vs GT corners""" 
    EDGE_PAIRS = [
        (0, 1), (1, 2), (2, 3), (3, 0),  
        (4, 5), (5, 6), (6, 7), (7, 4),  
        (0, 4), (1, 5), (2, 6), (3, 7), 
    ]

    def forward(self, pred_reg, bbox3d_gt, centers_2d, num_objects):
        
        gathered = _gather_at_centers(pred_reg, centers_2d, num_objects)
        total_loss = pred_reg.new_tensor(0.0)
        total_count = 0
        
        B = pred_reg.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_corners = gathered[b].reshape(n, 8, 3) 
            gt_corners = bbox3d_gt[b, :n] 
            
            for i, j in self.EDGE_PAIRS:
                pred_len = (pred_corners[:, i] - pred_corners[:, j]).norm(dim=1)  
                gt_len = (gt_corners[:, i] - gt_corners[:, j]).norm(dim=1) 
                total_loss = total_loss + F.l1_loss(pred_len, gt_len, reduction="sum")
            
            total_count += n * len(self.EDGE_PAIRS)
        
        if total_count == 0:
            return total_loss
        
        return total_loss / total_count


class Center3DLoss(nn.Module):
    """l1 loss on predicted 3D object center at GT center locations"""
    def forward(self, pred_center, center_3d_gt, centers_2d, num_objects):

        gathered = _gather_at_centers(pred_center, centers_2d, num_objects)
        total_loss = pred_center.new_tensor(0.0)
        total_count = 0
        B = pred_center.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_vals = gathered[b]  # (n, 3)
            gt_vals = center_3d_gt[b, :n]  # (n, 3)
            total_loss = total_loss + F.l1_loss(pred_vals, gt_vals, reduction="sum")
            total_count += n * 3

        if total_count == 0:
            return total_loss

        return total_loss / total_count


class HalfEdgeLoss(nn.Module):
    """l1 loss on predicted half-edge vectors (9 values) at GT center locations"""
    def forward(self, pred_reg, gt_half_edges, centers_2d, num_objects):
        gathered = _gather_at_centers(pred_reg, centers_2d, num_objects)
        total_loss = pred_reg.new_tensor(0.0)
        total_count = 0
        B = pred_reg.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_he = gathered[b]  # (n, 9)
            gt_he = gt_half_edges[b, :n]  # (n, 9)
            total_loss = total_loss + F.l1_loss(pred_he, gt_he, reduction="sum")
            total_count += n * 9

        if total_count == 0:
            return total_loss

        return total_loss / total_count


class HalfEdgeScaleLoss(nn.Module):
    """log-space magnitude loss on half-edge vectors"""
    
    def forward(self, pred_reg, gt_half_edges, centers_2d, num_objects):
        gathered = _gather_at_centers(pred_reg, centers_2d, num_objects)
        total_loss = pred_reg.new_tensor(0.0)
        total_count = 0
        B = pred_reg.shape[0]
        for b in range(B):
            n = num_objects[b].item()
            if n == 0:
                continue
            pred_he = gathered[b].reshape(n, 3, 3)  # (n, 3 vectors, 3 coords)
            gt_he = gt_half_edges[b, :n].reshape(n, 3, 3)

            pred_mag = pred_he.norm(dim=2).clamp(min=1e-6)  # (n, 3)
            gt_mag = gt_he.norm(dim=2).clamp(min=1e-6)  # (n, 3)

            log_ratio = torch.log(pred_mag) - torch.log(gt_mag)
            total_loss = total_loss + log_ratio.abs().sum()
            total_count += n * 3

        if total_count == 0:
            return total_loss

        return total_loss / total_count


class CombinedLoss(nn.Module):
    """aggregates all loss components with configurable weights"""
    def __init__(
        self,
        w_heatmap=3.0,
        w_offset=1.0,
        w_corners=15.0,
        w_center=15.0,
        w_scale=1.0,
        focal_alpha=2.0,
        focal_beta=4.0,
    ):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_offset = w_offset
        self.w_corners = w_corners
        self.w_center = w_center
        self.w_scale = w_scale

        self.heatmap_loss = HeatmapFocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.offset_loss = OffsetL1Loss()
        self.corner_loss = HalfEdgeLoss()
        self.center_loss = Center3DLoss()
        self.scale_loss = HalfEdgeScaleLoss()

    def forward(self, predictions, targets):
        
        pred_heatmap = predictions["heatmap"]
        pred_offset = predictions["offset"]
        pred_reg = predictions["regression"]
        pred_center = predictions["center_3d"]

        gt_heatmap = targets["heatmap"]
        gt_half_edges = targets["half_edges"]
        gt_center_3d = targets["center_3d"]
        centers_2d = targets["centers_2d"]
        num_objects = targets["num_objects"]

        l_heatmap = self.heatmap_loss(pred_heatmap, gt_heatmap)
        l_offset = self.offset_loss(pred_offset, centers_2d, num_objects)
        l_corners = self.corner_loss(pred_reg, gt_half_edges, centers_2d, num_objects)
        l_center = self.center_loss(pred_center, gt_center_3d, centers_2d, num_objects)
        l_scale = self.scale_loss(pred_reg, gt_half_edges, centers_2d, num_objects)

        total = (
            self.w_heatmap * l_heatmap
            + self.w_offset * l_offset
            + self.w_corners * l_corners
            + self.w_center * l_center
            + self.w_scale * l_scale
        )

        loss_dict = {
            "heatmap": l_heatmap.detach(),
            "offset": l_offset.detach(),
            "corners": l_corners.detach(),
            "center": l_center.detach(),
            "scale": l_scale.detach(),
            "total": total.detach(),
        }
        
        return total, loss_dict
