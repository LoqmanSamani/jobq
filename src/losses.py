"""
losses:
    - heatmap focal loss (center-net style)
    - offset l1 loss (sub-pixel center refinement) 
    - bbox3d corner loss (smooth l1 on 8×3 corners)
    - corner variance loss (optional)
    - size consistency loss (optional)
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


class CombinedLoss(nn.Module):
    """aggregates all loss components with configurable weights"""
    
    def __init__(
        self,
        w_heatmap=1.0,
        w_offset=1.0,
        w_corners=0.1,
        w_variance=0.0,
        w_size=0.0,
        focal_alpha=2.0,
        focal_beta=4.0,
        smooth_l1_beta=1.0,
    ):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_offset = w_offset
        self.w_corners = w_corners
        self.w_variance = w_variance
        self.w_size = w_size

        self.heatmap_loss = HeatmapFocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.offset_loss = OffsetL1Loss()
        self.corner_loss = BBox3DCornerLoss(beta=smooth_l1_beta)
        
        if w_variance > 0:
            self.variance_loss = CornerVarianceLoss()
        if w_size > 0:
            self.size_loss = SizeConsistencyLoss()

    def forward(self, predictions, targets):
        
        pred_heatmap = predictions["heatmap"]
        pred_offset = predictions["offset"]
        pred_reg = predictions["regression"]

        gt_heatmap = targets["heatmap"]
        gt_bbox3d = targets["bbox3d"]
        centers_2d = targets["centers_2d"]
        num_objects = targets["num_objects"]

        l_heatmap = self.heatmap_loss(pred_heatmap, gt_heatmap)
        l_offset = self.offset_loss(pred_offset, centers_2d, num_objects)
        l_corners = self.corner_loss(pred_reg, gt_bbox3d, centers_2d, num_objects)

        total = (
            self.w_heatmap * l_heatmap
            + self.w_offset * l_offset
            + self.w_corners * l_corners
        )

        loss_dict = {
            "heatmap": l_heatmap.detach(),
            "offset": l_offset.detach(),
            "corners": l_corners.detach(),
        }

        if self.w_variance > 0:
            l_var = self.variance_loss(pred_reg, centers_2d, num_objects)
            total = total + self.w_variance * l_var
            loss_dict["variance"] = l_var.detach()

        if self.w_size > 0:
            l_size = self.size_loss(pred_reg, gt_bbox3d, centers_2d, num_objects)
            total = total + self.w_size * l_size
            loss_dict["size"] = l_size.detach()

        loss_dict["total"] = total.detach()
        
        return total, loss_dict
