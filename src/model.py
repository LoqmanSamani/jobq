"""
CenterNet-style architecture for 3D bounding box prediction
architecture overview:

      input: image (b, 3, 256, 384)
                 |
esnet18 backbone (pretrained on image-net)
   extracts multi-scale feature maps:
   four feature levels:
       - c1: (b, 64, 64, 96)  stride 4
       - c2: (b, 128, 32, 48) stride 8
       - c3: (b, 256, 16, 24) stride 16
       - c4: (b, 512, 8, 12)  stride 32
                 |
             FPN neck
   fuses multi-scale features into one,
   upsampling deeper feats to stride 4:
       - fused: (b, 64, 64, 96) stride 4 
                 |
detection heads (three parallel heads on the same fused feat map)
    - heatmap head: (b, 1, 64, 96)  — object center probability
    - offset head:  (b, 2, 64, 96)  — sub-pixel center refinement (y, x)
    - regression head: (b, 24, 64, 96) — 3D bbox corners (8 corners × 3 coords)                                 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class Backbone(nn.Module):
    """
    wraps a timm model in 'feaures_only' model to extract multi-scale feature maps
    notw: we use level 1 to 4 and leave level 0 out due to its large spatial size, 
    which would consume too much memory on our 4 gpu
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,# return intermediate feature maps instead of final classifier output
            out_indices=[1, 2, 3, 4],
        )
        # channel counts: (64, 128, 256, 512) for resnet18
        self.channels = [info["num_chs"] for info in self.net.feature_info[1:]]

    def forward(self, x):
        """input is a batch of images(normalized) and it outputs a list of 4 feat maps"""
        return self.net(x)


class FPNNeck(nn.Module):
    """simple feats pyramid network that merges 4 backbone leverls(outputs of Backbone) into one"""
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list # (64, 128, 256, 512)
        ])
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
    
        laterals = [lat_conv(feat) for lat_conv, feat in zip(self.lateral_convs, features)]
        # top-down fusion: upsample deeper levels and add to shallower levels
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        return self.smooth(laterals[0]) # returns the fused feat map at stride 4


class DetectionHead(nn.Module):
    """detection head for one prediction type (heatmap, offset, or regression)"""
    def __init__(self, in_channels, out_channels, init_bias=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if init_bias is not None:
            # for heatmap head(set to -2.19), this helps with training stability (for sigmoid focal loss)
            nn.init.constant_(self.out_conv.bias, init_bias) 
            
    def forward(self, x):
        return self.out_conv(self.conv(x))



class BBox3DNet(nn.Module):
    """complete centernet-style model"""
    def __init__(self, config, pretrained=True):
        super().__init__()

        # backbone (resnet18)
        self.backbone = Backbone(pretrained=pretrained)
        # neck (fpn)
        fpn_out_channels = 64
        self.neck = FPNNeck(self.backbone.channels, out_channels=fpn_out_channels)

        # detection heads
        self.heatmap_head = DetectionHead(
            fpn_out_channels, out_channels=1, init_bias=-2.19# for focal loss stability
        )
        self.offset_head = DetectionHead(
            fpn_out_channels, out_channels=2
        )
        self.regression_head = DetectionHead(
            fpn_out_channels, out_channels=24
        )

    def forward(self, image, point_cloud=None):# point cloud is ignored for now, placeholder for future PC fusion

        features = self.backbone(image)
        fused = self.neck(features)
        heatmap = self.heatmap_head(fused) # (b, 1,  H', W')
        offset = self.offset_head(fused) # (b, 2,  H', W')
        regression = self.regression_head(fused) # (b, 24, H', W')

        return {
            "heatmap": heatmap,
            "offset": offset,
            "regression": regression,
        }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
