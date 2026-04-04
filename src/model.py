"""
centernet-style architecture with geometry-aware heads


   RGB image (b, 3, h, w)         point cloud (b, 3, h, w)
         |                                      |
   resnet18 backbone                      point-cloud encoder
   (pretrained, 3-ch)                  (3 conv layers -> 64ch @ stride 4)
         |                                      |
     FPN neck                                   |
   (64ch @ stride 4)                            |
         |                                      |
         +------------ concat + fuse -----------+
                           |
                   cross-modal fusion
                     (128 -> 64ch)
                           |
                +----------+----------+
                |                     |
          classification          geometry
          branch (3×3 conv)       branch (3×3 conv)
                |                     |
          +-----+-----+         +-----+-----+
          |           |         |           |
      heatmap     offset   regression   center_3d
      head        head     head (deep   head (deep
      (deep)    (shallow)  + geo-cond)  + geo-cond)

detection heads:
    - heatmap head:    (b, 1, 64, 96),  object center probability
    - offset head:     (b, 2, 64, 96),  sub-pixel center refinement (y, x)
    - regression head: (b, 9, 64, 96),  3 half-edge vectors × 3 coords
    - center_3d head:  (b, 3, 64, 96),  3D object center (x, y, z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm





class Backbone(nn.Module):
    """resnet18 backbone for images, pretrained on ImageNet"""
    def __init__(self, pretrained=True, in_chans=3):
        super().__init__()
        self.net = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],
            in_chans=in_chans,
        )
        self.channels = [info["num_chs"] for info in self.net.feature_info[1:]]

    def forward(self, x):
        return self.net(x)


class PointCloudEncoder(nn.Module):
    """encoder for organized point cloud xyz data"""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pc):
        return self.encoder(pc)


class FPNNeck(nn.Module):
    """feature pyramid network that fuses 4 backbone levels into a single stride-4 map"""
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        laterals = [lat_conv(feat) for lat_conv, feat in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled
        return self.smooth(laterals[0])


class DeepHead(nn.Module):
    """detection head"""
    def __init__(self, in_channels, mid_channels, out_channels, init_bias=None, drop_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_rate),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_rate),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        if init_bias is not None:
            nn.init.constant_(self.out_conv.bias, init_bias)

    def forward(self, x):
        return self.out_conv(self.conv(x))



# for backward compat 
DetectionHead = DeepHead



class BBox3DNet(nn.Module):
    """dual-stream centernet-style model with geometry-aware detection heads"""
    def __init__(self, config, pretrained=True):
        super().__init__()

        fpn_ch = config.fpn_channels
        pc_ch = config.fpn_channels   # match fpn output
        head_mid = config.head_mid_channels
        head_drop = config.head_drop_rate
        feat_drop = config.feat_drop_rate
        geo_extra = 5   # geometric conditioning channels (u, v, depth_x, depth_y, depth_z)

        self.backbone = Backbone(pretrained=pretrained, in_chans=3)
        # freeze layer1, prevents overfitting
        for name, param in self.backbone.named_parameters():
            if "layer1" in name:
                param.requires_grad = False

        self.pc_encoder = PointCloudEncoder(in_channels=3, out_channels=pc_ch)
        self.neck = FPNNeck(self.backbone.channels, out_channels=fpn_ch)
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(fpn_ch + pc_ch, fpn_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )
        self.cls_branch = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )
        self.geo_branch = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )
        self.feat_dropout = nn.Dropout2d(p=feat_drop)

        # detection heads
        self.heatmap_head = DeepHead(
            fpn_ch, head_mid, out_channels=1, init_bias=-2.19, drop_rate=head_drop
        )
        self.offset_head = DeepHead(
            fpn_ch, head_mid, out_channels=2, drop_rate=head_drop,
        )
        # geometry heads
        self.regression_head = DeepHead(
            fpn_ch + geo_extra, head_mid, out_channels=9, drop_rate=head_drop,
        )
        self.center_3d_head = DeepHead(
            fpn_ch + geo_extra, head_mid, out_channels=3, drop_rate=head_drop,
        )

    def _make_geo_features(self, fused):
        """create normalized (u, v) coordinate grids for geometric conditioning"""
        B, _, H, W = fused.shape
        u = torch.linspace(0, 1, W, device=fused.device, dtype=fused.dtype)
        v = torch.linspace(0, 1, H, device=fused.device, dtype=fused.dtype)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        coords = torch.stack([grid_u, grid_v], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return coords

    def forward(self, image, point_cloud=None):

        # RGB features
        rgb_features = self.backbone(image)
        fpn_out = self.neck(rgb_features)  # (b, 64, h, w)

        # point cloud features 
        if point_cloud is not None:
            pc_features = self.pc_encoder(point_cloud)  # (b, 64, h, w)
        
            if pc_features.shape[2:] != fpn_out.shape[2:]:
                pc_features = F.interpolate(
                    pc_features, size=fpn_out.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            # cross-modal fusion
            fused = self.cross_fusion(torch.cat([fpn_out, pc_features], dim=1))
        else:
            fused = fpn_out

        # task-specific branches
        cls_feat = self.cls_branch(fused)
        geo_feat = self.geo_branch(fused)   
        cls_feat = self.feat_dropout(cls_feat)
        geo_feat = self.feat_dropout(geo_feat)

        # classification heads 
        heatmap = self.heatmap_head(cls_feat)
        offset = self.offset_head(cls_feat)

        # geometry heads with coordinate conditioning
        geo_cond = self._make_geo_features(geo_feat)  # (b, 2, h, w)
        if point_cloud is not None:
            pc_down = F.interpolate(
                point_cloud, size=geo_feat.shape[2:],
                mode="bilinear", align_corners=False,
            )  # (b, 3, h, w)
            geo_input = torch.cat([geo_feat, geo_cond, pc_down], dim=1)  # 64+2+3 = 69
        else:
            pad = torch.zeros(
                geo_feat.shape[0], 3, geo_feat.shape[2], geo_feat.shape[3],
                device=geo_feat.device, dtype=geo_feat.dtype,
            )
            geo_input = torch.cat([geo_feat, geo_cond, pad], dim=1)  # 64+2+3 = 69

        regression = self.regression_head(geo_input)
        center_3d = self.center_3d_head(geo_input)

        return {
            "heatmap": heatmap,
            "offset": offset,
            "regression": regression,
            "center_3d": center_3d,
        }
