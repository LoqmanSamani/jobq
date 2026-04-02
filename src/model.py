# ============================================================================
# model.py — Neural network architecture for 3D bounding box prediction
# ============================================================================
#
# PURPOSE:
#   Defines the model that takes RGB (and optionally point cloud) input and
#   predicts 3D bounding boxes for each object in the scene.
#
# STRUCTURE / CONTENTS:
#   1. Backbone — feature extractor
#      - Use a lightweight pretrained backbone (ResNet-18 or EfficientNet-B0)
#        to keep memory and compute reasonable.
#      - Extract multi-scale feature maps (FPN-style) for detecting objects
#        at different scales.
#      - ImageBackbone(name, pretrained): wraps timm / torchvision models
#
#   2. Point cloud branch (optional)
#      - PointCloudEncoder: small CNN or shared MLP on the organized (H,W,3)
#        point cloud to produce per-pixel geometric features.
#      - Concatenate or add geometric features to RGB features (early or
#        mid-level fusion).
#
#   3. Neck — Feature Pyramid Network (FPN)
#      - FPNNeck(in_channels_list, out_channels)
#      - Merges multi-scale backbone features into a uniform representation.
#
#   4. Detection head
#      - BBox3DHead(in_channels, max_objects):
#          a. Heatmap head → (H', W', 1) object center heatmap (like CenterNet)
#          b. Offset head → (H', W', 2) sub-pixel center offset
#          c. Corner/regression head → (H', W', 24) = 8 corners × 3 coords
#             OR parameterize as center (3) + dimensions (3) + rotation (1–4)
#          d. (Optional) Mask head for auxiliary instance segmentation loss
#
#   5. Full model assembly
#      - BBox3DNet(config):
#          - Combines backbone + (optional PC branch) + FPN + detection head
#          - forward(image, point_cloud=None) → predictions dict
#
# ARCHITECTURE RATIONALE:
#   - CenterNet-style (anchor-free) is chosen because:
#       • Simpler than anchor-based (fewer hyperparameters, no NMS needed
#         during training)
#       • Works well on small datasets
#       • Direct regression of 3D bbox corners from center points
#   - Lightweight backbone because of limited compute resources.
#   - FPN to handle objects at different scales in the image.
#
# NOTES:
#   - With only 200 training samples, a smaller model with pretrained
#     backbone is strongly preferable over training from scratch.
#   - Consider freezing early backbone layers to reduce overfitting.
#   - The point cloud branch is optional — start with RGB-only, add PC
#     fusion later if time permits.
# ============================================================================
