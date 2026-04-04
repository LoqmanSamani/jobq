"""
visualization utilities for model diagnosis:
    - loss curves: 3-row grid (3 | 1 wide | 3) showing 5 loss components + total + lr
    - 2d wireframe overlay: project predicted/gt 3d bbox corners onto rgb image
    - bird's eye view (bev): top-down x-z plane comparison of gt vs pred boxes
    - heatmap inspection: predicted vs ground truth heatmap side by side
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2




# 12 edges of a cuboid defined by 8 corner indices
BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]


def estimate_intrinsics(raw_pc):
    """estimate pinhole camera intrinsics (fx, fy, cx, cy) from an organized
    point cloud where pc[:, v, u] = (X, Y, Z) at pixel (u, v).

    Uses the relationship: X/Z = (u - cx)/fx, Y/Z = (v - cy)/fy
    and fits via least squares over all valid pixels.
    """
    _, H, W = raw_pc.shape
    Z = raw_pc[2]  # (H, W)
    valid = Z > 1e-3  # avoid div-by-zero

    u_grid = np.arange(W, dtype=np.float64)[None, :] * np.ones((H, 1))
    v_grid = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))

    # fit X/Z = a_x * u + b_x  →  fx = 1/a_x, cx = -b_x/a_x
    xz = raw_pc[0][valid] / Z[valid]
    u_valid = u_grid[valid]
    A = np.vstack([u_valid, np.ones_like(u_valid)]).T
    (a_x, b_x), *_ = np.linalg.lstsq(A, xz, rcond=None)
    fx = 1.0 / a_x
    cx = -b_x / a_x

    # fit Y/Z = a_y * v + b_y  →  fy = 1/a_y, cy = -b_y/a_y
    yz = raw_pc[1][valid] / Z[valid]
    v_valid = v_grid[valid]
    A2 = np.vstack([v_valid, np.ones_like(v_valid)]).T
    (a_y, b_y), *_ = np.linalg.lstsq(A2, yz, rcond=None)
    fy = 1.0 / a_y
    cy = -b_y / a_y

    return float(fx), float(fy), float(cx), float(cy)


def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """perspective-project 3D points to 2D pixel coordinates.

    points_3d: (N, 3)  — world coordinates (X, Y, Z)
    returns:   (N, 2)  — pixel coordinates (u, v)
    """
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    Z_safe = np.where(np.abs(Z) > 1e-6, Z, 1e-6)
    u = fx * X / Z_safe + cx
    v = fy * Y / Z_safe + cy
    return np.stack([u, v], axis=1)


def save_figure(fig, path, dpi=150):
    """save a matplotlib figure to disk"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_projected_bbox3d(image, corners_3d, intrinsics, color=(0, 255, 0), thickness=2):
    """project 3d corners onto the image using pinhole camera model and draw the 12 cuboid edges.

    intrinsics: (fx, fy, cx, cy) at the IMAGE resolution
    """
    img = image.copy()
    H, W = img.shape[:2]
    fx, fy, cx, cy = intrinsics
    pts_2d = project_3d_to_2d(corners_3d, fx, fy, cx, cy)  # (8, 2) as (u, v)
    pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, W - 1)
    pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, H - 1)
    pts_2d = pts_2d.astype(np.int32)

    for i, j in BBOX_EDGES:
        pt1 = tuple(pts_2d[i])
        pt2 = tuple(pts_2d[j])
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def _plot_train_val(ax, history, key, title, nested=True):
    """helper: plot train/val curves for one loss component on *ax*."""
    if nested:
        train_vals = [h["train"].get(key) for h in history if key in h["train"]]
        val_vals = [h["val"].get(key) for h in history if key in h["val"]]
    else:
        train_vals = [h.get(key) for h in history if key in h]
        val_vals = []

    if train_vals:
        ax.plot(train_vals, label="train", color="tab:blue")
    if val_vals:
        ax.plot(val_vals, label="val", color="tab:orange")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_loss_curves(history, figsize=(18, 12)):
    """plot all loss components in a symmetric 3-row layout"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 2, 2, 3, 0],
                          hspace=0.45, wspace=0.30)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_off  = fig.add_subplot(gs[0, 1])
    ax_he   = fig.add_subplot(gs[0, 2])

    _plot_train_val(ax_heat, history, "heatmap",  "Heatmap (Focal)")
    _plot_train_val(ax_off,  history, "offset",   "Offset (L1)")
    _plot_train_val(ax_he,   history, "corners",  "Half-Edges (L1)")

    ax_total = fig.add_subplot(gs[1:3, :])
    _plot_train_val(ax_total, history, "total", "Total Loss")
    ax_total.title.set_fontsize(13)

    ax_cen = fig.add_subplot(gs[3, 0])
    ax_sc  = fig.add_subplot(gs[3, 1])
    ax_lr  = fig.add_subplot(gs[3, 2])

    _plot_train_val(ax_cen, history, "center", "Center 3D (L1)")
    _plot_train_val(ax_sc,  history, "scale",  "Scale (Log)")

    lr_vals = [h.get("lr") for h in history if "lr" in h]
    if lr_vals:
        ax_lr.plot(lr_vals, color="tab:green")
    ax_lr.set_title("Learning Rate", fontsize=11)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Losses", fontsize=14)
    return fig


def plot_wireframe_overlay(image, gt_corners_list, pred_corners_list,
                           intrinsics,
                           gt_color=(0, 255, 0), pred_color=(255, 0, 0),
                           thickness=2, figsize=(14, 5)):
    """draw gt (green) and pred (red) 3d wireframes projected onto the rgb image.

    intrinsics: (fx, fy, cx, cy) at the image resolution.
    returns a matplotlib figure with 3 panels: gt only, pred only, both overlaid.
    """
    gt_img = image.copy()
    for corners in gt_corners_list:
        gt_img = _draw_projected_bbox3d(gt_img, corners, intrinsics, color=gt_color, thickness=thickness)

    pred_img = image.copy()
    for corners in pred_corners_list:
        pred_img = _draw_projected_bbox3d(pred_img, corners, intrinsics, color=pred_color, thickness=thickness)

    both_img = image.copy()
    for corners in gt_corners_list:
        both_img = _draw_projected_bbox3d(both_img, corners, intrinsics, color=gt_color, thickness=thickness)
    for corners in pred_corners_list:
        both_img = _draw_projected_bbox3d(both_img, corners, intrinsics, color=pred_color, thickness=thickness)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, title in zip(axes,
                               [gt_img, pred_img, both_img],
                               ["GT (green)", "Pred (red)", "Overlay"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_bev(gt_corners_list, pred_corners_list,
             gt_color="tab:green", pred_color="tab:red",
             figsize=(8, 8)):
    """bird's eye view: plot gt and pred boxes in the x-z (top-down) plane.

    each box is drawn by projecting all 12 cuboid edges onto (x, z).
    this is ordering-agnostic and works for any corner convention.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for idx, corners in enumerate(gt_corners_list):
        for ei, (i, j) in enumerate(BBOX_EDGES):
            label = "GT" if idx == 0 and ei == 0 else None
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 2], corners[j, 2]],
                    color=gt_color, linewidth=2, label=label)
        cx, cz = corners[:, 0].mean(), corners[:, 2].mean()
        ax.plot(cx, cz, "o", color=gt_color, markersize=4)

    for idx, corners in enumerate(pred_corners_list):
        for ei, (i, j) in enumerate(BBOX_EDGES):
            label = "Pred" if idx == 0 and ei == 0 else None
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 2], corners[j, 2]],
                    color=pred_color, linewidth=2, linestyle="--", label=label)
        cx, cz = corners[:, 0].mean(), corners[:, 2].mean()
        ax.plot(cx, cz, "x", color=pred_color, markersize=6)

    ax.legend(fontsize=10)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Z (depth)", fontsize=11)
    ax.set_title("Bird's Eye View (X\u2013Z plane)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_heatmap_comparison(gt_heatmap, pred_heatmap, figsize=(12, 5)):
    """side-by-side comparison of gt and predicted (sigmoid) heatmaps"""
    # squeeze to 2d
    gt = gt_heatmap.squeeze()
    pred = pred_heatmap.squeeze()

    # apply sigmoid to raw logits
    pred = 1.0 / (1.0 + np.exp(-pred.astype(np.float64)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im0 = axes[0].imshow(gt, cmap="hot", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("GT Heatmap", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, cmap="hot", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Predicted Heatmap", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Heatmap Peak Inspection", fontsize=13)
    plt.tight_layout()
    return fig
