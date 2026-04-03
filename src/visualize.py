"""
visualization utilities:
    - 2d overlays on images (projected bbox edges, masks, gt vs pred)
    - 3d point cloud + bbox wireframe plots (matplotlib 3d)
    - training/validation curves
    - batch visualization grid with gt/pred comparison
    - save helpers for figures
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import cv2


# 12 edges of a cuboid defined by 8 corner indices
BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]


def draw_projected_bbox3d(image, corners_3d, color=(0, 255, 0), thickness=2):
    """project 3d corners onto the image and draw the 12 cuboid edges"""
    
    img = image.copy()
    H, W = img.shape[:2]
    pts_2d = corners_3d[:, :2].copy() 
    pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, W - 1)
    pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, H - 1)
    pts_2d = pts_2d.astype(np.int32)
    
    for i, j in BBOX_EDGES:
        pt1 = tuple(pts_2d[i])
        pt2 = tuple(pts_2d[j])
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def draw_masks(image, masks, alpha=0.4):
    """overlay instance segmentation masks on an RGB image"""
    
    img = image.copy().astype(np.float32)
    n = masks.shape[0]
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 255, size=(n, 3)).astype(np.float32)

    for i in range(n):
        m = masks[i].astype(bool)
        overlay = np.zeros_like(img)
        overlay[m] = colors[i]
        img[m] = (1 - alpha) * img[m] + alpha * overlay[m]

    return img.astype(np.uint8)


def draw_gt_vs_pred(image, gt_corners_list, pred_corners_list,
                    gt_color=(0, 255, 0), pred_color=(255, 0, 0), thickness=2):
    """draw gt boxes (green) and predicted boxes (red) on the same image"""
    
    img = image.copy()
    for corners in gt_corners_list:
        img = draw_projected_bbox3d(img, corners, color=gt_color, thickness=thickness)
    for corners in pred_corners_list:
        img = draw_projected_bbox3d(img, corners, color=pred_color, thickness=thickness)
        
    return img



def plot_point_cloud_with_bboxes(point_cloud, bboxes=None, title="Point Cloud",
                                  subsample=5000, figsize=(10, 8)):
    """3d scatter plot of a point cloud with optional wireframe bounding boxes"""
    
    if point_cloud.ndim == 3 and point_cloud.shape[0] == 3:
        pc = point_cloud.reshape(3, -1).T
    elif point_cloud.ndim == 2 and point_cloud.shape[1] == 3:
        pc = point_cloud
    else:
        raise ValueError(f"unexpected point cloud shape: {point_cloud.shape}")

    if len(pc) > subsample:
        idx = np.random.RandomState(0).choice(len(pc), subsample, replace=False)
        pc = pc[idx]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.3, c=pc[:, 2], cmap="viridis", alpha=0.5)

    if bboxes is not None and len(bboxes) > 0:
        for box in bboxes:
            lines = [[box[i], box[j]] for i, j in BBOX_EDGES]
            lc = Line3DCollection(lines, colors="red", linewidths=1.5)
            ax.add_collection3d(lc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    
    return fig



def plot_training_curves(history, figsize=(12, 5)):
    """plot training and validation loss curves from a history dict"""
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "lr" in history and len(history["lr"]) > 0:
        ax.plot(history["lr"], color="tab:orange")
        ax.set_ylabel("Learning Rate")
    else:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Epoch")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_metric_curves(metric_history, figsize=(12, 5)):
    """plot evaluation metrics over epochs"""
    
    n = len(metric_history)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        return fig

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 1.5))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (name, values) in zip(axes, metric_history.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    
    return fig



def visualize_batch(images, gt_bboxes=None, pred_bboxes=None, num_objects=None,
                    num_samples=4, figsize=(16, 4)):
    """grid of images from a batch with gt (green) and/or predicted (red) boxes"""
    
    if hasattr(images, "cpu"):
        images = images.cpu().numpy()

    B = min(images.shape[0], num_samples)
    fig, axes = plt.subplots(1, B, figsize=figsize)
    if B == 1:
        axes = [axes]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(B):
        img = images[i].transpose(1, 2, 0)  
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        if gt_bboxes is not None:
            n = int(num_objects[i]) if num_objects is not None else gt_bboxes.shape[1]
            for j in range(n):
                img = draw_projected_bbox3d(img, gt_bboxes[i, j], color=(0, 255, 0))

        if pred_bboxes is not None and i < len(pred_bboxes):
            for j in range(len(pred_bboxes[i])):
                img = draw_projected_bbox3d(img, pred_bboxes[i][j], color=(255, 0, 0))

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i}")

    plt.tight_layout()
    
    return fig



def save_figure(fig, path, dpi=150):
    """save a matplotlib figure to disk"""
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_prediction_gallery(images, gt_bboxes_list, pred_bboxes_list,
                             scores_list, output_dir, num_samples=20):
    """generate and save a gallery of prediction visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    n = min(len(images), num_samples)
    for i in range(n):
        img = draw_gt_vs_pred(
            images[i],
            gt_corners_list=[gt_bboxes_list[i][j] for j in range(len(gt_bboxes_list[i]))],
            pred_corners_list=[pred_bboxes_list[i][j] for j in range(len(pred_bboxes_list[i]))],
        )

        for j, score in enumerate(scores_list[i]):
            if j < len(pred_bboxes_list[i]):
                cx = int(pred_bboxes_list[i][j][:, 0].mean())
                cy = int(pred_bboxes_list[i][j][:, 1].mean())
                cx = max(0, min(cx, img.shape[1] - 1))
                cy = max(0, min(cy, img.shape[0] - 1))
                cv2.putText(img, f"{score:.2f}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        path = os.path.join(output_dir, f"pred_{i:03d}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved.append(path)

    return saved
