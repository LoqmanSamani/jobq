import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.model import BBox3DNet
from src.dataset import BBox3DDataset
from src.evaluator import Evaluator
from src.transforms import ValTransform, IMAGENET_MEAN, IMAGENET_STD
from src.utils import (
    set_seed, get_device, count_parameters,
    get_data_splits, load_checkpoint, decode_heatmap, nms_3d,
    export_to_onnx, validate_onnx, export_to_fp16_onnx,
)
from src.visualize import (
    plot_loss_curves, plot_wireframe_overlay, plot_bev,
    plot_heatmap_comparison, save_figure, estimate_intrinsics,
)


def setup_dirs(base_dir):
    """create inference output directories"""
    dirs = {
        "root": base_dir,
        "eval": os.path.join(base_dir, "eval"),
        "visualizations": os.path.join(base_dir, "visualizations"),
        "onnx": os.path.join(base_dir, "onnx"),
        "train": os.path.join(base_dir, "train"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def load_model(config, checkpoint_path, device):
    """load model from checkpoint"""
    model = BBox3DNet(config, pretrained=False).to(device)
    epoch, val_loss = load_checkpoint(checkpoint_path, model)
    total_params, trainable_params = count_parameters(model)
    print(f"Loaded checkpoint: epoch {epoch}, val_loss {val_loss:.4f}")
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    model.eval()
    return model


def evaluate(model, config, device, dirs):
    """evaluate on test set and save metrics"""

    train_ids, val_ids, test_ids = get_data_splits(
        config.data_dir, config.train_ratio, config.val_ratio, config.seed
    )
    transform = ValTransform(config)
    test_dataset = BBox3DDataset(config.data_dir, test_ids, config, transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    evaluator = Evaluator(
        model, test_loader, config, device,
    )
    summary, per_sample = evaluator.evaluate()

    print("\nTest Set Metrics:")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    with open(os.path.join(dirs["eval"], "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(dirs["eval"], "per_sample_results.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    print(f"Evaluation results saved to {dirs['eval']}/")

    return summary, per_sample


def run_inference(model, config, device):
    """run inference on all test samples, return data for visualizations"""
    print("\n" + "=" * 60)
    print("INFERENCE")
    print("=" * 60)

    train_ids, val_ids, test_ids = get_data_splits(
        config.data_dir, config.train_ratio, config.val_ratio, config.seed
    )

    transform = ValTransform(config)
    dataset = BBox3DDataset(config.data_dir, test_ids, config, transform)

    images_rgb = []
    gt_bboxes_list = []
    pred_bboxes_list = []
    scores_list = []
    gt_heatmaps = []
    pred_heatmaps = []
    intrinsics_list = []

    num_samples = len(dataset)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            point_cloud = sample["point_cloud"].unsqueeze(0).to(device) if "point_cloud" in sample else None
            n_obj = int(sample["num_objects"])

            preds = model(image_tensor, point_cloud=point_cloud)
            decoded = decode_heatmap(
                preds["heatmap"], preds["offset"], preds["regression"],
                preds["center_3d"],
                top_k=config.top_k, conf_thresh=config.conf_thresh,
            )
            det = decoded[0]

            if len(det["scores"]) > 0:
                keep = nms_3d(det["corners"], det["scores"], iou_threshold=config.nms_iou_thresh)
                pred_corners = det["corners"][keep]
                pred_scores = det["scores"][keep]
            else:
                pred_corners = np.zeros((0, 8, 3), dtype=np.float32)
                pred_scores = np.zeros(0, dtype=np.float32)

            # denormalize image for visualization
            img_np = sample["image"].numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            gt_corners = sample["bbox3d"][:n_obj].numpy()

            # load raw point cloud and estimate camera intrinsics for projection
            sample_id = test_ids[i]
            raw_pc = np.load(os.path.join(config.data_dir, sample_id, "pc.npy"))
            orig_h, orig_w = raw_pc.shape[1], raw_pc.shape[2]
            fx_orig, fy_orig, cx_orig, cy_orig = estimate_intrinsics(raw_pc)
            sx = config.image_width / orig_w
            sy = config.image_height / orig_h
            intrinsics = (fx_orig * sx, fy_orig * sy, cx_orig * sx, cy_orig * sy)

            images_rgb.append(img_np)
            gt_bboxes_list.append(gt_corners)
            pred_bboxes_list.append(pred_corners)
            scores_list.append(pred_scores)
            gt_heatmaps.append(sample["heatmap"].numpy())
            pred_heatmaps.append(preds["heatmap"][0].cpu().numpy())
            intrinsics_list.append(intrinsics)

            print(f"  Sample {i+1}/{num_samples}: {len(pred_scores)} detections, "
                  f"{n_obj} GT objects")

    return (images_rgb, gt_bboxes_list, pred_bboxes_list, scores_list,
            test_ids, gt_heatmaps, pred_heatmaps, intrinsics_list)


def generate_visualizations(images_rgb, gt_bboxes_list, pred_bboxes_list,
                            gt_heatmaps, pred_heatmaps, intrinsics_list,
                            dirs, history=None):
    """generate wireframe, BEV, and heatmap visualizations for every sample"""
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    vis_dir = dirs["visualizations"]
    n = len(images_rgb)

    # loss curves (only if history is available from training)
    if history is not None:
        fig = plot_loss_curves(history)
        save_figure(fig, os.path.join(vis_dir, "loss_curves.png"))
        print("  Saved loss_curves.png")

    # wireframe overlays
    for i in range(n):
        gt_list = [gt_bboxes_list[i][j] for j in range(len(gt_bboxes_list[i]))]
        pred_list = [pred_bboxes_list[i][j] for j in range(len(pred_bboxes_list[i]))]
        fig = plot_wireframe_overlay(images_rgb[i], gt_list, pred_list, intrinsics_list[i])
        save_figure(fig, os.path.join(vis_dir, f"wireframe_{i}.png"))
    print(f"  Saved {n} wireframe overlays")

    # BEV plots
    for i in range(n):
        gt_list = [gt_bboxes_list[i][j] for j in range(len(gt_bboxes_list[i]))]
        pred_list = [pred_bboxes_list[i][j] for j in range(len(pred_bboxes_list[i]))]
        fig = plot_bev(gt_list, pred_list)
        save_figure(fig, os.path.join(vis_dir, f"bev_{i}.png"))
    print(f"  Saved {n} BEV plots")

    # heatmap comparisons 
    for i in range(n):
        fig = plot_heatmap_comparison(gt_heatmaps[i], pred_heatmaps[i])
        save_figure(fig, os.path.join(vis_dir, f"heatmap_{i}.png"))
    print(f"  Saved {n} heatmap comparisons")

    print(f"\nAll visualizations saved to {vis_dir}/")


def export_onnx(model, config, dirs):
    """export model to ONNX and FP16 ONNX"""
    print("\n" + "=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)

    onnx_path = os.path.join(dirs["onnx"], "model.onnx")
    export_to_onnx(model, config, onnx_path)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model exported: {onnx_path} ({onnx_size:.1f} MB)")

    result = validate_onnx(onnx_path, model, config)
    print(f"ONNX validation: matches={result['matches']}, diffs={result['max_diff']}")

    fp16_path = os.path.join(dirs["onnx"], "model_fp16.onnx")
    export_to_fp16_onnx(model, config, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"FP16 ONNX exported: {fp16_path} ({fp16_size:.1f} MB)")

    export_info = {
        "onnx_path": onnx_path,
        "onnx_size_mb": round(onnx_size, 2),
        "onnx_validation": result,
        "fp16_path": fp16_path,
        "fp16_size_mb": round(fp16_size, 2),
    }
    with open(os.path.join(dirs["onnx"], "export_info.json"), "w") as f:
        json.dump(export_info, f, indent=2)

    return export_info


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: outputs/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--no-onnx", action="store_true",
        help="Skip ONNX export",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    dirs = setup_dirs(config.output_dir)

    # resolve checkpoint path
    ckpt_path = args.checkpoint or os.path.join(
        config.output_dir, "checkpoints", "best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Train the model first with: python -m pipeline.train")
        return

    model = load_model(config, ckpt_path, device)

    # evaluate
    evaluate(model, config, device, dirs)

    # inference + visualizations for all test samples
    (images_rgb, gt_bboxes_list, pred_bboxes_list, scores_list,
     test_ids, gt_heatmaps, pred_heatmaps, intrinsics_list) = run_inference(
        model, config, device
    )

    # load training history for loss curves if available
    history = None
    history_path = os.path.join(dirs["train"], "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    generate_visualizations(
        images_rgb, gt_bboxes_list, pred_bboxes_list,
        gt_heatmaps, pred_heatmaps, intrinsics_list,
        dirs, history=history,
    )

    # ONNX export
    if not args.no_onnx:
        export_onnx(model.cpu(), config, dirs)

    print("\nDone!")


if __name__ == "__main__":
    main()
