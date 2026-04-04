"""
end to end pipeline:
    - train the model
    - evaluate on test set
    - run inference on sample images
    - expert to ONNX / FP16 ONNX
    - genrate all visualizations 
"""
import os
import sys
import json
import time
import numpy as np
import torch
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import BBox3DNet
from src.losses import CombinedLoss
from src.dataset import build_dataloaders, BBox3DDataset
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.inference import Predictor
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
    """create experiment directory tree"""
    dirs = {
        "root": base_dir,
        "checkpoints": os.path.join(base_dir, "checkpoints"),
        "train": os.path.join(base_dir, "train"),
        "eval": os.path.join(base_dir, "eval"),
        "inference": os.path.join(base_dir, "inference"),
        "inference_samples": os.path.join(base_dir, "inference", "samples"),
        "visualizations": os.path.join(base_dir, "visualizations"),
        "onnx": os.path.join(base_dir, "onnx"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def train(config, device, dirs):
    """train model and return history + model"""
    
    model = BBox3DNet(config, pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    loss_fn = CombinedLoss(
        w_heatmap=config.w_heatmap,
        w_offset=config.w_offset,
        w_corners=config.w_corners,
        w_center=config.w_center,
        w_scale=config.w_scale,
        focal_alpha=config.focal_alpha,
        focal_beta=config.focal_beta,
    )

    train_loader, val_loader, test_loader = build_dataloaders(config)
    print(f"Data splits: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Batch size: {config.batch_size}, Grad accum: {config.grad_accum_steps}, "
          f"Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"Epochs: {config.epochs}, LR: {config.learning_rate}, "
          f"Warmup: {config.warmup_epochs}, Patience: {config.early_stop_patience}")

    trainer = Trainer(model, train_loader, val_loader, loss_fn, config, device)

    start_time = time.time()
    history = trainer.fit()
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time / 60:.1f} minutes "
          f"({len(history)} epochs)")

    # save history
    history_path = os.path.join(dirs["train"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # reload best checkpoint
    ckpt_path = os.path.join(dirs["checkpoints"], "best.pt")
    if os.path.exists(ckpt_path):
        epoch, val_loss = load_checkpoint(ckpt_path, model)
        print(f"Best checkpoint: epoch {epoch}, val_loss {val_loss:.4f}")

    return model, history, train_loader, val_loader, test_loader, train_time


def evaluate(model, test_loader, config, device, dirs):
    """run evaluation on test set"""
   
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

    # save results
    results_path = os.path.join(dirs["eval"], "test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_sample_path = os.path.join(dirs["eval"], "per_sample_results.json")
    with open(per_sample_path, "w") as f:
        json.dump(per_sample, f, indent=2)

    print(f"Evaluation results saved to {dirs['eval']}/")
    return summary, per_sample


def run_inference(model, config, device, dirs):
    """run inference on test samples, collect data for visualizations"""
    
    # get test sample ids
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

    model.eval()
    num_samples = min(20, len(dataset))

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
            # scale intrinsics to the resized image resolution
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

    print(f"Collected {num_samples} inference samples for visualization")

    return images_rgb, gt_bboxes_list, pred_bboxes_list, scores_list, test_ids, gt_heatmaps, pred_heatmaps, intrinsics_list


def export_onnx(model, config, dirs):
    """export model to ONNX and FP16 ONNX"""

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


def generate_visualizations(history, images_rgb, gt_bboxes_list,
                            pred_bboxes_list, gt_heatmaps, pred_heatmaps,
                            intrinsics_list, dirs):
    """generate and save all visualization figures"""
   
    vis_dir = dirs["visualizations"]

    # loss curves (2x3 grid: heatmap, offset, corners, variance, size, total)
    fig = plot_loss_curves(history)
    save_figure(fig, os.path.join(vis_dir, "loss_curves.png"))
    print("  Saved loss_curves.png")

    # wireframe overlays
    n_wire = min(20, len(images_rgb))
    for i in range(n_wire):
        gt_list = [gt_bboxes_list[i][j] for j in range(len(gt_bboxes_list[i]))]
        pred_list = [pred_bboxes_list[i][j] for j in range(len(pred_bboxes_list[i]))]
        fig = plot_wireframe_overlay(images_rgb[i], gt_list, pred_list, intrinsics_list[i])
        save_figure(fig, os.path.join(vis_dir, f"wireframe_{i}.png"))
    print(f"  Saved {n_wire} wireframe overlays")

    # bird's eye view 
    n_bev = min(20, len(images_rgb))
    for i in range(n_bev):
        gt_list = [gt_bboxes_list[i][j] for j in range(len(gt_bboxes_list[i]))]
        pred_list = [pred_bboxes_list[i][j] for j in range(len(pred_bboxes_list[i]))]
        fig = plot_bev(gt_list, pred_list)
        save_figure(fig, os.path.join(vis_dir, f"bev_{i}.png"))
    print(f"  Saved {n_bev} BEV plots")
    print(f"  Saved {n_bev} BEV plots")

    # heatmap comparision
    n_hm = min(6, len(gt_heatmaps))
    for i in range(n_hm):
        fig = plot_heatmap_comparison(gt_heatmaps[i], pred_heatmaps[i])
        save_figure(fig, os.path.join(vis_dir, f"heatmap_{i}.png"))
    print(f"  Saved {n_hm} heatmap comparisons")

    print(f"\nAll visualizations saved to {vis_dir}/")


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: epochs={config.epochs}, lr={config.learning_rate}, "
          f"batch={config.batch_size}, grad_accum={config.grad_accum_steps}")
    
    dirs = setup_dirs(config.output_dir)
    # train
    model, history, train_loader, val_loader, test_loader, train_time = train(
        config, device, dirs
    )

    # evaluate
    summary, per_sample = evaluate(model, test_loader, config, device, dirs)

    # inference
    (images_rgb, gt_bboxes_list, pred_bboxes_list, scores_list,
     test_ids, gt_heatmaps, pred_heatmaps, intrinsics_list) = run_inference(
        model, config, device, dirs
    )

    # ONNX export
    export_info = export_onnx(model.cpu(), config, dirs)

    # visualizations
    generate_visualizations(
        history, images_rgb, gt_bboxes_list,
        pred_bboxes_list, gt_heatmaps, pred_heatmaps, intrinsics_list, dirs
    )

    # save experiment summary
    exp_summary = {
        "config": {
            "image_size": f"{config.image_height}x{config.image_width}",
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "effective_batch_size": config.batch_size * config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "warmup_epochs": config.warmup_epochs,
            "early_stop_patience": config.early_stop_patience,
            "use_amp": config.use_amp,
            "loss_weights": {
                "heatmap": config.w_heatmap,
                "offset": config.w_offset,
                "corners": config.w_corners,
                "center": config.w_center,

            },
        },
        "training": {
            "actual_epochs": len(history),
            "best_val_loss": min(h["val"]["total"] for h in history),
            "final_train_loss": history[-1]["train"]["total"],
            "training_time_minutes": round(train_time / 60, 1),
        },
        "evaluation": summary,
        "onnx_export": export_info,
    }

    with open(os.path.join(dirs["root"], "experiment_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)
    print(f"\nExperiment summary saved to {dirs['root']}/experiment_summary.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
