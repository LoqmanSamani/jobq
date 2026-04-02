# Project Report — 3D Bounding Box Prediction

---

## Step 0: Project Setup & Data Exploration

### Dataset Overview

The dataset contains **200 samples**, each stored in a UUID-named folder under `data/`. Every sample provides four files:

| File           | Shape     | Dtype   | Description                                               |
| -------------- | --------- | ------- | --------------------------------------------------------- |
| `rgb.jpg`    | (H, W, 3) | uint8   | RGB image                                                 |
| `pc.npy`     | (3, H, W) | float64 | Organized point cloud (X, Y, Z per pixel)                 |
| `bbox3d.npy` | (N, 8, 3) | float32 | 3D bounding box corners (8 vertices in world coordinates) |
| `mask.npy`   | (N, H, W) | bool    | Per-instance binary segmentation masks                    |

### Key Statistics (across all 200 samples)

| Property                     | Min     | Max    | Mean |
| ---------------------------- | ------- | ------ | ---- |
| Image width (px)             | 442     | 1003   | 685  |
| Image height (px)            | 343     | 715    | 517  |
| Objects per image            | 1       | 21     | 9.6  |
| Bbox3d coordinate range      | -0.3504 | 1.4277 | —   |
| Point cloud coordinate range | -1.5933 | 3.0000 | —   |

### Observations

- **Variable image sizes** — cannot batch directly; must resize to a common resolution.
- **Variable number of objects** — need a padding strategy to form fixed-size tensors.
- **Point cloud is organized** — same spatial layout as the image (3×H×W), so it resizes alongside the RGB.
- **3D bbox corners are in world/camera coordinates** — they do NOT depend on image resolution and do not need resizing.
- **Mask centroids closely approximate 3D bbox center projections** — mean distance between the point cloud value at the mask centroid and the bbox 3D center is ~0.02–0.05 units. This validates using mask centroids to generate 2D heatmap targets.
- **No NaN or Inf** found in any point cloud across the dataset.

### System Specs

| Component | Spec                                  |
| --------- | ------------------------------------- |
| CPU       | Intel i5-9300HF (8 threads @ 2.4 GHz) |
| GPU       | NVIDIA GTX 1650 (4 GB VRAM)           |
| RAM       | 16 GB                                 |
| OS        | Ubuntu 24.04                          |
| Python    | 3.12.3                                |
| PyTorch   | 2.11.0+cu126                          |

### Project Structure Created

```
jobq/
├── src/                     # Source code (10 modules)
│   ├── config.py            # Configuration & hyperparameters
│   ├── dataset.py           # Dataset, splits, DataLoaders
│   ├── transforms.py        # Preprocessing & augmentation
│   ├── model.py             # Network architecture (scaffold)
│   ├── losses.py            # Loss functions (scaffold)
│   ├── trainer.py           # Training loop (scaffold)
│   ├── evaluator.py         # Metrics & evaluation (scaffold)
│   ├── inference.py         # Inference & export (scaffold)
│   ├── visualize.py         # Visualization (scaffold)
│   └── utils.py             # Shared helpers
├── tests/                   # Test suite (10 files, mirrors src/)
├── data/                    # Raw data (200 UUID folders)
├── instruction.md           # Design document
├── report.md                # This file
└── README.md                # Dependencies
```

---

## Step 1: Data Exploration & Preprocessing

### Goal

Build the complete data pipeline: load raw data → resize → augment → normalize → generate training targets → batch into DataLoaders.

### What Was Implemented

#### `src/config.py` — Configuration

A single `@dataclass` holding every tunable parameter:

- **Image size**: 256×384 (small enough for 4 GB VRAM, aspect ratio close to dataset mean ~3:2).
- **Output stride**: 4 (heatmap resolution = 64×96).
- **Data split**: 70/15/15 → 140 train / 30 val / 30 test samples.
- **Batch size**: 2 (fits in GPU memory with mixed precision).
- **Max objects**: 25 (dataset max is 21, gives headroom).

#### `src/utils.py` — Helpers

- `set_seed(seed)` — fix Python, NumPy, and PyTorch random seeds for reproducibility.
- `get_device()` — return CUDA if available, else CPU.

#### `src/transforms.py` — Preprocessing & Augmentation

**Resize** (`resize_sample`):

- Image: bilinear interpolation to (256, 384).
- Point cloud: bilinear (XYZ values are continuous).
- Masks: nearest-neighbor (keeps binary edges crisp).
- 3D bbox corners are NOT resized (world coordinates, resolution-independent).

**Normalization**:

- `normalize_image`: uint8 → float32, ImageNet mean/std, channels-first (3, H, W).
- `normalize_point_cloud`: per-sample zero-mean, global std normalization. Uses a single scalar std across X/Y/Z to preserve relative 3D proportions.

**Augmentation** (training only):

- `random_horizontal_flip(p=0.5)`: flips image, masks, AND negates point cloud X-axis for geometric consistency.
- `random_color_jitter`: brightness, contrast, saturation perturbation. Photometric only — does not affect geometry.

**Compose pipelines**:

- `TrainTransform`: resize → flip → jitter → normalize image → normalize PC.
- `ValTransform`: resize → normalize image → normalize PC (deterministic, no randomness).

#### `src/dataset.py` — Dataset & DataLoaders

**Splitting** (`get_data_splits`):

- Scans `data/` for UUID folders, shuffles with a fixed seed, splits by ratio.
- Deterministic: same seed always gives the same split.

**Heatmap target generation** (`generate_heatmap_target`):

- For each object: find mask centroid → map to output resolution (÷ stride) → stamp a Gaussian blob.
- Gaussian radius is proportional to object size (larger objects get wider peaks).
- Overlapping Gaussians use element-wise maximum (CenterNet convention, not sum).
- Returns: heatmap (1, 64, 96) and per-object 2D center coordinates.

**Dataset class** (`BBox3DDataset`):

- `__getitem__` loads rgb.jpg, pc.npy, bbox3d.npy, mask.npy from disk.
- Applies the transform pipeline (resize + augment + normalize).
- Generates heatmap target from the resized masks.
- Pads bbox3d, centers_2d, and masks to `max_objects=25` for batching.
- Returns a dict of PyTorch tensors.

**DataLoader builder** (`build_dataloaders`):

- Creates train (shuffled, drop_last), val, and test DataLoaders.
- Uses `pin_memory=True` for faster GPU transfer.

### Design Decisions

| Decision            | Choice                        | Rationale                                                                  |
| ------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| Input resolution    | 256×384                      | Fits in 4 GB VRAM; aspect ratio ≈ dataset mean                            |
| Output stride       | 4                             | 64×96 heatmap — good balance of precision vs. memory                     |
| Heatmap centers     | Mask centroids                | Directly available, well-correlated with 3D centers (verified empirically) |
| PC normalization    | Per-sample, single global std | Preserves 3D geometry; handles varying depth ranges                        |
| Image normalization | ImageNet stats                | Backbone will be pretrained on ImageNet                                    |
| Object padding      | Fixed max_objects=25          | Allows standard batching with PyTorch default collate                      |
| Augmentation        | Flip + color jitter           | Simple, safe, geometry-consistent; more can be added later                 |

### Tests

**39 tests — all passing.**

| Test file              | Count | What's tested                                                                                                                                                                                                                                                      |
| ---------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_config.py`     | 5     | Default creation, auto-paths, ratios sum to 1, positive values, overrides                                                                                                                                                                                          |
| `test_transforms.py` | 15    | Resize (shapes, dtypes, empty masks), normalize (range, dtype, zero-mean), flip (shape, roundtrip, p=0), jitter (shape, value range), compose (shapes, determinism, no NaN)                                                                                        |
| `test_dataset.py`    | 19    | Splits (no overlap, full coverage, reproducibility, different seeds, sizes), Gaussian kernel (peak, symmetry), heatmap (shape, range, peaks, centers shape, empty mask), dataset (length, shapes, dtypes, num_objects, no NaN), DataLoader (one batch, full epoch) |

```
tests/test_config.py       ✓✓✓✓✓           (5 passed)
tests/test_transforms.py   ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ (15 passed)
tests/test_dataset.py      ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ (19 passed)
─────────────────────────────────────────────
39 passed in ~10s
```

---

## Step 2: Model Architecture

### Goal

Build the detection network: a CenterNet-style anchor-free architecture that takes a 256×384 RGB image and produces dense per-pixel predictions for 3D bounding boxes.

### What Was Implemented

#### `src/model.py` — Network Architecture

Four components, composed sequentially:

**1. Backbone (`Backbone`)** — Feature extraction

- Wraps `timm.create_model("resnet18", features_only=True, out_indices=[1,2,3,4])`.
- Extracts 4 feature levels at strides 4, 8, 16, 32.
- Channel counts: [64, 128, 256, 512].
- Supports pretrained ImageNet weights (default) or random init (for testing).

**2. FPN Neck (`FPNNeck`)** — Multi-scale feature fusion

- 4 lateral 1×1 convolutions project each backbone level to a common channel dimension (64).
- Top-down pathway: deepest level is upsampled (bilinear) and element-wise added to the level above, repeated until stride-4.
- A single 3×3 smoothing convolution is applied to the final stride-4 output to reduce aliasing artifacts from upsampling.
- Output: one feature map at stride-4 resolution (64×96 for 256×384 input).

**3. Detection Head (`DetectionHead`)** — Per-task prediction

- Two layers: 3×3 Conv + ReLU → 1×1 Conv.
- Heatmap head uses a negative bias initialization (`-2.19`) to start with low confidence everywhere (CenterNet convention).

**4. Full Model (`BBox3DNet`)** — End-to-end assembly

- Chains backbone → neck → 3 parallel heads:
  - **Heatmap** (1 channel): object center probability, sigmoid-activated.
  - **Offset** (2 channels): sub-pixel center refinement (fractional part lost to stride).
  - **Regression** (24 channels): 8 × 3 corner coordinates per pixel location.
- `point_cloud` argument accepted in `forward()` for future multimodal fusion (currently unused).

#### `count_parameters(model)` — Helper

Returns `(total_params, trainable_params)` for quick model size checks.

### Architecture Summary

```
Input: (B, 3, 256, 384)
         │
    ┌────▼─────────┐
    │  ResNet-18   │  ← pretrained ImageNet backbone
    │   (timm)     │
    └──┬──┬──┬──┬──┘
       │  │  │  │   stride: 4    8     16    32
       │  │  │  │   chans:  64   128   256   512
    ┌──▼──▼──▼──▼──┐
    │   FPN Neck   │  ← lateral 1×1 + top-down upsample+add + smooth 3×3
    └──────┬───────┘
           │  (B, 64, 64, 96)  ← stride-4 fused features
     ┌─────┼─────┐
     ▼     ▼     ▼
  heatmap offset regression
  (B,1,..) (B,2,..) (B,24,..)
  sigmoid  raw     raw
```

### Parameter Count

| Component             | Parameters       |
| --------------------- | ---------------- |
| Backbone (ResNet-18)  | ~11.2M           |
| FPN Neck              | ~53K             |
| Detection Heads (×3) | ~13K             |
| **Total**       | **~11.3M** |

All parameters are trainable by default.

### Design Decisions

| Decision            | Choice                     | Rationale                                                                                                             |
| ------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Backbone            | ResNet-18                  | Lightweight (~11M params), fits 4 GB VRAM, proven with pretrained features                                            |
| Feature extraction  | timm `features_only`     | Clean API, reliable multi-scale feature access, easy to swap backbones                                                |
| Neck                | FPN (single output level)  | Only stride-4 predictions needed; deeper levels fused for semantics but only the finest is kept                       |
| Smooth conv         | Single (stride-4 only)     | Only the output level needs smoothing; creating convs for discarded levels wastes parameters and breaks gradient flow |
| FPN out channels    | 64                         | Matches stride-4 backbone channels; keeps memory low                                                                  |
| Head architecture   | Conv3×3 + Conv1×1        | Simple two-layer design from CenterNet; enough for per-pixel regression                                               |
| Heatmap init bias   | -2.19 = ln(0.01/0.99)      | Starts with ~1% confidence everywhere; prevents early training instability from overconfident initial predictions     |
| Heatmap activation  | Sigmoid                    | Outputs object-center probability in [0, 1]                                                                           |
| Regression channels | 24 = 8 corners × 3 coords | Directly regresses 3D bbox corners per pixel; no anchor boxes needed                                                  |
| Point cloud input   | Accepted but unused        | Forward-compatible for multimodal fusion in later steps                                                               |

### Tests

**17 tests — all passing.**

| Test file         | Count | What's tested                                                                                                                                                                                                                                                                                                         |
| ----------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_model.py` | 17    | Backbone (output count, shapes, channels attr), FPN (output shape, custom out_channels), DetectionHead (output shape, heatmap bias init), BBox3DNet (instantiation, param count, output keys, output shapes, no NaN, point_cloud arg, backward pass gradient flow, CPU device, batch_size=1, pretrained weight check) |

```
tests/test_model.py   ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓  (17 passed)
──────────────────────────────────────────
17 passed in ~5s
```

### Cumulative Test Count

```
tests/test_config.py       ✓✓✓✓✓                   (5 passed)
tests/test_transforms.py   ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓         (15 passed)
tests/test_dataset.py      ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓     (19 passed)
tests/test_model.py        ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓       (17 passed)
──────────────────────────────────────────────────────
56 passed total
```
