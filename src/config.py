"""
central configuration for hyperparameters, 
all settings for data loading, model, training, and inference!
for parameter finetuning, just change the defaults here.
"""
import os
from dataclasses import dataclass#, field




@dataclass
class Config:
    """
    it holds every tunable parameter.
    defaults are chosen based on my local system (GTX 1650 with 4 GB vRAM, 16 GB RAM)
    """
    project_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = "" # path to the data
    output_dir: str = "" # results/checkpoints dir

    # data settings
    image_height: int = 256 # both w and h are chosen to fit in my local
    image_width: int = 384 
    train_ratio: float = 0.8 # 160 (this with augmentation should be enough for training)
    val_ratio: float = 0.1  # 20 
    test_ratio: float = 0.1 # 20
    num_workers: int = 2
    max_objects: int = 25 # 21 is the dataset max, with fewer will be zero-padded

    # training hyperparams
    batch_size: int = 4
    learning_rate: float = 1e-4 
    weight_decay: float = 1e-4 
    epochs: int = 150
    seed: int = 42
    grad_accum_steps: int = 4   
    warmup_epochs: int = 10   
    early_stop_patience: int = 30 
    use_amp: bool = False  # resnet18 layer1 overflows in fp16 on GTX 1650

    # model architecture
    fpn_channels: int = 64
    head_mid_channels: int = 128
    head_drop_rate: float = 0.1
    feat_drop_rate: float = 0.2

    # augmentation
    flip_p: float = 0.6
    scale_range_min: float = 1.0
    scale_range_max: float = 1.25
    scale_p: float = 0.5
    jitter_brightness: float = 0.4
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.4
    gauss_noise_std_min: float = 0.1
    gauss_noise_std_max: float = 15.0
    gauss_noise_p: float = 0.4
    erasing_p: float = 0.4
    pc_noise_std: float = 0.01
    pc_noise_p: float = 0.6

    # loss weights
    w_heatmap: float = 0.3
    w_offset: float = 0.1
    w_corners: float = 1.5
    w_center: float = 1.5
    w_scale: float = 1.5
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    # inference / evaluation
    top_k: int = 25
    conf_thresh: float = 0.3
    nms_iou_thresh: float = 0.25
    match_dist: float = 0.5

    # heatmap generation
    output_stride: int = 4
    heatmap_min_radius: int = 4
    gaussian_sigma_divisor: float = 6.0

    def __post_init__(self):
        """auto-fill dirs"""
        if not self.data_dir:
            self.data_dir = os.path.join(self.project_dir, "data")
        if not self.output_dir:
            self.output_dir = os.path.join(self.project_dir, "outputs")
