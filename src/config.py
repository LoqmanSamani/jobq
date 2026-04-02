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
    train_ratio: float = 0.7 # 140 
    val_ratio: float = 0.15  # 30 
    test_ratio: float = 0.15 # 30
    num_workers: int = 2
    max_objects: int = 25 # 21 is the dataset max, with fewer will be zero-padded

    # training hyperparams
    batch_size: int = 2 
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    seed: int = 42

    # heatmap generation
    output_stride: int = 4
    heatmap_min_radius: int = 2

    def __post_init__(self):
        """auto-fill dirs"""
        if not self.data_dir:
            self.data_dir = os.path.join(self.project_dir, "data")
        if not self.output_dir:
            self.output_dir = os.path.join(self.project_dir, "outputs")
