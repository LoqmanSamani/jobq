"""
shared helpers and utilities
more will be added as later steps are implemented.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
   
   
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
