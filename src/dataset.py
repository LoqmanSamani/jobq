"""
dataset and dataloader, the pipeline is as follows:
    - disk(storage)  
    - load raw(load row data frm disk)  
    - transform(apply augmentation and normalization)  
    - generate targets(build centernet-style heatmap from 3d bbox corners and masks)  
    - pad(pad var-len objects to fixed max_objects)  
    - tensor(return pytorch tensors)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.transforms import TrainTransform, ValTransform
from src.utils import set_seed, get_data_splits, gaussian_2d, generate_heatmap_target, canonicalize_half_edges




class BBox3DDataset(Dataset):
    """pytorch dataset"""
    def __init__(self, data_dir, sample_ids, config, transform):
        self.data_dir = data_dir
        self.sample_ids = sample_ids
        self.config = config
        self.transform = transform
        self.output_h = config.image_height // config.output_stride
        self.output_w = config.image_width // config.output_stride

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.data_dir, sample_id)

        # load raw data from disk
        image = np.array(Image.open(os.path.join(sample_dir, "rgb.jpg")))
        point_cloud = np.load(os.path.join(sample_dir, "pc.npy"))
        bbox3d = np.load(os.path.join(sample_dir, "bbox3d.npy"))
        masks = np.load(os.path.join(sample_dir, "mask.npy"))

        num_objects = bbox3d.shape[0]

        # apply transfroms (resize, flip, color jitter, normalize)
        image, point_cloud, masks, flipped = self.transform(image, point_cloud, masks)

        # when horizontally flipped, negate x-components of 3d bboxes to match
        if flipped:
            bbox3d[:, :, 0] = -bbox3d[:, :, 0]

        # filter out objects whose masks disappeared during augmentation
        surviving = np.array([masks[i].any() for i in range(num_objects)])
        if not surviving.all():
            bbox3d = bbox3d[surviving]
            masks = masks[surviving]
            num_objects = int(surviving.sum())

        # heatmap targets and 2d centers
        heatmap, centers_2d = generate_heatmap_target(
            masks, self.output_h, self.output_w,
            min_radius=self.config.heatmap_min_radius,
            sigma_divisor=self.config.gaussian_sigma_divisor,
        )

        # padding variable-length objects to fixed max_objects
        max_obj = self.config.max_objects
        bbox3d_padded = np.zeros((max_obj, 8, 3), dtype=np.float32)
        center_3d_padded = np.zeros((max_obj, 3), dtype=np.float32)
        half_edges_padded = np.zeros((max_obj, 9), dtype=np.float32)
        centers_padded = np.zeros((max_obj, 2), dtype=np.float32)
        masks_padded = np.zeros(
            (max_obj, self.config.image_height, self.config.image_width),
            dtype=np.float32,
        )

        n = min(num_objects, max_obj)
        bbox3d_padded[:n] = bbox3d[:n]
        centers_padded[:n] = centers_2d[:n]
        masks_padded[:n] = masks[:n].astype(np.float32)

        # compute 3d center and half-edge vectors for cuboid parameterization
        if n > 0:
            center_3d = bbox3d[:n].mean(axis=1)  # (n, 3)
            center_3d_padded[:n] = center_3d
            # half-edge vectors: h1=(c1-c0)/2, h2=(c3-c0)/2, h3=(c4-c0)/2
            h1 = (bbox3d[:n, 1, :] - bbox3d[:n, 0, :]) / 2  # (n, 3)
            h2 = (bbox3d[:n, 3, :] - bbox3d[:n, 0, :]) / 2  # (n, 3)
            h3 = (bbox3d[:n, 4, :] - bbox3d[:n, 0, :]) / 2  # (n, 3)
            # canonicalize to remove 48-fold sign/permutation ambiguity
            h1, h2, h3 = canonicalize_half_edges(h1, h2, h3)
            half_edges = np.stack([h1, h2, h3], axis=1).reshape(n, 9)  # (n, 9)
            half_edges_padded[:n] = half_edges

        return {
            "image": torch.from_numpy(image), # (3, h, w)                  
            "point_cloud": torch.from_numpy(point_cloud), # (3, h, w)
            "heatmap": torch.from_numpy(heatmap), # (1, h', w')
            "bbox3d": torch.from_numpy(bbox3d_padded), # (max_obj, 8, 3) absolute corners
            "center_3d": torch.from_numpy(center_3d_padded), # (max_obj, 3)
            "half_edges": torch.from_numpy(half_edges_padded), # (max_obj, 9)
            "centers_2d": torch.from_numpy(centers_padded), # (max_obj, 2)
            "masks": torch.from_numpy(masks_padded), # (max_obj, h, w)
            "num_objects": torch.tensor(n, dtype=torch.long), # scalar
        }



def build_dataloaders(config):
    """build train/val/test dataloaders"""
    set_seed(config.seed)

    train_ids, val_ids, test_ids = get_data_splits(
        config.data_dir, config.train_ratio, config.val_ratio, config.seed,
    )
    train_ds = BBox3DDataset(
        config.data_dir, train_ids, config, TrainTransform(config),
    )
    val_ds = BBox3DDataset(
        config.data_dir, val_ids, config, ValTransform(config),
    )
    test_ds = BBox3DDataset(
        config.data_dir, test_ids, config, ValTransform(config),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
