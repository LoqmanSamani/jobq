"""
preprocessing and augmentation
every sample is defined by 4 aspects:
    - image ()h, w, 3) uint8
    - point cloud (3, h, w) float64
    - instance masks (n, h, w) bool
    - 3d bounding boxes (n, 8, 3) float32

transformations:
    - train transform: resize + augmentation + normalization
    - val transform:   resize + normalization (deterministic)

augmentations:
    - random horizontal flip (geometry + appearance)
    - random color jitter (appearance only) 
"""
import cv2
import numpy as np





def resize_sample(image, point_cloud, masks, target_h, target_w):
    """resize sample (image, point cloud, and masks) to a fixed (target_h, target_w)"""
    
    image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR) # image
    pc_hwc = point_cloud.transpose(1, 2, 0)
    pc_hwc_resized = cv2.resize(pc_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pc_resized = pc_hwc_resized.transpose(2, 0, 1) # point cloud

    n_objects = masks.shape[0]
    masks_resized = np.zeros((n_objects, target_h, target_w), dtype=bool)
    for i in range(n_objects):
        m = masks[i].astype(np.uint8) * 255
        m_resized = cv2.resize(m, (target_w, target_h),interpolation=cv2.INTER_NEAREST)
        masks_resized[i] = m_resized > 127 # masks
    return image_resized, pc_resized, masks_resized



# image net stats are used because out backbone is pretrained on image net !
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_image(image):
    """convert unit8 image to float32 and normalize with image net stats"""
    
    img = image.astype(np.float32) / 255.0      
    img = (img - IMAGENET_MEAN) / IMAGENET_STD     
    img = img.transpose(2, 0, 1)
    return img


def normalize_point_cloud(point_cloud):
    """normalize point cloud to zero(mean), unit(std) per sample"""
    
    pc = point_cloud.astype(np.float32)
    mean = pc.mean(axis=(1, 2), keepdims=True)# center each channel independently 
    pc = pc - mean
    std = pc.std() + 1e-8 
    pc = pc / std
    return pc


def random_horizontal_flip(image, point_cloud, masks, p=0.5):
    """randomly flip image, point cloud and masks horizontally with prob p"""
    
    if np.random.rand() < p:
        image = np.ascontiguousarray(image[:, ::-1, :])
        point_cloud = np.ascontiguousarray(point_cloud[:, :, ::-1])
        point_cloud[0] = -point_cloud[0] # x channels should be negated when flipped horizontally
        masks = np.ascontiguousarray(masks[:, :, ::-1]) 
    return image, point_cloud, masks


def random_color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3):
    """change the color of the image randomly"""
    
    img = image.astype(np.float32)

    # shift all pixels up/down
    if brightness > 0:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        img = img * factor

    # scale relative to the mean gray value
    if contrast > 0:
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean

    # blend toward grayscale
    if saturation > 0:
        factor = 1.0 + np.random.uniform(-saturation, saturation)
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * factor

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TrainTransform:
    """transform pipeline for training data"""

    def __init__(self, config):
        self.h = config.image_height
        self.w = config.image_width

    def __call__(self, image, point_cloud, masks):
    
        image, point_cloud, masks = resize_sample(
            image, point_cloud, masks, self.h, self.w
        )
        image, point_cloud, masks = random_horizontal_flip(
            image, point_cloud, masks, p=0.5
        )
        image = random_color_jitter(image)
        image = normalize_image(image)
        point_cloud = normalize_point_cloud(point_cloud)
        
        return image, point_cloud, masks


class ValTransform:
    """transform pipeline for validation/test data"""

    def __init__(self, config):
        self.h = config.image_height
        self.w = config.image_width

    def __call__(self, image, point_cloud, masks):
        
        image, point_cloud, masks = resize_sample(
            image, point_cloud, masks, self.h, self.w
        )
        image = normalize_image(image)
        point_cloud = normalize_point_cloud(point_cloud)
        
        return image, point_cloud, masks
