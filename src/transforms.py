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


def random_point_cloud_noise(point_cloud, std=0.01, p=0.5):
    """add random gaussian noise to point cloud channels (before normalization)"""
    if np.random.rand() >= p:
        return point_cloud
    noise = np.random.randn(*point_cloud.shape).astype(point_cloud.dtype) * std
    return point_cloud + noise


def random_horizontal_flip(image, point_cloud, masks, p=0.5):
    """randomly flip image, point cloud and masks horizontally with prob p"""
    flipped = np.random.rand() < p
    if flipped:
        image = np.ascontiguousarray(image[:, ::-1, :])
        point_cloud = np.ascontiguousarray(point_cloud[:, :, ::-1])
        point_cloud[0] = -point_cloud[0] # x channels should be negated when flipped horizontally
        masks = np.ascontiguousarray(masks[:, :, ::-1]) 
    return image, point_cloud, masks, flipped


def random_color_jitter(image, brightness=0.5, contrast=0.5, saturation=0.5):
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


def random_gaussian_noise(image, std_range=(0.0, 25.0), p=0.5):
    """add random gaussian noise to image (before normalization, uint8 range)"""
    if np.random.rand() >= p:
        return image
    std = np.random.uniform(*std_range)
    noise = np.random.randn(*image.shape) * std
    img = image.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def random_erasing(image, p=0.5, sl=0.02, sh=0.15, r1=0.3):
    """randomly erase a rectangular region of the image (cutout-style)"""
    if np.random.rand() >= p:
        return image
    img = image.copy()
    H, W = img.shape[:2]
    area = H * W
    for _ in range(10):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1.0 / r1)
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        if h < H and w < W:
            y = np.random.randint(0, H - h)
            x = np.random.randint(0, W - w)
            img[y:y+h, x:x+w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            break
    return img


def random_scale_crop(image, point_cloud, masks, scale_range=(0.8, 1.2), p=0.5):
    """random scale + center crop back to original size"""
    if np.random.rand() >= p:
        return image, point_cloud, masks
    H, W = image.shape[:2]
    scale = np.random.uniform(*scale_range)
    new_h, new_w = int(H * scale), int(W * scale)
    # resize all
    img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pc_hwc = point_cloud.transpose(1, 2, 0)
    pc_hwc = cv2.resize(pc_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    n = masks.shape[0]
    ms = np.zeros((n, new_h, new_w), dtype=bool)
    for i in range(n):
        m = masks[i].astype(np.uint8) * 255
        ms[i] = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 127
    # center crop or pad back to (H, W)
    if new_h >= H and new_w >= W:
        y0 = (new_h - H) // 2
        x0 = (new_w - W) // 2
        img = img[y0:y0+H, x0:x0+W]
        pc_hwc = pc_hwc[y0:y0+H, x0:x0+W]
        ms = ms[:, y0:y0+H, x0:x0+W]
    else:
        pad_img = np.zeros((H, W, 3), dtype=img.dtype)
        pad_pc = np.zeros((H, W, 3), dtype=pc_hwc.dtype)
        pad_ms = np.zeros((n, H, W), dtype=bool)
        y0 = max(0, (H - new_h) // 2)
        x0 = max(0, (W - new_w) // 2)
        h_paste = min(new_h, H)
        w_paste = min(new_w, W)
        pad_img[y0:y0+h_paste, x0:x0+w_paste] = img[:h_paste, :w_paste]
        pad_pc[y0:y0+h_paste, x0:x0+w_paste] = pc_hwc[:h_paste, :w_paste]
        pad_ms[:, y0:y0+h_paste, x0:x0+w_paste] = ms[:, :h_paste, :w_paste]
        img, pc_hwc, ms = pad_img, pad_pc, pad_ms
        
    return img, pc_hwc.transpose(2, 0, 1), ms


class TrainTransform:
    """transform pipeline for training data"""
    def __init__(self, config):
        self.h = config.image_height
        self.w = config.image_width
        self.flip_p = config.flip_p
        self.scale_range = (config.scale_range_min, config.scale_range_max)
        self.scale_p = config.scale_p
        self.jitter_brightness = config.jitter_brightness
        self.jitter_contrast = config.jitter_contrast
        self.jitter_saturation = config.jitter_saturation
        self.gauss_noise_std = (config.gauss_noise_std_min, config.gauss_noise_std_max)
        self.gauss_noise_p = config.gauss_noise_p
        self.erasing_p = config.erasing_p
        self.pc_noise_std = config.pc_noise_std
        self.pc_noise_p = config.pc_noise_p

    def __call__(self, image, point_cloud, masks):
    
        image, point_cloud, masks = resize_sample(
            image, point_cloud, masks, self.h, self.w
        )
        image, point_cloud, masks, flipped = random_horizontal_flip(
            image, point_cloud, masks, p=self.flip_p
        )
        image, point_cloud, masks = random_scale_crop(
            image, point_cloud, masks, scale_range=self.scale_range, p=self.scale_p
        )
        image = random_color_jitter(image, brightness=self.jitter_brightness,
                                    contrast=self.jitter_contrast,
                                    saturation=self.jitter_saturation)
        image = random_gaussian_noise(image, std_range=self.gauss_noise_std, p=self.gauss_noise_p)
        image = random_erasing(image, p=self.erasing_p)
        image = normalize_image(image)
        point_cloud = random_point_cloud_noise(point_cloud, std=self.pc_noise_std, p=self.pc_noise_p)
        point_cloud = normalize_point_cloud(point_cloud)
        
        return image, point_cloud, masks, flipped


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
        
        return image, point_cloud, masks, False
