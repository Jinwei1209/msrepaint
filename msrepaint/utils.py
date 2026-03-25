import numpy as np
import os
import random
import torch
import torch.nn.functional as F

# from torchvision.utils import save_image


def normalize_image(image):
    min_vals = image.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_vals = image.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    norm_image = (image - min_vals) / (max_vals - min_vals)
    return norm_image

def save_sampled_image(image, fpath):
    batch_size = image.shape[0]
    num_channels = image.shape[1]
    image = image.permute(1,0,2,3).reshape(batch_size * num_channels, 1, image.shape[2], image.shape[3])
    image = normalize_image(image)
    # save_image(image, fpath, nrow=batch_size, normalize=False)

def setSeed(seed = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def find_best_contrast(folder_path, sub_id, contrast):

    candidates = []

    for file_name in sorted(os.listdir(folder_path)):
        if sub_id in file_name and contrast in file_name:
            candidates.append(file_name)

    if 'T1' in contrast or ('FLAIR' in contrast and 'CALABRESI' in sub_id):
        for agent in ['PRE', 'POST', '']:
            for acquisition in ['3D', '2D', '']:
                for candidate in candidates:
                    if acquisition in candidate and agent in candidate:
                        return candidate

    else:
        for acquisition in ['3D', '2D', '']:
            for candidate in candidates:
                if acquisition in candidate:
                    return candidate
    
    return ''

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh

def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g /= g.sum()  # Normalize
    kernel = g[:, None] * g[None, :]
    return kernel

def apply_gaussian_filter_2d(data, sigma):
    """Applies a 2D Gaussian filter to each channel in the input tensor."""
    size = int(2 * (3 * sigma) + 1)  # 3-sigma rule for kernel size
    kernel = gaussian_kernel(size, sigma).to(data.device)  # Create the kernel on the same device as data
    kernel = kernel.expand(data.size(1), 1, size, size)  # Expand kernel to apply to each channel separately

    # Explicit padding to handle potential mismatches
    padding = size // 2
    if size % 2 == 0:
        padding_layer = torch.nn.ZeroPad2d((padding, padding - 1, padding, padding - 1))
    else:
        padding_layer = torch.nn.ZeroPad2d((padding, padding, padding, padding))

    # Apply padding before convolution
    data_padded = padding_layer(data)
    filtered_data = F.conv2d(data_padded, kernel, padding=0, groups=data.size(1))
    
    return filtered_data
