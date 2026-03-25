import torch
import math
import numpy as np
import nibabel as nib
import os
import argparse
from tqdm import tqdm

from scipy import ndimage
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage, Pad
from .model import MSInpaintingWithDDPM
from pathlib import Path

transform = Compose([ToPILImage(), Pad((40, 40)), CenterCrop((224, 224)), ToTensor()])

def background_removal(image_vol):
    [n_row, n_col, n_slc] = image_vol.shape
    thresh = threshold_otsu(image_vol)
    mask = (image_vol >= thresh) * 1.0
    mask = zero_pad(mask, 256)
    mask = isotropic_closing(mask, radius=20)
    mask = crop(mask, n_row, n_col, n_slc)
    image_vol[mask < 1e-4] = 0.0
    return image_vol


def obtain_single_image(image_path, bg_removal=True):
    image_vol = nib.load(image_path).get_fdata().astype(np.float32)
    header = nib.load(image_path).header
    if bg_removal:
        image_vol = background_removal(image_vol)
    return image_vol, header


def load_source_images(image_paths, bg_removal=False):
    source_images = []
    image_header = None
    for image_path in image_paths:
        if str(image_path) != 'no_exist_fpath':
            image_vol, image_header = obtain_single_image(image_path, bg_removal)
            input_size = image_vol.shape
            print('Input image size: ', input_size)
            break

    print('Input Contrast: ')
    for image_path in image_paths:
        if str(image_path) != 'no_exist_fpath':
            image_vol, image_header = obtain_single_image(image_path, bg_removal)
            print('Exist')
        else:
            # image_vol = np.zeros((192, 224, 192))
            image_vol = np.zeros(input_size)
            print('Missing')
        source_images.append(image_vol)
    source_images = np.stack(source_images, axis=0)
    return source_images, image_header

def transform_2d(source_images_2d, task, factor=1.05):  # default
    n_contrast, n_row, n_col = source_images_2d.shape
    image_padded_2d = np.zeros((n_contrast, 224, 224)).astype(np.float32)
    threshs = np.zeros((n_contrast, n_row, n_col)).astype(np.float32)
    for idx_contrast in range(n_contrast):
        image_padded_2d[idx_contrast,
                    112 - n_row // 2:112 + n_row // 2 + n_row % 2,
                    112 - n_col // 2:112 + n_col // 2 + n_col % 2] = source_images_2d[idx_contrast, ...]
        source_image_2d = image_padded_2d[idx_contrast, ...]
        thresh = np.percentile(source_image_2d.flatten(), 95)
        if task == "lesion_generation":
            if idx_contrast == 0:
                source_image_2d = source_image_2d / (thresh + 1e-5) * factor
                threshs[idx_contrast, ...] = (thresh + 1e-5) / factor
            else:
                if idx_contrast == 1:
                    source_image_2d = source_image_2d / (thresh + 1e-5) / factor / 1.05
                    threshs[idx_contrast, ...] = (thresh + 1e-5) * factor * 1.05
                else:
                    source_image_2d = source_image_2d / (thresh + 1e-5) / factor
                    threshs[idx_contrast, ...] = (thresh + 1e-5) * factor
        elif task == "lesion_filling":
            if idx_contrast == 0:
                source_image_2d = source_image_2d / (thresh + 1e-5) / factor
                threshs[idx_contrast, ...] = (thresh + 1e-5) * factor
            else:
                source_image_2d = source_image_2d / (thresh + 1e-5) * factor
                threshs[idx_contrast, ...] = (thresh + 1e-5) / factor
        elif task == "partial_filling_and_generation":
            if idx_contrast == 0:
                source_image_2d = source_image_2d / (thresh + 1e-5) * factor
                threshs[idx_contrast, ...] = (thresh + 1e-5) / factor
            else:
                source_image_2d = source_image_2d / (thresh + 1e-5) / factor
                threshs[idx_contrast, ...] = (thresh + 1e-5) * factor
        else:
            raise ValueError(f'{task} is not a valid task. Choose from: "lesion_generation" and "lesion_filling".')
        source_image_2d = np.clip(source_image_2d, a_min=0.0, a_max=5.0)
        image_padded_2d[idx_contrast, ...] = source_image_2d
    return torch.from_numpy(image_padded_2d).unsqueeze(0), threshs

def transform_2d_pre_threshs(source_images_2d, threshs):
    n_contrast, n_row, n_col = source_images_2d.shape
    image_padded_2d = np.zeros((n_contrast, 224, 224)).astype(np.float32)
    for idx_contrast in range(n_contrast):
        image_padded_2d[idx_contrast,
                    112 - n_row // 2:112 + n_row // 2 + n_row % 2,
                    112 - n_col // 2:112 + n_col // 2 + n_col % 2] = source_images_2d[idx_contrast, ...]
        source_image_2d = image_padded_2d[idx_contrast, ...]
        thresh = threshs[idx_contrast, 0, 0]
        source_image_2d = source_image_2d / thresh
        source_image_2d = np.clip(source_image_2d, a_min=0.0, a_max=5.0)
        image_padded_2d[idx_contrast, ...] = source_image_2d
    return torch.from_numpy(image_padded_2d).unsqueeze(0)

def resize_volume(volume, target_size=(192, 224, 192)):
    """
    Resize a 3D volume by zero-padding or cropping to match the target size.
    
    Parameters:
    volume (numpy array): The input 3D volume of size (Nx, Ny, Nz).
    target_size (tuple): The target size (Tx, Ty, Tz).
    
    Returns:
    numpy array: The resized 3D volume.
    """
    Nx, Ny, Nz = volume.shape
    Tx, Ty, Tz = target_size
    
    # Initialize a new volume with the target size and filled with zeros
    if isinstance(volume, np.ndarray):
        resized_volume = np.zeros(target_size, dtype=volume.dtype)
    elif isinstance(volume, torch.Tensor):
        resized_volume = torch.zeros(target_size, dtype=volume.dtype)
    else:
        raise ValueError("Input volume must be a numpy array or a torch tensor")
    
    # Calculate cropping/padding indices for each dimension
    x_min = max((Tx - Nx) // 2, 0)
    x_max = min((Tx + Nx) // 2, Tx)
    x_start = max((Nx - Tx) // 2, 0)
    x_end = x_start + (x_max - x_min)
    
    y_min = max((Ty - Ny) // 2, 0)
    y_max = min((Ty + Ny) // 2, Ty)
    y_start = max((Ny - Ty) // 2, 0)
    y_end = y_start + (y_max - y_min)
    
    z_min = max((Tz - Nz) // 2, 0)
    z_max = min((Tz + Nz) // 2, Tz)
    z_start = max((Nz - Tz) // 2, 0)
    z_end = z_start + (z_max - z_min)
    
    # Fill the resized volume with the cropped/centered volume
    resized_volume[x_min:x_max, y_min:y_max, z_min:z_max] = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    
    return resized_volume

def resize_back(volume, original_size):
    """
    Resize a 3D volume back to its original size by undoing zero-padding or cropping.
    
    Parameters:
    volume (numpy array): The input 3D volume of size (Tx, Ty, Tz).
    original_size (tuple): The original size (Nx, Ny, Nz).
    
    Returns:
    numpy array: The volume resized back to its original size.
    """
    Nx, Ny, Nz = original_size
    Tx, Ty, Tz = volume.shape
    
    # Initialize a new volume with the original size and filled with zeros
    if isinstance(volume, np.ndarray):
        original_volume = np.zeros(original_size, dtype=volume.dtype)
    elif isinstance(volume, torch.Tensor):
        original_volume = torch.zeros(original_size, dtype=volume.dtype)
    else:
        raise ValueError("Input volume must be a numpy array or a torch tensor")
    
    # Calculate cropping/padding indices for each dimension
    x_min = max((Nx - Tx) // 2, 0)
    x_max = min((Nx + Tx) // 2, Nx)
    x_start = max((Tx - Nx) // 2, 0)
    x_end = x_start + (x_max - x_min)
    
    y_min = max((Ny - Ty) // 2, 0)
    y_max = min((Ny + Ty) // 2, Ny)
    y_start = max((Ty - Ny) // 2, 0)
    y_end = y_start + (y_max - y_min)
    
    z_min = max((Nz - Tz) // 2, 0)
    z_max = min((Nz + Tz) // 2, Nz)
    z_start = max((Tz - Nz) // 2, 0)
    z_end = z_start + (z_max - z_min)
    
    # Fill the original volume with the cropped/centered volume
    original_volume[x_min:x_max, y_min:y_max, z_min:z_max] = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    
    return original_volume



class DDPMConfig:
    def __init__(self, in_ch, out_ch, loss_weighting_factor=10, training_label_flag=0, pretrained_model=None):

        self.device = torch.device('cuda:0')
        self.num_diffusion_steps = int(1e3)
        self.betas = self.cosine_scheduling(self.num_diffusion_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.time_embedding_dim = 16
        self.num_epochs = 300
        self.lr = 3e-4
        self.batch_size = 1
        self.dataset_dirs = []
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.loss_weighting_factor = loss_weighting_factor
        self.training_label_flag = training_label_flag
        self.out_dir = '../experiment_train_on_ms_mix_mc'
        # self.pretrained_model = './experiment_train_on_treat_mc/training_models_20240812-223543/epoch_0300_model.pt'  # 148 best
        # self.pretrained_model = './experiment_train_on_treat_mc/training_models_20240813-114122/epoch_0300_model.pt'  # 153 best for lesion generation
        self.pretrained_model = pretrained_model

    def cosine_scheduling(self, num_diffusion_steps, s=8e-3):
        # Create a time variable that goes from 0 to num_diffusion_steps (inclusive)
        timesteps = torch.linspace(0, num_diffusion_steps, steps=num_diffusion_steps, dtype=torch.float32)
        # Calculate the beta schedule as a cosine function
        cos_values = torch.cos(((timesteps / num_diffusion_steps) + s) / (1 + s) * math.pi * 0.5)
        alphas = cos_values**2
        alphas_prev = torch.cat((torch.tensor([1.0]), alphas[:-1]))
        betas = 1 - alphas / alphas_prev
        # Clamp betas to ensure they are in a valid range
        betas = torch.clip(betas, 1e-5, 1.0-1e-5)

        return betas


if __name__ == '__main__':

    opt = {}

    parser = argparse.ArgumentParser(description = "DDPM Bi-directional Lesion Synthesis and Filling with multi-contrast input")
    parser.add_argument("--gpu_id", type = str, default = "0")
    parser.add_argument('--input_fpath', type=Path, action='append', required=True)
    parser.add_argument('--input_synthetic_fpath', type=Path, action='append', required=True)  # synthetic axial input as warm start for coronal and sagittal ddpm
    parser.add_argument('--output_fpath', type=Path, action='append', required=True)
    parser.add_argument("--brain_mask_fpath", type = str, default = "")
    parser.add_argument("--lesion_mask_fpath", type = str, default = "")  # this is mask repaint
    parser.add_argument("--lesion_input_mask_fpath", type = str, default = "")  # this is mask target
    parser.add_argument("--task", type = int, default = 0)  # 0: "lesion_filling"; 1: "lesion_generation"
    parser.add_argument("--inference_plane", type = str, default = 'axial')  # axial, coronal, or sagittal
    parser.add_argument("--permute_id", type = int, default = 0)  # 0: no permute; 1: permute 2D image
    parser.add_argument("--flip_id", type = int, default = 0)  # 0: no flip; 1: flip on dim=[2]; 2: flip on dim][3]; 3: flip on dim=[2,3]
    parser.add_argument("--loop_times", type = int, default = 2)  # repaint inner loop times
    parser.add_argument("--truncation_factor", type = int, default = 25)  # truncation factor for sagittal and coronal view ddim
    parser.add_argument("--loss_weighting_factor", type = int, default = 10)  # lesion region loss weighting factor
    parser.add_argument("--training_label_flag", type = int, default = 0)  # 0: default; 1: LST-AI; 2: HD-MS-Lesions
    parser.add_argument("--pretrained_model", type = str, required = True)  # path to network weight file
    opt = {**opt, **vars(parser.parse_args())}

    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    import numpy as np
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_fpath = opt['input_fpath']
    input_synthetic_fpath = opt['input_synthetic_fpath']
    lesion_mask_fpath = opt['lesion_mask_fpath']
    lesion_input_mask_fpath = opt['lesion_input_mask_fpath']
    output_fpaths = opt['output_fpath']
    task = opt['task']
    inference_plane = opt['inference_plane']
    permute_id = opt['permute_id']
    flip_id = opt['flip_id']
    loop_times = opt['loop_times']
    truncation_factor = opt['truncation_factor']
    loss_weighting_factor = opt['loss_weighting_factor']
    training_label_flag = opt['training_label_flag']
    pretrained_model = opt['pretrained_model']

    if task == 0:
        task = "lesion_filling"
    elif task == 1:
        task = "lesion_generation"
    elif task == 2:
        task = "partial_filling_and_generation"

    if lesion_input_mask_fpath == "":
        lesion_input_mask_fpath = lesion_mask_fpath

    # initialize model
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    config = DDPMConfig(in_ch=len(input_fpath)+1, out_ch=len(input_fpath),
                        loss_weighting_factor=loss_weighting_factor,
                        training_label_flag=training_label_flag,
                        pretrained_model=pretrained_model)
    trainer = MSInpaintingWithDDPM(config)

    # load data
    flag_resize = 0
    image_volume, header = load_source_images(input_fpath, False)
    if image_volume[0, ...].shape != (192, 224, 192):
        flag_resize = 1
        print('Resize image volumes to MNI size (192, 224, 192)')
        original_size = image_volume[0, ...].shape
        print('Original image size is: ', original_size)
        image_volume_new = np.zeros((image_volume.shape[0],)+(192, 224, 192))
        for idx in range(image_volume.shape[0]):
            image_volume_new[idx, ...] = resize_volume(image_volume[idx, ...], target_size=(192, 224, 192))
        image_volume = image_volume_new
    
    if not (inference_plane == 'axial' and permute_id == 0 and flip_id == 0):
        synthetic_image_volume, _ = load_source_images(input_synthetic_fpath, False)
        if synthetic_image_volume[0, ...].shape != (192, 224, 192):
            print('Resize synthetic image volumes to MNI size (192, 224, 192)')
            original_size_syn = synthetic_image_volume[0, ...].shape
            synthetic_image_volume_new = np.zeros((synthetic_image_volume.shape[0],)+(192, 224, 192))
            for idx in range(synthetic_image_volume.shape[0]):
                synthetic_image_volume_new[idx, ...] = resize_volume(synthetic_image_volume[idx, ...], target_size=(192, 224, 192))
            synthetic_image_volume = synthetic_image_volume_new
    
    image_synth_volume = np.copy(image_volume)
    mask_volume = nib.load(lesion_mask_fpath).get_fdata().astype(np.float32)
    mask_input_volume = nib.load(lesion_input_mask_fpath).get_fdata().astype(np.float32)
    if mask_volume.shape != (192, 224, 192):
        print('Resize repaint mask volume to MNI size (192, 224, 192)')
        original_size_mask = mask_volume.shape
        mask_volume = resize_volume(mask_volume, target_size=(192, 224, 192))
    if mask_input_volume.shape != (192, 224, 192):
        print('Resize input mask volume to MNI size (192, 224, 192)')
        original_size_mask = mask_input_volume.shape
        mask_input_volume = resize_volume(mask_input_volume, target_size=(192, 224, 192))

    if inference_plane == 'axial':
        nslices = image_volume.shape[3]
    elif inference_plane == 'coronal':
        nslices = image_volume.shape[2]
    elif inference_plane == 'sagittal':
        nslices = image_volume.shape[1]

    for slice_id in tqdm(range(nslices)):
    # for slice_id in range(100, 180):
        if inference_plane == 'axial':
            if np.sum(mask_volume[:, :, slice_id]) == 0:
                # print('Skip slice #', slice_id)
                continue
            # else:
            #     print('Processing slice #', slice_id)
        elif inference_plane == 'coronal':
            if np.sum(mask_volume[:, slice_id, :]) == 0:
                # print('Skip slice #', slice_id)
                continue
            # else:
            #     print('Processing slice #', slice_id)
        elif inference_plane == 'sagittal':
            if np.sum(mask_volume[slice_id, :, :]) == 0:
                # print('Skip slice #', slice_id)
                continue
            # else:
            #     print('Processing slice #', slice_id)

        if inference_plane == 'axial':
            image = image_volume[:, :, :, slice_id].transpose([0, 2, 1])
            image, threshs = transform_2d(image, task)
            mask = mask_volume[:, :, slice_id].transpose([1, 0])
            mask_input = mask_input_volume[:, :, slice_id].transpose([1, 0])
            if permute_id == 0 and flip_id == 0:
                synthetic_image = None
            else:
                synthetic_image = synthetic_image_volume[:, :, :, slice_id].transpose([0, 2, 1])
                synthetic_image = transform_2d_pre_threshs(synthetic_image, threshs)
        elif inference_plane == 'coronal':
            image = np.flip(image_volume[:, :, slice_id, :], 2).transpose([0, 2, 1])
            image, threshs = transform_2d(image, task)
            mask = np.flip(mask_volume[:, slice_id, :], 1).transpose([1, 0])
            mask_input = np.flip(mask_input_volume[:, slice_id, :], 1).transpose([1, 0])
            synthetic_image = np.flip(synthetic_image_volume[:, :, slice_id, :], 2).transpose([0, 2, 1])
            synthetic_image = transform_2d_pre_threshs(synthetic_image, threshs)
        elif inference_plane == 'sagittal':
            image = np.flip(image_volume[:, slice_id, :, :], 2).transpose([0, 2, 1])
            image, threshs = transform_2d(image, task)
            mask = np.flip(mask_volume[slice_id, :, :], 1).transpose([1, 0])
            mask_input = np.flip(mask_input_volume[slice_id, :, :], 1).transpose([1, 0])
            synthetic_image = np.flip(synthetic_image_volume[:, slice_id, :, :], 2).transpose([0, 2, 1])
            synthetic_image = transform_2d_pre_threshs(synthetic_image, threshs)
        else:
            raise ValueError(f'{inference_plane} is not a valid inference plane. Choose from: "axial", "coronal", and "sagittal".')

        # if task == "lesion_filling":
        #     # Perform dilation
        #     se = np.ones((5, 5))
        #     mask = ndimage.binary_dilation(mask, structure=se).astype(np.float32)

        mask = transform(mask).unsqueeze(0)
        mask_input = transform(mask_input).unsqueeze(0)

        # additional transpose and flip
        if permute_id == 1:
            image = image.permute(0 ,1, 3, 2)
            mask = mask.permute(0 ,1, 3, 2)
            mask_input = mask_input.permute(0 ,1, 3, 2)
            if synthetic_image is not None:
                synthetic_image = synthetic_image.permute(0 ,1, 3, 2)
        if flip_id == 1:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            mask_input = torch.flip(mask_input, dims=[2])
            if synthetic_image is not None:
                synthetic_image = torch.flip(synthetic_image, dims=[2])
        elif flip_id == 2:
            image = torch.flip(image, dims=[3])
            mask = torch.flip(mask, dims=[3])
            mask_input = torch.flip(mask_input, dims=[3])
            if synthetic_image is not None:
                synthetic_image = torch.flip(synthetic_image, dims=[3])
        elif flip_id == 3:
            image = torch.flip(image, dims=[2, 3])
            mask = torch.flip(mask, dims=[2, 3])
            mask_input = torch.flip(mask_input, dims=[2, 3])
            if synthetic_image is not None:
                synthetic_image = torch.flip(synthetic_image, dims=[2, 3])

        # # contrast dropout test
        # image[:, 1:, ...] = 0
        # if synthetic_image is not None:
        #     synthetic_image[:, 1:, ...] = 0
        
        # synth_image = trainer.inference(image, mask, 
        #                                 task=task,
        #                                 loop_times=2)

        steps_subset = list(range(0, 1000, 10))  # Use every 10th step
        synth_image = trainer.inference_ddim(
                        image=image, 
                        lesion_mask=mask,
                        lesion_mask_input=mask_input, 
                        task=task,
                        steps_subset=steps_subset, 
                        loop_times=loop_times,
                        synthetic_image=synthetic_image,
                        truncation_factor=truncation_factor)

        # permute and flip back
        if flip_id == 1:
            synth_image = torch.flip(synth_image, dims=[2])
        elif flip_id == 2:
            synth_image = torch.flip(synth_image, dims=[3])
        elif flip_id == 3:
            synth_image = torch.flip(synth_image, dims=[2, 3])
        if permute_id == 1:
            synth_image = synth_image.permute(0 ,1, 3, 2)

        if inference_plane == 'axial':
            image_synth_volume[:, :, :, slice_id] = synth_image.squeeze().cpu().numpy().transpose([0, 2, 1])[:, 112 - 96:112 + 96, :] * threshs.transpose([0, 2, 1])
        elif inference_plane == 'coronal':
            image_synth_volume[:, :, slice_id, :] = np.flip(synth_image.squeeze().cpu().numpy().transpose([0, 2, 1]), 2)[:, 112 - 96:112 + 96, 112 - 96:112 + 96] * threshs.transpose([0, 2, 1])
        elif inference_plane == 'sagittal':
            image_synth_volume[:, slice_id, :, :] = np.flip(synth_image.squeeze().cpu().numpy().transpose([0, 2, 1]), 2)[:, :, 112 - 96:112 + 96] * threshs.transpose([0, 2, 1])

    if flag_resize:
        image_synth_volume_new = np.zeros((image_synth_volume.shape[0],) + original_size)
        for idx in range(image_synth_volume.shape[0]):
            image_synth_volume_new[idx, ...] = resize_back(image_synth_volume[idx, ...], original_size)
        image_synth_volume = image_synth_volume_new

    print(image_synth_volume.shape)
    for idx, output_path in enumerate(output_fpaths):
        if output_path != 'no_exist_fpath':
            img_save = nib.Nifti1Image(image_synth_volume[idx], None, header)
            nib.save(img_save, output_path)