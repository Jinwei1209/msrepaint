import os
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage, Pad
import nibabel as nib

transform = Compose([ToPILImage(), Pad((40, 40)), CenterCrop((224, 224)), ToTensor()])


class DDPMSynthesisDataset(Dataset):
    def __init__(self, dataset_dirs, mode='train', contrast_name='T1Pre', multicontrast_names=['T1Pre']):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.contrast_name = contrast_name
        self.multicontrast_names = multicontrast_names
        self.image_paths = self.get_image_paths()
        if len(self.multicontrast_names) > 1:
            self.image_paths = self.verify_multicontrast_paths()
        print('{} slices to {} in total'.format(len(self.image_paths), self.mode))

    def get_image_paths(self):
        if self.mode == 'train':
            paths = []
            for dataset_dir in self.dataset_dirs:
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*AXIAL*nii.gz')))
                paths += paths_tmp
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*SAGITTAL*nii.gz')))
                paths += paths_tmp
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*CORONAL*nii.gz')))
                paths += paths_tmp
        else:
            paths = []
            for dataset_dir in self.dataset_dirs[-1:]:
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*AXIAL*100*nii.gz')))
                paths += paths_tmp
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*SAGITTAL*100*nii.gz')))
                paths += paths_tmp
                paths_tmp = sorted(glob(os.path.join(dataset_dir, self.mode, f'*{self.contrast_name}*CORONAL*100*nii.gz')))
                paths += paths_tmp
        return paths

    def verify_multicontrast_paths(self):
        paths_new = []
        for image_path in self.image_paths:
            flag_exist = 1
            for multicontrast_name in self.multicontrast_names:
                if not os.path.exists(image_path.replace(self.contrast_name, multicontrast_name)):
                    flag_exist = 0
            if flag_exist == 1:
                paths_new.append(image_path)
        return paths_new
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        if len(self.multicontrast_names) == 1:
            image_path = self.image_paths[idx]
            image = np.squeeze(nib.load(image_path).get_fdata().astype(np.float32)).transpose((1, 0))
            p99 = np.percentile(image.flatten(), 95)
            image = image / (p99 + 1e-5)
            image = transform(np.clip(image, a_min=0.0, a_max=5.0))

        else:
            images = []
            for multicontrast_name in self.multicontrast_names:
                image_path = self.image_paths[idx].replace(self.contrast_name, multicontrast_name)
                image = np.squeeze(nib.load(image_path).get_fdata().astype(np.float32)).transpose((1, 0))
                p99 = np.percentile(image.flatten(), 95)
                image = image / (p99 + 1e-5)
                image = transform(np.clip(image, a_min=0.0, a_max=5.0))
                images.append(image)
            image = torch.cat(images, dim=0)


        # mask_path = self.image_paths[idx].replace(self.contrast_name, 'LesionSeg')
        # mask_path = self.image_paths[idx].replace(self.contrast_name, 'LesionSeg_LST_AI')
        mask_path = self.image_paths[idx].replace(self.contrast_name, 'LesionSeg_HD_MS_Lesions')

        if not os.path.exists(mask_path):
            mask_path = self.image_paths[idx].replace(self.contrast_name, 'LesionSeg')

        if self.mode == 'train':
            if os.path.exists(mask_path):
                mask = np.squeeze(nib.load(mask_path).get_fdata().astype(np.float32)).transpose((1, 0))
                mask = transform(mask)
                if torch.mean(mask) == 0:
                    attention = torch.ones_like(image)
                else:
                    attention = torch.ones_like(image)
                    for dim in range(len(image)):
                        attention[dim:dim+1, mask[0]==1] = 10
                        # attention[dim:dim+1, mask[0]==1] = 1
            else:
                print('No lesion mask found')
                mask = torch.zeros_like(image)
                attention = 1.0 - mask
        else:
            if os.path.exists(mask_path):
                # lesion filling validation
                mask = np.squeeze(nib.load(mask_path).get_fdata().astype(np.float32)).transpose((1, 0))
                mask = transform(mask)
                attention = mask
                mask = torch.zeros_like(attention)
            else:
                print('No lesion mask found')
                mask = torch.zeros_like(image)
                attention = 1.0 - mask

        return image, mask, attention