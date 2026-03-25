import os
import torch
import numpy as np
import argparse
import nibabel as nib

from torchvision.transforms import ToTensor
from .utils import normalize_intensity
from .network import FusionNet
from torch.cuda.amp import autocast
from .test_volume_mc import resize_volume, resize_back


if __name__ == '__main__':

    opt = {}
    parser = argparse.ArgumentParser(description = "Fusion model")
    parser.add_argument("--gpu_id", type = str, default = "0")
    parser.add_argument("--image_paths", type = str, default = [])
    parser.add_argument("--out_path", type = str, default = "")
    parser.add_argument("--norm_val", type = float, default = 1.0)
    parser.add_argument("--pretrained_fusion", type = str, default = "")
    opt = {**opt, **vars(parser.parse_args())}


    image_paths = opt['image_paths'].split(',')
    out_path = opt['out_path']
    norm_val = opt['norm_val']
    pretrained_fusion = opt['pretrained_fusion']

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']

    # obtain images
    images = []
    flag_resize = 0
    for image_path in image_paths:
        image_pad = torch.zeros((224, 224, 224))
        image_obj = nib.load(image_path)
        if pretrained_fusion != "":
            image_vol, _ = normalize_intensity(torch.from_numpy(image_obj.get_fdata().astype(np.float32)))
        else:
            image_vol = torch.from_numpy(image_obj.get_fdata().astype(np.float32))
        if image_vol.shape != (192, 224, 192):
            flag_resize = 1
            original_size = image_vol.shape
            image_vol = resize_volume(image_vol)
        image_pad[112 - 96:112 + 96, :, 112 - 96:112 + 96] = image_vol
        image_header = image_obj.header
        images.append(image_pad.numpy())

    if pretrained_fusion != "":
        print('Fusion with fusion network')
        checkpoint = torch.load(pretrained_fusion, map_location='cuda')
        fusion_net = FusionNet(in_ch=3, out_ch=1)
        fusion_net.load_state_dict(checkpoint['fusion_net'])
        fusion_net.to('cuda')
        fusion_net.eval()
        with autocast():
            image = torch.cat(
                [ToTensor()(im).permute(2, 1, 0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) for im in images],
                dim=1).to('cuda')
            image_fusion = fusion_net(image).squeeze().detach().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
    else:
        # calculate median
        print('Fusion with median')
        image_cat = np.stack(images, axis=-1)
        image_fusion = np.median(image_cat, axis=-1)

        # # calculate mean
        # print('Fusion with mean')
        # image_cat = np.stack(images, axis=-1)
        # image_fusion = np.mean(image_cat, axis=-1)

    # save fusion_image
    img_save = image_fusion[112 - 96:112 + 96, :, 112 - 96:112 + 96] * norm_val
    if flag_resize:
        img_save = resize_back(img_save, original_size)
    img_save = nib.Nifti1Image(img_save, None, image_header)
    nib.save(img_save, out_path)
                    
                    
                
