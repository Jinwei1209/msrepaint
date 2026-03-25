import torch
import os
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from .network import *
from .dataset import *
from .utils import *

class MSInpaintingWithDDPM:
    def __init__(self, config):
        self.config = config
        self.timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_loader, self.valid_loader = None, None
        self.optimizer = None
        self.checkpoint = None

        self.l2_loss = None

        # define network
        self.ddpm = DDPM(config)

        if self.config.pretrained_model is not None:
            self.checkpoint = torch.load(self.config.pretrained_model, 
                                         map_location=self.config.device)
            self.ddpm.load_state_dict(self.checkpoint['ddpm'])
        self.ddpm.to(self.config.device)
        self.start_epoch = 0

    def initialize_training(self):
        self.l2_loss = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=self.config.lr)
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            if 'timestr' in self.checkpoint:
                self.timestr = self.checkpoint['timestr']
            self.start_epoch = self.checkpoint['epoch']
        self.start_epoch += 1

        os.makedirs(self.config.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.out_dir, 
                                 f'training_models_{self.timestr}'), exist_ok=True)
        os.makedirs(os.path.join(self.config.out_dir, 
                                 f'training_results_{self.timestr}'), exist_ok=True)

        setSeed()
        
    def load_dataset(self):
        train_dataset = DDPMSynthesisDataset(self.config.dataset_dirs, mode='train', contrast_name=self.config.contrast_name, multicontrast_names=self.config.multicontrast_names)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        valid_dataset = DDPMSynthesisDataset(self.config.dataset_dirs, mode='valid', contrast_name=self.config.contrast_name, multicontrast_names=self.config.multicontrast_names)
        self.valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

    def forward_diffusion(self, image):
        batch_size = image.shape[0]
        noise = torch.randn_like(image)
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size, ), device=self.config.device)
        alpha_bar_t = self.config.alpha_bars[t].view(-1, 1, 1, 1)
        noisy_image = torch.sqrt(alpha_bar_t) * image + torch.sqrt(1.0 - alpha_bar_t) * noise
        return noisy_image, t, noise
    
    def unconditional_reverse_diffusion(self, image_dim):
        synthetic_image = torch.randn(image_dim, device=self.config.device)
        batch_size = image_dim[0]
        for t in reversed(range(self.config.num_diffusion_steps)):
            t_full = torch.full((batch_size, ), t, device=self.config.device, dtype=torch.float)
            alpha_bar_t = self.config.alpha_bars[t]
            alpha_t = self.config.alphas[t]
            beta_t = self.config.betas[t]
            noise_predict = self.ddpm(synthetic_image, t=t_full, condition=None)
            synthetic_image = (synthetic_image - noise_predict * beta_t / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if t > 0:
                synthetic_image += torch.sqrt(beta_t) * torch.randn(image_dim, device=self.config.device)
        return synthetic_image
    
    def conditional_reverse_diffusion(self, image, lesion_mask, attention, loop_times=1):
        synthetic_image = torch.randn_like(image)
        batch_size = image.shape[0]
        for t in reversed(range(self.config.num_diffusion_steps)):
            # if t % 50 == 0:
            #     print('     Reverse diffusion step: ', t)
            t_full = torch.full((batch_size, ), t, device=self.config.device, dtype=torch.float)
            alpha_bar_t = self.config.alpha_bars[t]
            alpha_t = self.config.alphas[t]
            beta_t = self.config.betas[t]
            for i in range(loop_times):
                noise_predict = self.ddpm(torch.cat([synthetic_image, lesion_mask], dim=1), t=t_full, condition=None)
                synthetic_image = (synthetic_image - noise_predict * beta_t / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
                if t > 0:
                    synthetic_image += torch.sqrt(beta_t) * torch.randn_like(image)
                    noisy_image = torch.sqrt(alpha_bar_t) * image + torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(image)
                    synthetic_image = noisy_image * (1.0 - attention) + synthetic_image * attention
                    if i < loop_times - 1:
                        beta_t1 = self.config.betas[t-1]
                        synthetic_image = torch.randn_like(image) * torch.sqrt(beta_t1) + torch.sqrt(1.0 - beta_t1) * synthetic_image
                else:
                    synthetic_image = image * (1.0 - attention) + synthetic_image * attention
                    break
        return synthetic_image

    def conditional_reverse_diffusion_ddim(self, image, lesion_mask, attention, scheduler, loop_times=1, synthetic_image=None, truncation_factor=25):
        if synthetic_image is None:
            flag_axial = 1
            synthetic_image = torch.randn_like(image)
        else:
            # print('Inference with warm start')
            flag_axial = 0
            scheduler.steps_subset = scheduler.steps_subset[:len(scheduler.steps_subset)//truncation_factor]
            alpha_bar_t = self.config.alpha_bars[scheduler.steps_subset[-1]]
            # synthetic_image from axial plane (right) to generate warm start synthetic_image (left)
            synthetic_image = torch.sqrt(alpha_bar_t) * synthetic_image + torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(synthetic_image)
        batch_size = image.shape[0]
        for idx, t in enumerate(reversed(scheduler.steps_subset)):
            # if idx % 10 == 0:
            #     print('     Reverse diffusion step: ', t)
            t_full = torch.full((batch_size,), t, device=self.config.device, dtype=torch.float)
            alpha_bar_t = self.config.alpha_bars[t]
            alpha_t = self.config.alphas[t]
            for i in range(loop_times):
                noise_predict = self.ddpm(torch.cat([synthetic_image, lesion_mask], dim=1), t=t_full, condition=None)
                synthetic_image, x0_est = scheduler.ddim_step(synthetic_image, t, noise_predict)
                if t > 0:
                    # synthetic_image += scheduler.sigmas[t] * torch.randn_like(synthetic_image)
                    noisy_image = torch.sqrt(alpha_bar_t) * image + torch.sqrt(1.0 - alpha_bar_t) * torch.randn_like(image)
                    synthetic_image = noisy_image * (1.0 - attention) + synthetic_image * attention
                    x0_est = image * (1.0 - attention) + x0_est * attention
                    if i < loop_times - 1:
                        beta_t1 = self.config.betas[t-1]
                        # synthetic_image = torch.randn_like(image) * torch.sqrt(beta_t1) + torch.sqrt(1.0 - beta_t1) * synthetic_image
                        synthetic_image = torch.randn_like(image) * torch.sqrt(1.0 - alpha_bar_t) + torch.sqrt(alpha_bar_t) * x0_est
                else:
                    synthetic_image = image * (1.0 - attention) + synthetic_image * attention
                    break
        return synthetic_image

    def calculate_loss(self, prediction, reference, is_train=False, attention=None):
        loss = self.l2_loss(prediction, reference)
        if attention is not None:
            # loss = loss[attention == 1.0].mean()
            loss = (loss * attention).mean()
        else:
            loss = loss.mean()
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def save_model(self, epoch):
        state = {'epoch': epoch, 
                 'ddpm': self.ddpm.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'timestr': self.timestr}
        torch.save(obj=state, 
                   f=os.path.join(self.config.out_dir, f'training_models_{self.timestr}', 
                                  f'epoch_{str(epoch).zfill(4)}_model.pt'))
        
    def train_and_valid(self):
        for epoch in range(self.start_epoch, self.config.num_epochs+1):
            # TRAINING
            self.train_loader = tqdm(self.train_loader)
            self.ddpm.train()
            train_loss_sum = 0.0
            num_train_images = 0
            for _, (image, lesion_mask, attention) in enumerate(self.train_loader):
                image = image.to(self.config.device)
                lesion_mask = lesion_mask.to(self.config.device)
                attention = attention.to(self.config.device)
                batch_size = image.shape[0]

                if self.config.dropout_training == True and image.shape[1] > 1:
                    all_zeros = True
                    while all_zeros:
                        self.dropout_code = np.random.randint(2, size=(image.shape[1],))
                        if not np.all(self.dropout_code == 0):
                            all_zeros = False
                    print('Contrast dropout code: ', self.dropout_code)
                    image[:, self.dropout_code==0, ...] = 0
                    attention[:, self.dropout_code==0, ...] = 0


                noisy_image, t, noise = self.forward_diffusion(image)
                noise_predict = self.ddpm(torch.cat([noisy_image, lesion_mask], dim=1), t=t, condition=None)
                
                loss = self.calculate_loss(noise_predict, noise, is_train=True, attention=attention)
                train_loss_sum += loss * batch_size
                num_train_images += batch_size
                self.train_loader.set_description((f'epoch: {epoch}; '
                                                   f'loss: {loss:.3f}; '
                                                   f'avg: {train_loss_sum / num_train_images:.3f}; '))

            self.save_model(epoch)

            # VALIDATION
            with torch.set_grad_enabled(False):
                self.ddpm.eval()
                for _, (image, lesion_mask, attention) in enumerate(self.valid_loader):
                    image = image.to(self.config.device)
                    lesion_mask = lesion_mask.to(self.config.device)
                    attention = attention.to(self.config.device)
                    image_dim = image.shape
                    batch_size = image_dim[0]
                    # Conditional image synthesis
                    print(f'epoch {epoch} validation: conditional synthesis')
                    synthetic_image = self.conditional_reverse_diffusion(image, lesion_mask, attention, loop_times=5)
                    fname = os.path.join(self.config.out_dir, 
                                         f'training_results_{self.timestr}',
                                         f'epoch{str(epoch).zfill(3)}_condition_sample.png')
                    save_sampled_image(torch.cat([synthetic_image, image * (1.0 - lesion_mask), image, lesion_mask], dim=1), fname)
                    break

    def inference(self, image, lesion_mask, task, loop_times=5, fname='test.png'):
        """
        Case #1: Lesion filling (MS --> Healthy)
            * image: torch.Tensor (num_slices, 1, 224, 224)
                MR images with lesions
            * lesion_mask: torch.Tensor (num_slices, 1, 224, 224)
                Binary lesion mask with value of '1' highlighting lesion regions
            
        Case #2: Lesion generation (Healthy --> MS)
            * image: torch.Tensor (num_slices, 1, 224, 224)
                MR images from healthy controls (no lesions in the image)
            * lesion_mask: torch.Tensor (num_slices, 1, 224, 224)
                Binary mask indicating regions to put synthetic lesions on.
        """
        with torch.set_grad_enabled(False):
            self.ddpm.eval()
            image = image.to(self.config.device)
            lesion_mask = lesion_mask.to(self.config.device)
            if image.shape[1] == 1:
                attention = lesion_mask.clone()
            else:  
                attention = lesion_mask.clone().repeat(1, image.shape[1], 1, 1)
            
            if task == 'lesion_filling':
                synthetic_image = self.conditional_reverse_diffusion(image, 
                                                                     lesion_mask=torch.zeros_like(image[:, 0:1, ...]), 
                                                                     attention=attention,
                                                                     loop_times=loop_times)
            elif task == 'lesion_generation':
                synthetic_image = self.conditional_reverse_diffusion(image, 
                                                                     lesion_mask=lesion_mask,
                                                                     attention=attention,
                                                                     loop_times=loop_times)
            else:
                print(f'{task} is not a valid input. Choose from: "lesion_filling" and "lesion_generation"...')
            save_sampled_image(torch.cat([synthetic_image, image * (1.0 - attention), image, attention], dim=0), fname)
        return synthetic_image

    def inference_ddim(self, image, lesion_mask, lesion_mask_input, task, steps_subset=None, loop_times=5, fname='test.png', synthetic_image=None, truncation_factor=25):
        scheduler = DDIMScheduler(num_steps=self.config.num_diffusion_steps, 
                                  alphas=self.config.alphas, 
                                  alpha_bars=self.config.alpha_bars, 
                                  steps_subset=steps_subset,
                                  task=task)
        with torch.set_grad_enabled(False):
            self.ddpm.eval()
            # setSeed()
            image = image.to(self.config.device)
            lesion_mask = lesion_mask.to(self.config.device)
            lesion_mask_input = lesion_mask_input.to(self.config.device)

            # # dilated attention
            # dilation_kernel_size = 3
            # dilation_kernel = torch.ones((1, 1, dilation_kernel_size, dilation_kernel_size), dtype=torch.float32).to(self.config.device)
            # dilated_lesion_mask = F.conv2d(lesion_mask.clone(), dilation_kernel, padding=dilation_kernel_size // 2)
            # dilated_lesion_mask = (dilated_lesion_mask > 1).float()

            if synthetic_image is not None:
                synthetic_image = synthetic_image.to(self.config.device)
            if image.shape[1] == 1:
                if task == 'lesion_generation':
                    # attention = dilated_lesion_mask.clone()
                    attention = (lesion_mask.clone() > 1e-2).float()
                else:
                    # attention = lesion_mask.clone()
                    attention = (lesion_mask.clone() > 1e-2).float()
            else:
                if task == 'lesion_generation':  
                    # attention = dilated_lesion_mask.clone().repeat(1, image.shape[1], 1, 1)
                    attention = (lesion_mask.clone().repeat(1, image.shape[1], 1, 1) > 1e-2).float()
                else:
                    # attention = lesion_mask.clone().repeat(1, image.shape[1], 1, 1)
                    attention = (lesion_mask.clone().repeat(1, image.shape[1], 1, 1) > 1e-2).float()

            # input contrast dropout, modify attention accordlingly
            for mc_idx in range(image.shape[1]):
                if torch.sum(image[:, mc_idx, ...]) == 0:
                    attention[:, mc_idx, ...] = 0

            # if inputs are all blank, switch attention to all 1s
            if torch.sum(image[:, :, ...]) == 0:
                attention[:, :, ...] = 1
            
            

            if task == 'lesion_filling':
                synthetic_image = self.conditional_reverse_diffusion_ddim(image, 
                                                                          lesion_mask=torch.zeros_like(image[:, 0:1, ...]), 
                                                                          attention=attention, 
                                                                          scheduler=scheduler, 
                                                                          loop_times=loop_times,
                                                                          synthetic_image=synthetic_image,
                                                                          truncation_factor=truncation_factor)
            elif task == 'lesion_generation':
                synthetic_image = self.conditional_reverse_diffusion_ddim(image, 
                                                                          lesion_mask=lesion_mask, 
                                                                          attention=attention, 
                                                                          scheduler=scheduler, 
                                                                          loop_times=loop_times,
                                                                          synthetic_image=synthetic_image,
                                                                          truncation_factor=truncation_factor)
            elif task == 'partial_filling_and_generation':
                synthetic_image = self.conditional_reverse_diffusion_ddim(image, 
                                                                          lesion_mask=lesion_mask_input, 
                                                                          attention=attention, 
                                                                          scheduler=scheduler, 
                                                                          loop_times=loop_times,
                                                                          synthetic_image=synthetic_image,
                                                                          truncation_factor=truncation_factor)
            else:
                raise ValueError(f'{task} is not a valid input. Choose from: "lesion_filling" and "lesion_generation".')
            save_sampled_image(torch.cat([synthetic_image, image * (1.0 - attention), image, attention], dim=0), fname)
        return synthetic_image


class DDIMScheduler:
    def __init__(self, num_steps, alphas, alpha_bars, eta=1.0, steps_subset=None, task='lesion_filling'):
        self.num_steps = num_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.task = task
        if self.task == 'lesion_filling':
            self.eta = 1.0
        elif self.task == 'lesion_generation':
            self.eta = 1.0
        elif self.task == 'partial_filling_and_generation':
            self.eta = 1.0
        else:
            raise ValueError(f'{task} is not a valid task. Choose from: "lesion_generation" and "lesion_filling".')
        # self.eta = eta
        self.steps_subset = steps_subset if steps_subset is not None else list(range(num_steps))
        self.alpha_bars_subset =  self.alpha_bars[self.steps_subset]
        self.alpha_bars_prev_subset = torch.cat((torch.tensor([1.0]).to('cuda'), self.alpha_bars_subset[:-1]))
        self.calculate_ddim_params()

    def calculate_ddim_params(self):
        self.sigmas_subset = self.eta * torch.sqrt((1 - self.alpha_bars_prev_subset) / (1 - self.alpha_bars_subset) * (1 - self.alpha_bars_subset / self.alpha_bars_prev_subset))

    def ddim_step(self, x_t, t, noise_pred):
        """
        Perform one step of the DDIM sampling process.
        """
        t_subset_idx = self.steps_subset.index(t)
        alpha_bar_tau_i = self.alpha_bars_subset[t_subset_idx]
        alpha_bar_tau_i_1 = self.alpha_bars_prev_subset[t_subset_idx]
        x0_est = (x_t - torch.sqrt(1 - alpha_bar_tau_i) * noise_pred) / torch.sqrt(alpha_bar_tau_i)
        pred_x0 = torch.sqrt(alpha_bar_tau_i_1) * x0_est
        dir_xt = pred_x0 + torch.sqrt(1 - alpha_bar_tau_i_1 - self.sigmas_subset[t_subset_idx]**2) * noise_pred
        noise = torch.randn_like(x_t) if t_subset_idx > 0 else torch.zeros_like(x_t)
        if self.task == 'lesion_filling':
            x_prev = dir_xt + 1.0 * self.sigmas_subset[t_subset_idx] * noise
        elif self.task == 'lesion_generation':
            x_prev = dir_xt + 1.5 * self.sigmas_subset[t_subset_idx] * noise
        elif self.task == 'partial_filling_and_generation':
            x_prev = dir_xt + 1.5 * self.sigmas_subset[t_subset_idx] * noise
        else:
            raise ValueError(f'{task} is not a valid task. Choose from: "lesion_generation" and "lesion_filling".')
        return x_prev, x0_est
