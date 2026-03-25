import torch
from torch import nn
import torch.nn.functional as F
import math

class FusionNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        # return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))


class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = UNet(in_ch=config.in_ch, out_ch=config.out_ch, condition_ch=None, time_embed_ch=config.time_embedding_dim, base_ch=16,  num_levels=4)
        self.cal_time_embedding = SinusodialEmbedding(embedding_size=config.time_embedding_dim)

    def forward(self, x, t, condition=None):
        time_embedding = self.cal_time_embedding(t)
        return self.model(x, t=time_embedding, condition=condition)
    

class EmbeddingFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embedding_fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x, unsqueeze=False):
        x = self.embedding_fc(x)
        if unsqueeze and len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_ch, condition_ch=None, num_levels=3, base_ch=8, final_activation='noact'):
        super().__init__()
        self.final_activation = final_activation
        self.out_ch = out_ch
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 1, 1, 0),
            nn.LeakyReLU(0.2),
            ConvBlock2d(base_ch, base_ch, base_ch))
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.time_embed_fcs = nn.ModuleList()
        self.condition_fcs = nn.ModuleList() 
        for level in range(num_levels):
            curr_ch = base_ch * (2 ** level)
            self.down_convs.append(ConvBlock2d(curr_ch, curr_ch, curr_ch*2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(UpsampleBlock2d(curr_ch*4))
            self.up_convs.append(ConvBlock2d(curr_ch*4, curr_ch*2, curr_ch*2))
            self.time_embed_fcs.append(EmbeddingFC(time_embed_ch, curr_ch*2))
            self.condition_fcs.append(EmbeddingFC(condition_ch, curr_ch*2) if condition_ch else None)
        bottleneck_ch = base_ch * (2 ** num_levels)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch, bottleneck_ch*2)
        self.out_conv = nn.Conv2d(base_ch*2, out_ch, 1, 1, 0)

    def forward(self, in_tensor, t, condition=None):
        """
        Forward pass of U-Net.
        ==INPUT==
        * in_tensor: torch.Tensor (batch_size, in_ch, image_dim, image_dim)
            Input tensor.
        * t: torch.Tensor (batch_size, time_embed_ch)
            Time embedding.
        * condition: torch.Tensor (batch_size, condition_ch)
            Condition variable. For example, time embedding of DDPM.
        ==OUTPUT==
        * torch.Tensor (batch_size, out_ch, image_dim, image_dim)
            U-Net should not change the spatial dimension of an input tensor.
        """
        encoded_tensors = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_tensors.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_tensor, up_conv, up_sample, time_embed_fc, condition_fc in zip(
            reversed(encoded_tensors), 
            reversed(self.up_convs), 
            reversed(self.up_samples),
            reversed(self.time_embed_fcs),
            reversed(self.condition_fcs)):
            x = up_sample(x, encoded_tensor)
            if condition_fc is not None and condition is not None:
                x = up_conv(x) * condition_fc(condition).unsqueeze(-1).unsqueeze(-1) + time_embed_fc(t).unsqueeze(-1).unsqueeze(-1)
            else:
                x = up_conv(x) + time_embed_fc(t).unsqueeze(-1).unsqueeze(-1)
        x = in_tensor[:, 0:self.out_ch, ...] + self.out_conv(x)
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'relu':
            x = torch.relu(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


class UpsampleBlock2d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, in_tensor, tensor_from_skip_connection):
        upsampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        upsampled_tensor = self.conv_block(upsampled_tensor)
        return torch.cat([tensor_from_skip_connection, upsampled_tensor], dim=1)
    

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, in_tensor):
        return self.conv_block(in_tensor)
    

class SinusodialEmbedding(nn.Module):
    def __init__(self, embedding_size, max_period=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_period = max_period

    def forward(self, timesteps):
        timesteps = timesteps.view(-1, 1).float()
        half_dim = self.embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        args = timesteps * freqs.unsqueeze(0)
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return encoding