import torch
import torch.nn as nn
from diffusers import UNet2DModel

class DiffusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(sample_size=64, in_channels=3, out_channels=3)

    def forward(self, x_t, t, cond):
        return self.unet(x_t, t, encoder_hidden_states=cond).sample
