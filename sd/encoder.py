import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self, *args):
        super().__init__(
            # (batch, channel, h, w) -> (batch, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch, 128 h, w) -> (batch, 128 h, w)
            VAE_ResidualBlock(128, 128),
            # (batch, 128 h, w) -> (batch, 128 h, w)
            VAE_ResidualBlock(128, 128),
            # (batch, 128, h, w) -> (batch, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (batch, 128, h/2, w/2) -> (batch, 256, h/2, w/2)
            VAE_ResidualBlock(128, 256),
            # (batch, 256, h/2, w/2) -> (batch, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (batch, 256, h/2, w/2) -> (batch, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (batch, 256, h/4, w/4) -> (batch, 512, h/4, w/4)
            VAE_ResidualBlock(256, 512),
            # (batch, 512, h/4, w/4) -> (batch, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, h/4, w/4) -> (batch, 512, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            VAE_AttentionBlock(512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            nn.GroupNorm(32, 512),
            # (batch, 512, h/8, w/8) -> (batch, 512, h/8, w/8)
            nn.SiLU(),
            # (batch, 512, h/8, w/8) -> (batch, 8, h/8, w/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch, 512, h/8, w/8) -> (batch, 8, h/8, w/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch, channel, h, w)
        # noise: (batch, out_channel, h/8, w/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # we learn the distributuion on the latent space and sample one from the latent space
        # (batch, 8, h/8, w/8) -> two tensors of (batch, 4, h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        x = mean + stdev * noise
        # scale the output by a constant
        x *= 0.18215
        return x
