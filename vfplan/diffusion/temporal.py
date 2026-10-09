# Adapted from Diffuser (Janner et al. 2022, https://github.com/jannerm/diffuser, MIT License).
# See THIRD_PARTY_NOTICES.md.
import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from vfplan.diffusion.helpers import (
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )
        self.embed_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, emb):
        """
        x : [batch_size x inp_channels x horizon]
        emb : [batch_size x embed_dim]
        returns : [batch_size x out_channels x horizon]
        """
        out = self.blocks[0](x) + self.embed_mlp(emb)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalTemporalUnet(nn.Module):
    """1D temporal U-Net; the conditioning embedding is added in every residual block."""

    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dims_down=(128, 256), kernel_size=3):
        super().__init__()

        dims = [transition_dim, *dims_down]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        embed_dim = cond_dim + time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # The (unused) attention slot of Diffuser's U-Net is kept as nn.Identity so
        # that checkpoint parameter names stay unchanged.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim, kernel_size),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim, kernel_size),
                        nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, kernel_size),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim, kernel_size),
                        nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=kernel_size),
            nn.Conv1d(dim_in, transition_dim, 1),
        )

    def forward(self, x, embed, time):
        """
        x : [batch x horizon x transition]
        embed : [batch x cond_dim]
        time : [batch]
        """
        x = einops.rearrange(x, "b h t -> b t h")
        emb = torch.cat((embed, self.time_mlp(time)), -1)

        h = []
        for resnet, resnet2, _, downsample in self.downs:
            x = resnet(x, emb)
            x = resnet2(x, emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        for resnet, resnet2, _, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, emb)
            x = resnet2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")
