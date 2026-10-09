"""Feature encoders shared by the diffusion planner and the viability filters."""

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class MLP(nn.Module):
    """Linear layers with optional LayerNorm, ReLU activations and a linear output.

    Adapted from robomimic (https://github.com/ARISE-Initiative/robomimic, MIT License).
    """

    def __init__(self, input_dim, output_dim, layer_dims=(), normalization=False):
        super().__init__()
        layers = []
        dim = input_dim
        for width in layer_dims:
            layers.append(nn.Linear(dim, width))
            if normalization:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            dim = width
        layers.append(nn.Linear(dim, output_dim))
        self._model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self._model(inputs)


class HeightMapEncoder(nn.Module):
    """ResNet-18 encoder for the egocentric heightfield.

    For inputs smaller than 64x64 the last ResNet stage is removed, giving a
    256-D feature for the 32x32 heightfields used in the paper.
    """

    def __init__(self, input_image_shape=(1, 32, 32)):
        super().__init__()
        self.map_model = resnet18()
        self.map_model.conv1 = nn.Conv2d(input_image_shape[0], 64, kernel_size=(5, 5))
        self.map_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_model.fc = nn.Identity()
        if input_image_shape[1] < 64 or input_image_shape[2] < 64:
            self.map_model.layer4 = nn.Identity()
            self.output_dim = 256
        else:
            self.output_dim = 512
        self._output_activation = nn.ReLU()

    def forward(self, map_inputs):
        return self._output_activation(self.map_model(map_inputs))


class PlannerConditionEncoder(nn.Module):
    """Encodes the diffusion planner's conditioning: heightfield, character state, waypoint."""

    def __init__(self, init_st_dim, hm_shape=(1, 32, 32)):
        super().__init__()
        self.hum_st_mlp = MLP(init_st_dim, 128, layer_dims=(init_st_dim, 64, 128), normalization=True)
        self.wpt_mlp = MLP(3, 128, layer_dims=(32, 64), normalization=True)
        self.feature_extractor = HeightMapEncoder(input_image_shape=hm_shape)
        self.output_dim = self.feature_extractor.output_dim + 128 + 128

    def forward(self, cond, state_hist, wpt):
        c = torch.flatten(self.feature_extractor(cond), 1)
        st = self.hum_st_mlp(state_hist)
        w = self.wpt_mlp(wpt)
        return torch.cat((c, st, w), -1)
