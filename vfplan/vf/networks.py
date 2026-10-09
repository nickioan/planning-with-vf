"""Viability filters: Q-functions VF(s, p) over a state s and a 4-footstep plan p.

Each VF embeds every input with its own network, concatenates the embeddings
and regresses the expected discounted "alive" return (paper Sec. 4, Supp. A.2).
"""

import torch
import torch.nn as nn

from vfplan.encoders import MLP, HeightMapEncoder


class ViabilityFilter(nn.Module):
    """Base class: `state` is a tuple whose last element is the character state."""

    def __init__(self, horizon, plan_dim, init_st_dim, gamma=0.75):
        super().__init__()
        self.hum_st_mlp = MLP(init_st_dim, 256, layer_dims=(64, 128), normalization=True)
        self.traj_mlp = MLP(horizon * plan_dim, 256, layer_dims=(32, 64, 128), normalization=True)
        self.gamma = gamma

    def _build_critic(self, embed_dim):
        self.critic = MLP(embed_dim, 1, (512, 256, 128, 64, 32), normalization=True)

    def get_state_embeddings(self, state):
        raise NotImplementedError

    def predict_from_embeddings(self, embed, plans):
        x = self.traj_mlp(torch.flatten(plans, 1))
        return self.critic(torch.cat((embed, x), -1))

    def forward(self, state, plans):
        """state: tuple of [batch x ...] tensors, plans: [batch x horizon x plan_dim]."""
        return self.predict_from_embeddings(self.get_state_embeddings(state), plans)

    @torch.no_grad()
    def evaluate(self, state, plans):
        """Scores N plans for a single state (state tensors have batch size 1)."""
        embed = self.get_state_embeddings(state).repeat(len(plans), 1)
        return self.predict_from_embeddings(embed, plans)


class PlatformVF(ViabilityFilter):
    """State: (heightfield [B x 1 x 32 x 32], character state [B x 56])."""

    task = "platform"

    def __init__(self, horizon=4, plan_dim=5, init_st_dim=56, hm_shape=(1, 32, 32), gamma=0.75):
        super().__init__(horizon, plan_dim, init_st_dim, gamma)
        self.feature_extractor = HeightMapEncoder(input_image_shape=hm_shape)
        self._build_critic(self.feature_extractor.output_dim + 256 + 256)

    def get_state_embeddings(self, state):
        cond, state_hist = state
        c = torch.flatten(self.feature_extractor(cond), 1)
        return torch.cat((c, self.hum_st_mlp(state_hist)), -1)


class HurdleVF(ViabilityFilter):
    """State: (hurdle [B x 4] = position relative to root + yaw, character state [B x 56])."""

    task = "hurdle"

    def __init__(self, horizon=4, plan_dim=5, init_st_dim=56, hurdle_dim=4, gamma=0.75):
        super().__init__(horizon, plan_dim, init_st_dim, gamma)
        self.obst_mlp = MLP(hurdle_dim, 256, layer_dims=(32, 64, 128), normalization=True)
        self._build_critic(256 + 256 + 256)

    def get_state_embeddings(self, state):
        hurdle, state_hist = state
        return torch.cat((self.obst_mlp(hurdle), self.hum_st_mlp(state_hist)), -1)


class ObstacleVF(ViabilityFilter):
    """State: (avoidance zone [B x 4] = center relative to root + radius,
    waypoint relative to root [B x 3], character state [B x 56])."""

    task = "obstacle"

    def __init__(self, horizon=4, plan_dim=5, init_st_dim=56, obstacle_dim=4, wpt_dim=3, gamma=0.75):
        super().__init__(horizon, plan_dim, init_st_dim, gamma)
        self.mines_mlp = MLP(obstacle_dim, 256, layer_dims=(32, 64, 128), normalization=True)
        self.wpt_mlp = MLP(wpt_dim, 256, layer_dims=(32, 64, 128), normalization=True)
        self._build_critic(256 + 256 + 256 + 256)

    def get_state_embeddings(self, state):
        obstacle, wpt, state_hist = state
        return torch.cat((self.mines_mlp(obstacle), self.wpt_mlp(wpt), self.hum_st_mlp(state_hist)), -1)


VF_CLASSES = {"platform": PlatformVF, "hurdle": HurdleVF, "obstacle": ObstacleVF}


def build_vf(task, **kwargs):
    return VF_CLASSES[task](**kwargs)


def load_vf(path, device="cpu"):
    """Loads a VF checkpoint written by the VF training scripts."""
    data = torch.load(path, map_location=device, weights_only=True)
    vf = build_vf(data["task"], **data.get("config", {}))
    vf.load_state_dict(data["model"])
    return vf.to(device).eval()


def save_vf(vf, path, step, config=None):
    torch.save(
        {"task": vf.task, "config": config or {}, "step": step, "model": vf.state_dict()},
        path,
    )
