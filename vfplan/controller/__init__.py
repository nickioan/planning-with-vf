"""Pre-trained low-level ALLSTEPS footstep controller (Xie et al. 2020).

The controller maps the humanoid state plus the next two target footsteps
(60-D observation) to 21 joint torques. It is treated as part of the
environment: planners only choose the footsteps it should follow.
"""

import os

import torch
import torch.nn as nn

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "allsteps_walker3d.pt")


class SoftsignActor(nn.Module):
    """ALLSTEPS actor network; actions are `tanh` of the network output."""

    def __init__(self, obs_dim=60, action_dim=21, hidden_size=256):
        super().__init__()
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softsign(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))

    @torch.no_grad()
    def act(self, obs):
        """Deterministic action for a single numpy observation."""
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self(obs).squeeze(0).numpy()


def load_controller(path=DEFAULT_CHECKPOINT):
    data = torch.load(path, map_location="cpu", weights_only=True)
    controller = SoftsignActor(**data["config"])
    controller.load_state_dict(data["state_dict"])
    controller.eval()
    return controller
