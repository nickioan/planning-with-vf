"""Diffusion planner + viability filters (paper Fig. 2).

`DiffusionPlanner.plan` samples N 4-footstep plans for the character's current
situation; `score_plans` evaluates them with a VF; `select` picks the best plan,
optionally combining several VFs.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vfplan.data import GaussianNormalizer
from vfplan.diffusion import ConditionalGaussianDiffusion, ConditionalTemporalUnet
from vfplan.encoders import PlannerConditionEncoder
from vfplan.geometry import rel_to_glob, state_feet_heading_frame, state_feet_root_offset

DIFFUSION_CONFIG = dict(
    horizon=4,
    plan_dim=5,
    init_st_dim=56,
    hm_shape=(1, 32, 32),
    n_timesteps=20,
    predict_epsilon=True,
    loss_type="l2",
)
NORMALIZER_KEYS = ("state", "wpt", "cond", "humanoid")


def build_diffusion(config=None):
    config = {**DIFFUSION_CONFIG, **(config or {})}
    encoder = PlannerConditionEncoder(config["init_st_dim"], hm_shape=tuple(config["hm_shape"]))
    model = ConditionalTemporalUnet(
        horizon=config["horizon"],
        transition_dim=config["plan_dim"],
        cond_dim=encoder.output_dim,
    )
    return ConditionalGaussianDiffusion(
        model,
        encoder,
        horizon=config["horizon"],
        observation_dim=config["plan_dim"],
        n_timesteps=config["n_timesteps"],
        predict_epsilon=config["predict_epsilon"],
        loss_type=config["loss_type"],
    )


@dataclass
class PlanSet:
    """N candidate plans made at one replanning step."""

    plans: np.ndarray  # (N, 4, 5) heading frame, z re-projected onto the terrain
    plans_root: np.ndarray  # (N, 4, 5) world-axis offsets from the root, zero tilts
    body_xyz: np.ndarray  # root position when planning
    heading: float  # yaw from the root towards the waypoint
    heightmap: np.ndarray  # (32, 32) heightfield minus root height, clipped to [-3, 3]
    state_heading: np.ndarray  # (56,) character state, feet in the heading frame
    state_root: np.ndarray  # (56,) character state, feet as offsets from the root

    def __len__(self):
        return len(self.plans)


class DiffusionPlanner:
    def __init__(self, diffusion, normalizers, device="cpu"):
        self.diffusion = diffusion.to(device).eval()
        self.normalizers = normalizers
        self.device = device

    @classmethod
    def load(cls, path, device="cpu", use_ema=False):
        """Loads a checkpoint written by `scripts/train_diffusion.py`."""
        data = torch.load(path, map_location=device, weights_only=True)
        diffusion = build_diffusion(data["config"])
        diffusion.load_state_dict(data["ema" if use_ema else "model"])
        normalizers = {k: GaussianNormalizer.from_state_dict(data["normalizers"][k]) for k in NORMALIZER_KEYS}
        return cls(diffusion, normalizers, device)

    @torch.no_grad()
    def sample(self, heightmap, state_heading, waypoint_rel, n_plans):
        """Samples plans from (unnormalized) inputs; returns (N, 4, 5) float32 in the heading frame."""
        nz = self.normalizers
        cond = nz["cond"].normalize(np.clip(heightmap, -3, 3))[None, None]
        hum = nz["humanoid"].normalize(state_heading)[None]
        wpt = nz["wpt"].normalize(waypoint_rel)[None]
        to_t = lambda x: torch.Tensor(x).to(self.device)  # noqa: E731
        plans = self.diffusion(to_t(cond), to_t(hum), to_t(wpt), n_samples=n_plans)
        plans = plans.detach().cpu().numpy()
        plans[:, :, :3] = nz["state"].unnormalize(plans[:, :, :3])
        return plans

    def plan(self, env, waypoint, state, n_plans):
        """Samples `n_plans` plans for the current situation in `env`.

        The z of every footstep is replaced by the terrain height at its x, y.
        """
        e = env.unwrapped
        hm, _, heading = e.get_scandot()
        heightmap = np.clip(hm - e.robot.robot_body.pose().xyz()[2], -3, 3)
        body_xyz = np.copy(e.robot.body_xyz)

        state_heading = state_feet_heading_frame(state, body_xyz, heading)
        state_root = state_feet_root_offset(state, body_xyz)

        e.disable_humanoid_collisions()
        plans = self.sample(heightmap, state_heading, waypoint - body_xyz, n_plans)
        plans_root = np.copy(plans)
        for s in range(n_plans):
            world = rel_to_glob(plans[s], body_xyz, heading)
            heights, _ = e.get_height_from_coordinate(world)
            plans[s, :, 2] = np.asarray(heights) - body_xyz[2]
            plans_root[s] = world
            plans_root[s, :, 2] = heights
        e.enable_humanoid_collisions()
        plans_root[:, :, :3] = plans_root[:, :, :3] - body_xyz

        return PlanSet(
            plans=plans,
            plans_root=plans_root,
            body_xyz=body_xyz,
            heading=heading,
            heightmap=heightmap,
            state_heading=state_heading,
            state_root=state_root,
        )


@torch.no_grad()
def score_plans(vf, state, plans, device):
    """VF value of every plan; `state` is a tuple of numpy arrays with batch size 1."""
    state = tuple(torch.Tensor(s).to(device) for s in state)
    return vf.evaluate(state, torch.Tensor(plans).to(device)).cpu().numpy().flatten()


def combine_scores(scores, mode="product"):
    """Combines the scores of several VFs for the same plans.

    "product": VF(p) = prod_k VF_k(p) (paper Eq. 9).
    "std_weighted_sum": sum_k std(VF_k) * VF_k(p), as used for the composition
        experiment of Sec. 7.4.
    """
    scores = [np.asarray(s) for s in scores]
    if mode == "product":
        return np.prod(scores, axis=0)
    if mode == "std_weighted_sum":
        return sum(np.std(s) * s for s in scores)
    raise ValueError(mode)
