"""Train the diffusion planner on successful plans (paper Sec. 6.1, Supp. Tab. 5).

Example:
    python scripts/train_diffusion.py --data data/flat data/platform_* --out runs/diffusion
"""

import argparse
import copy
import os

import numpy as np
import torch

from vfplan.controller import load_controller
from vfplan.data import GaussianNormalizer, load_datasets
from vfplan.envs import EPISODE_STEPS, make_env
from vfplan.planner import DIFFUSION_CONFIG, NORMALIZER_KEYS, DiffusionPlanner, build_diffusion
from vfplan.rollout import RandomSelector, run_planning_episode
from vfplan.tasks import PlatformCourse
from vfplan.utils import Logger, get_device, make_dir, report_parameters, save_json, set_seed


def cycle(dataloader):
    while True:
        yield from dataloader


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = ma_params.data * self.beta + (1 - self.beta) * current_params.data


def load_training_data(paths):
    data = load_datasets(paths, ["plans", "heightmaps", "humanoid_state", "waypoints"])
    plans = data["plans"]
    plans[:, :, 2] = np.clip(plans[:, :, 2], -3, 3)

    # Only plans that were executed successfully are used to train the planner.
    success = plans[:, 0, -1] == 1.0
    plans = plans[success][:, :, :-1]
    cond = np.clip(data["heightmaps"][success], -3, 3)
    humanoid = data["humanoid_state"][success]
    wpts = data["waypoints"][success]
    print(f"{len(plans)} successful plans out of {len(success)}")

    normalizers = {
        "state": GaussianNormalizer.fit(plans[:, :, [0, 1, 2]]),
        "wpt": GaussianNormalizer.fit(wpts),
        "cond": GaussianNormalizer.fit(np.expand_dims(cond, -1)),
        "humanoid": GaussianNormalizer.fit(humanoid),
    }
    normed_plans = np.copy(plans)
    normed_plans[:, :, [0, 1, 2]] = normalizers["state"].normalize(plans[:, :, [0, 1, 2]])
    tensors = (
        torch.Tensor(normed_plans),
        torch.Tensor(normalizers["cond"].normalize(cond)[:, None]),
        torch.Tensor(normalizers["humanoid"].normalize(humanoid)),
        torch.Tensor(normalizers["wpt"].normalize(wpts)),
    )
    return torch.utils.data.TensorDataset(*tensors), normalizers


def evaluate(diffusion, normalizers, device, n_episodes):
    """Success rate of executing single diffusion samples on the platform task."""
    planner = DiffusionPlanner(diffusion, normalizers, device)
    env = make_env("platform", max_episode_steps=EPISODE_STEPS)
    controller = load_controller()
    course = PlatformCourse(env, waypoint_tol=0.45)
    lengths = []
    for _ in range(n_episodes):
        obs = env.reset()
        course.reset()
        result = run_planning_episode(env, obs, controller, planner, course, RandomSelector(), n_plans=1)
        lengths.append(result.steps)
    env.close()
    lengths = np.array(lengths)
    return {"success_rate": np.mean(lengths == EPISODE_STEPS), "avg_episode_length": lengths.mean()}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", nargs="+", required=True, help="directories / files with data.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_steps", type=int, default=750001)
    parser.add_argument("--diffusion_steps", type=int, default=DIFFUSION_CONFIG["n_timesteps"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accumulate", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--step_start_ema", type=int, default=2000)
    parser.add_argument("--update_ema_every", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=5, help="0 disables evaluation")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt_dir = make_dir(os.path.join(args.out, "checkpoints"))
    save_json(vars(args), os.path.join(args.out, "args.json"))
    logger = Logger(args.out, "train_log", args.wandb, "vfplan-diffusion", vars(args))

    dataset, normalizers = load_training_data(args.data)
    config = {**DIFFUSION_CONFIG, "n_timesteps": args.diffusion_steps}
    diffusion = build_diffusion(config).to(device)
    report_parameters(diffusion)

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(diffusion)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True))

    start_step = 0
    if args.resume:
        data = torch.load(args.resume, map_location=device, weights_only=True)
        diffusion.load_state_dict(data["model"])
        ema_model.load_state_dict(data["ema"])
        start_step = data["step"]

    def save(step):
        path = os.path.join(ckpt_dir, f"diffusion_{step}.pt")
        torch.save(
            {
                "step": step,
                "config": config,
                "model": diffusion.state_dict(),
                "ema": ema_model.state_dict(),
                "normalizers": {k: normalizers[k].state_dict() for k in NORMALIZER_KEYS},
            },
            path,
        )
        print(f"Saved {path}", flush=True)

    for step in range(start_step, args.train_steps):
        for _ in range(args.grad_accumulate):
            batch, cond, state, wpt = (x.to(device) for x in next(dataloader))
            loss = diffusion.loss(batch, cond, state, wpt) / args.grad_accumulate
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.update_ema_every == 0:
            if step < args.step_start_ema:
                ema_model.load_state_dict(diffusion.state_dict())
            else:
                ema.update_model_average(ema_model, diffusion)

        if step % args.save_freq == 0:
            save(step)
            if args.eval_episodes > 0:
                diffusion.eval()
                logger.log(step, **evaluate(diffusion, normalizers, device, args.eval_episodes))
                diffusion.train()

        if step % args.log_freq == 0:
            logger.log(step, loss=loss.item() * args.grad_accumulate)

    if (args.train_steps - 1) % args.save_freq != 0:
        save(args.train_steps - 1)


if __name__ == "__main__":
    main()
