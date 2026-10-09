"""Offline training of a viability filter on collected plans and their returns (paper Eq. 4).

Uses both successful and failed plans from `scripts/collect_data.py`. The diffusion
planner is only needed for the periodic evaluation.

Examples:
    python scripts/train_vf_offline.py --task platform --data data/platform_* \
        --diffusion_ckpt runs/diffusion/checkpoints/diffusion_750000.pt --out runs/vf_offline_platform
    python scripts/train_vf_offline.py --task hurdle --data data/hurdle_* \
        --diffusion_ckpt runs/diffusion/checkpoints/diffusion_750000.pt --out runs/vf_offline_hurdle
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vfplan.controller import load_controller
from vfplan.data import load_datasets
from vfplan.evaluation import evaluate
from vfplan.planner import DiffusionPlanner
from vfplan.rollout import VFSelector
from vfplan.utils import Logger, get_device, make_dir, save_json, set_seed
from vfplan.vf import build_vf, save_vf

EVAL_SETTING = {"platform": 0.6, "hurdle": 0.35}


def load_vf_dataset(task, paths):
    state_key = "heightmaps" if task == "platform" else "hurdle_states"
    data = load_datasets(paths, ["plans", state_key, "humanoid_state", "returns"])
    plans = data["plans"][:, :, :-1]
    plans[:, :, 2] = np.clip(plans[:, :, 2], -3, 3)
    state = data[state_key]
    if task == "platform":
        state = np.clip(state, -3, 3)[:, None]
    return TensorDataset(
        torch.Tensor(plans),
        torch.Tensor(state),
        torch.Tensor(data["humanoid_state"]),
        torch.Tensor(data["returns"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["platform", "hurdle"], required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--diffusion_ckpt", default=None, help="needed for evaluation only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=5, help="in epochs")
    parser.add_argument("--eval_episodes", type=int, default=100, help="0 disables evaluation")
    parser.add_argument("--n_plans", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt_dir = make_dir(os.path.join(args.out, "checkpoints"))
    save_json(vars(args), os.path.join(args.out, "args.json"))
    logger = Logger(args.out, "train_log", args.wandb, f"vfplan-vf-offline-{args.task}", vars(args))

    dataset = load_vf_dataset(args.task, args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = {"gamma": args.gamma}
    vf = build_vf(args.task, **config).to(device)
    optimizer = torch.optim.Adam(vf.parameters(), lr=args.lr)
    criterion = torch.nn.SmoothL1Loss()

    planner = controller = None
    if args.diffusion_ckpt and args.eval_episodes > 0:
        planner = DiffusionPlanner.load(args.diffusion_ckpt, device)
        controller = load_controller()

    def run_eval(step):
        if planner is None:
            return
        vf.eval()
        rates = evaluate(
            args.task,
            EVAL_SETTING[args.task],
            controller,
            planner,
            VFSelector({args.task: vf}, device),
            args.n_plans,
            trials=1,
            episodes_per_trial=args.eval_episodes,
            verbose=False,
        )
        vf.train()
        logger.log(step, eval_success_rate=rates[0])

    step = 1
    run_eval(0)
    for epoch in range(args.epochs):
        for plans, state, humanoid, returns in tqdm(dataloader, desc=f"epoch {epoch}"):
            plans, state, humanoid, returns = (x.to(device) for x in (plans, state, humanoid, returns))
            out = torch.squeeze(vf((state, humanoid), plans))
            loss = criterion(out, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.log_freq == 0:
                logger.log(step, loss=loss.item())
            step += 1

        if (epoch + 1) % args.save_freq == 0:
            run_eval(step)
            save_vf(vf, os.path.join(ckpt_dir, f"vf_{epoch + 1}.pt"), epoch + 1, config)


if __name__ == "__main__":
    main()
