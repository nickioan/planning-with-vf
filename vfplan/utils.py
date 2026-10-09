import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seeds python, numpy (also used for terrain randomization) and torch."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


class Logger:
    """Prints scalars, appends them to `<out_dir>/<name>.csv` (columns: step, metric, value)
    and optionally logs them to wandb."""

    def __init__(self, out_dir, name="log", use_wandb=False, wandb_project=None, config=None):
        self.path = os.path.join(out_dir, f"{name}.csv")
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("step,metric,value\n")
        self.wandb = None
        if use_wandb:
            import wandb

            wandb.init(project=wandb_project, config=config, dir=out_dir)
            self.wandb = wandb

    def log(self, step, **scalars):
        scalars = {k: float(v) for k, v in scalars.items()}
        print(f"[{step}] " + " | ".join(f"{k}: {v:.4f}" for k, v in scalars.items()), flush=True)
        if self.wandb is not None:
            self.wandb.log(scalars, step=step)
        with open(self.path, "a") as f:
            for k, v in scalars.items():
                f.write(f"{step},{k},{v}\n")


def report_parameters(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_parameters / 1e6:.2f} M")
    return n_parameters
