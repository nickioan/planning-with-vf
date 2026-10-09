"""Composition of viability filters (paper Sec. 7.4): platforms with a hurdle on top.

Plans from a single diffusion planner are scored by the platform VF and the hurdle VF.
The scores are combined either by `std_weighted_sum` (each VF weighted by the standard
deviation of its scores over the sampled plans; the setting of the paper's composition
experiment) or by `product` (paper Eq. 9).

Example:
    python scripts/evaluate_composition.py --diffusion_ckpt DP.pt \
        --platform_vf VF_PLATFORM.pt --hurdle_vf VF_HURDLE.pt --out results/composition.csv
"""

import argparse
import csv
import os

import numpy as np

from vfplan.controller import load_controller
from vfplan.evaluation import evaluate
from vfplan.planner import DiffusionPlanner
from vfplan.rollout import RandomSelector, VFSelector
from vfplan.utils import get_device, set_seed
from vfplan.vf import load_vf


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--platform_vf", default=None)
    parser.add_argument("--hurdle_vf", default=None)
    parser.add_argument("--combine", choices=["std_weighted_sum", "product"], default="std_weighted_sum")
    parser.add_argument("--n_plans", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes_per_trial", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    planner = DiffusionPlanner.load(args.diffusion_ckpt, device)
    vfs = {}
    if args.platform_vf:
        vfs["platform"] = load_vf(args.platform_vf, device)
    if args.hurdle_vf:
        vfs["hurdle"] = load_vf(args.hurdle_vf, device)
    selector = VFSelector(vfs, device, combine=args.combine) if vfs else RandomSelector()

    rates = evaluate(
        "composed",
        None,
        load_controller(),
        planner,
        selector,
        args.n_plans,
        trials=args.trials,
        episodes_per_trial=args.episodes_per_trial,
        render=args.render,
    )
    name = "+".join(vfs) if vfs else "dp"
    print(f"composed ({name}, {args.combine}): {np.mean(rates):.2f} +- {np.std(rates):.2f}")

    if args.out:
        new_file = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["task", "planner", "value", "n_plans", "metric", "result"])
            planner_name = f"dp_vf[{name},{args.combine}]" if vfs else "dp"
            writer.writerows(
                ["composed", planner_name, "", args.n_plans, f"trial_{i}", r] for i, r in enumerate(rates)
            )


if __name__ == "__main__":
    main()
