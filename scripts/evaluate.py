"""Evaluate a planner on the platform, hurdle or obstacle task (paper Fig. 7, Tab. 1-3, Fig. 8).

Planners:
    procedural  follow the environment's procedurally generated trajectory ("PG")
    dp          execute a random diffusion sample ("DP")
    dp_vf       execute the diffusion sample with the highest VF value ("DP w/ VF")

Results are appended to a CSV with one row per (setting, number of plans, trial);
`scripts/summarize.py` turns it into mean +- std tables.

Examples:
    # Fig. 7: success rate per platform height
    python scripts/evaluate.py --task platform --planner dp_vf --diffusion_ckpt DP.pt --vf_ckpt VF.pt \
        --out results/platform.csv
    # Fig. 8: number of plans, hardest setting
    python scripts/evaluate.py --task obstacle --planner dp_vf --values 1.5 --n_plans 2 4 8 20 50 100 200 ...
    # Tab. 3: planning time
    python scripts/evaluate.py --task obstacle --planner dp_vf --time_inference --n_plans 100 200 400 ...
"""

import argparse
import csv
import os
import time

import numpy as np

from vfplan.controller import load_controller
from vfplan.evaluation import DEFAULT_VALUES, evaluate, make_eval_course, make_eval_env
from vfplan.geometry import policy_state
from vfplan.planner import DiffusionPlanner
from vfplan.rollout import RandomSelector, VFSelector
from vfplan.utils import get_device, set_seed
from vfplan.vf import load_vf

DEFAULT_N_PLANS = {"platform": 100, "hurdle": 200, "obstacle": 200}


def time_inference(task, value, planner, selector, n_plans, iterations=20):
    """Average wall time of one sample-and-select step (first call excluded)."""
    env = make_eval_env(task, value)
    obs = env.reset()
    course = make_eval_course(task, env)
    course.reset()
    state = policy_state(env, obs)
    times = []
    for i in range(iterations + 1):
        start = time.time()
        planset = planner.plan(env, course.waypoint, state, n_plans)
        selector.select(planset, course)
        if i > 0:
            times.append(time.time() - start)
    env.close()
    return float(np.mean(times))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["platform", "hurdle", "obstacle"], required=True)
    parser.add_argument("--planner", choices=["procedural", "dp", "dp_vf"], required=True)
    parser.add_argument("--diffusion_ckpt", default=None)
    parser.add_argument("--vf_ckpt", default=None)
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="platform heights / hurdle heights / zone radii [m] (default: paper settings)",
    )
    parser.add_argument(
        "--n_plans", type=int, nargs="+", default=None, help="default: 100 (platform), 200 (others)"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes_per_trial", type=int, default=20)
    parser.add_argument("--time_inference", action="store_true", help="report planning time instead (Tab. 3)")
    parser.add_argument("--out", default=None, help="CSV file to append results to")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    values = args.values or DEFAULT_VALUES[args.task]
    n_plans_list = args.n_plans or [DEFAULT_N_PLANS[args.task]]
    if args.planner == "procedural":
        n_plans_list = [0]
    if args.task == "obstacle" and args.planner == "procedural":
        parser.error("the procedural trajectory ignores avoidance zones; use dp or dp_vf")

    controller = load_controller()
    planner = selector = None
    if args.planner != "procedural":
        if args.diffusion_ckpt is None:
            parser.error("--diffusion_ckpt is required")
        planner = DiffusionPlanner.load(args.diffusion_ckpt, device)
        selector = RandomSelector()
    if args.planner == "dp_vf":
        if args.vf_ckpt is None:
            parser.error("--vf_ckpt is required for dp_vf")
        vf = load_vf(args.vf_ckpt, device)
        assert vf.task == args.task, f"VF was trained for {vf.task!r}"
        selector = VFSelector({args.task: vf}, device)

    rows = []
    for n_plans in n_plans_list:
        for value in values:
            if args.time_inference:
                seconds = time_inference(args.task, value, planner, selector, n_plans)
                print(f"{args.task} n_plans={n_plans}: {seconds:.4f} s per planning step")
                rows.append([args.task, args.planner, value, n_plans, "time_s", seconds])
                continue
            rates = evaluate(
                args.task,
                value,
                controller,
                planner,
                selector,
                n_plans,
                trials=args.trials,
                episodes_per_trial=args.episodes_per_trial,
                render=args.render,
            )
            print(f"{args.task}={value} n_plans={n_plans}: {np.mean(rates):.2f} +- {np.std(rates):.2f}")
            rows += [[args.task, args.planner, value, n_plans, f"trial_{i}", r] for i, r in enumerate(rates)]

    if args.out:
        new_file = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["task", "planner", "value", "n_plans", "metric", "result"])
            writer.writerows(rows)
        print(f"Appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
