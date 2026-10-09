"""Online training of a viability filter with a frozen diffusion planner (paper Alg. 1).

Every episode, plans sampled from the diffusion planner are executed (epsilon-greedy
w.r.t. the VF). Transitions that reach the next footstep get reward 1 and go to a
"success" buffer; the transition during which the character falls gets reward 0 and
goes to a "failure" buffer. After each episode the VF takes a gradient step on a batch
drawn equally from both buffers with the target
    y = r + gamma * max_{p' in P'} VF_target(s', p')
where P' are the plans sampled at s'.

Examples:
    # platform: bootstrap from the offline VF (paper Supp. B.3)
    python scripts/train_vf_online.py --task platform --diffusion_ckpt DP.pt \
        --init_from runs/vf_offline_platform/checkpoints/vf_40.pt --out runs/vf_online_platform
    # hurdle / obstacle: from scratch
    python scripts/train_vf_online.py --task obstacle --diffusion_ckpt DP.pt --out runs/vf_online_obstacle
"""

import argparse
import os

import numpy as np
import torch

from vfplan.controller import load_controller
from vfplan.envs import EPISODE_STEPS, make_env
from vfplan.evaluation import evaluate
from vfplan.planner import DiffusionPlanner, score_plans
from vfplan.rollout import PlanSelector, VFSelector, run_planning_episode
from vfplan.tasks import HurdleCourse, ObstacleCourse, PlatformCourse, vf_inputs
from vfplan.utils import Logger, get_device, make_dir, save_json, set_seed
from vfplan.vf import ReplayBuffer, build_vf, load_vf, save_vf

STATE_SHAPES = {
    "platform": [(1, 32, 32), (56,)],
    "hurdle": [(4,), (56,)],
    "obstacle": [(4,), (3,), (56,)],
}
# Waypoint tolerance and evaluation setting used while training each task's VF.
WAYPOINT_TOL = {"platform": 0.55, "hurdle": 0.45, "obstacle": 0.55}
EVAL_SETTING = {"platform": 0.6, "hurdle": 0.35, "obstacle": 1.5}


class EpsilonGreedyCollector(PlanSelector):
    """Selects plans epsilon-greedily w.r.t. the VF and records transitions."""

    def __init__(self, task, vf, device, success_buffer, failure_buffer, epsilon):
        self.task = task
        self.vf = vf
        self.device = device
        self.success_buffer = success_buffer
        self.failure_buffer = failure_buffer
        self.epsilon = epsilon
        self.previous = None

    def select(self, planset, course):
        state, plans = vf_inputs(self.task, planset, course)
        if np.random.rand() < (1 - self.epsilon):
            index = int(np.argmax(score_plans(self.vf, state, plans, self.device)))
        else:
            index = np.random.randint(len(planset))
        self._current = (state, plans, index)
        return index

    def on_replan(self, planset, index, course):
        state, plans, index = self._current
        if self.previous is not None:
            prev_state, prev_plan, _ = self.previous
            # The previous plan reached its first footstep: reward 1.
            self.success_buffer.insert(prev_state, prev_plan, 1.0, state, plans, True)
        self.previous = (state, plans[index], plans)

    def on_episode_end(self, result):
        self.failed = result.steps != EPISODE_STEPS
        if self.failed:
            state, plan, plans = self.previous
            self.failure_buffer.insert(state, plan, 0.0, state, plans, False)
        self.previous = None


def td_loss(vf, target_vf, success_buffer, failure_buffer, batch_size, device):
    s_batch = success_buffer.sample(batch_size // 2)
    f_batch = failure_buffer.sample(batch_size // 2)

    def cat(a, b):
        if isinstance(a, list):
            return [torch.cat([x, y], dim=0).to(device) for x, y in zip(a, b)]
        return torch.cat([a, b], dim=0).to(device)

    states, plans, rewards, next_states, next_plans, masks = (cat(a, b) for a, b in zip(s_batch, f_batch))

    q_values = vf(tuple(states), plans)
    with torch.no_grad():
        # max over the plans sampled at s' (paper Eq. 6)
        embed = target_vf.get_state_embeddings(tuple(next_states))
        next_plans = next_plans.transpose(0, 1)  # [n_plans x batch x 4 x 5]
        target_values = torch.stack(
            [target_vf.predict_from_embeddings(embed, p).flatten() for p in next_plans]
        )
        max_q = target_values.max(dim=0).values.unsqueeze(1)
        target = rewards + masks * vf.gamma * max_q
    return torch.nn.functional.smooth_l1_loss(q_values, target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["platform", "hurdle", "obstacle"], required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init_from", default=None, help="VF checkpoint to start from (e.g. the offline VF)")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--n_plans", type=int, default=200, help="plans sampled per replanning step")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--tau", type=float, default=0.01, help="soft target-update rate")
    parser.add_argument("--success_buffer_size", type=int, default=100000)
    parser.add_argument("--failure_buffer_size", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=500, help="failure transitions before updating")
    parser.add_argument("--eps_start", type=float, default=0.3)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=5000, help="in episodes")
    parser.add_argument("--hurdle_heights", type=float, nargs="+", default=[0.25, 0.30, 0.35])
    parser.add_argument("--obstacle_radius", type=float, nargs=2, default=[1.5, 1.5], metavar=("MIN", "MAX"))
    parser.add_argument("--save_freq", type=int, default=5000, help="in episodes")
    parser.add_argument("--eval_freq", type=int, default=2500, help="in episodes; 0 disables")
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt_dir = make_dir(os.path.join(args.out, "checkpoints"))
    save_json(vars(args), os.path.join(args.out, "args.json"))
    logger = Logger(args.out, "train_log", args.wandb, f"vfplan-vf-online-{args.task}", vars(args))

    env = make_env(
        args.task,
        render=args.render,
        max_episode_steps=EPISODE_STEPS,
        hurdle_heights=args.hurdle_heights,
        obstacle_radius_range=args.obstacle_radius,
    )
    course = {
        "platform": PlatformCourse,
        "hurdle": HurdleCourse,
        "obstacle": ObstacleCourse,
    }[args.task](env, WAYPOINT_TOL[args.task])
    controller = load_controller()
    planner = DiffusionPlanner.load(args.diffusion_ckpt, device)

    config = {"gamma": args.gamma}
    if args.init_from:
        vf = load_vf(args.init_from, device).train()
        vf.gamma = args.gamma
        if args.task == "platform":
            # Keep the offline-trained state encoders fixed (paper Supp. B.3).
            for name, module in vf.named_children():
                if name in ("hum_st_mlp", "feature_extractor"):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
    else:
        vf = build_vf(args.task, **config).to(device)
    target_vf = build_vf(args.task, **config).to(device)
    target_vf.load_state_dict(vf.state_dict())
    target_vf.eval()
    optimizer = torch.optim.Adam(vf.parameters(), lr=args.lr)

    shapes = STATE_SHAPES[args.task]
    success_buffer = ReplayBuffer(args.success_buffer_size, shapes, (4, 5), (args.n_plans, 4, 5))
    failure_buffer = ReplayBuffer(args.failure_buffer_size, shapes, (4, 5), (args.n_plans, 4, 5))
    collector = EpsilonGreedyCollector(args.task, vf, device, success_buffer, failure_buffer, args.eps_start)

    n_successes = 0
    progress = []
    randomize = True
    for episode in range(args.episodes):
        collector.epsilon = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-episode / args.eps_decay)
        obs = env.reset(randomize=randomize)
        course.reset()
        run_planning_episode(env, obs, controller, planner, course, collector, args.n_plans)
        # After a failure the same terrain is used again.
        randomize = not collector.failed
        n_successes += not collector.failed
        progress.append(course.progress())

        scalars = {
            "success_rate_running_avg": n_successes / (episode + 1),
            "objects_per_episode": np.mean(progress),
            "success_buffer": len(success_buffer),
            "failure_buffer": len(failure_buffer),
        }
        ready = (
            len(failure_buffer) >= args.warmup
            and min(len(success_buffer), len(failure_buffer)) >= args.batch_size // 2
        )
        if ready:
            optimizer.zero_grad()
            loss = td_loss(vf, target_vf, success_buffer, failure_buffer, args.batch_size, device)
            loss.backward()
            optimizer.step()
            scalars["loss"] = loss.item()

            target_state = target_vf.state_dict()
            for key, value in vf.state_dict().items():
                target_state[key] = value * args.tau + target_state[key] * (1 - args.tau)
            target_vf.load_state_dict(target_state)
        logger.log(episode, **scalars)

        if (episode + 1) % args.save_freq == 0:
            save_vf(target_vf, os.path.join(ckpt_dir, f"vf_{episode + 1}.pt"), episode + 1, config)

        if args.eval_freq and (episode + 1) % args.eval_freq == 0:
            selector = VFSelector({args.task: target_vf}, device)
            rates = evaluate(
                args.task,
                EVAL_SETTING[args.task],
                controller,
                planner,
                selector,
                args.n_plans,
                trials=1,
                episodes_per_trial=args.eval_episodes,
                verbose=False,
            )
            logger.log(episode, eval_success_rate=rates[0])

    save_vf(target_vf, os.path.join(ckpt_dir, "vf_final.pt"), args.episodes, config)


if __name__ == "__main__":
    main()
