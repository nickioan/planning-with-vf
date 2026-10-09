"""Collect 4-footstep plans by following procedurally generated trajectories (paper Fig. 6).

The ALLSTEPS controller follows the environment's own footstep trajectory. Every time
a footstep is reached, the next four footsteps are recorded as a plan together with the
planner's conditioning (heightfield, character state, waypoint) or, for hurdles, the
hurdle state used by the hurdle VF.

When an episode ends (fall or time limit), plans that extend beyond the last footstep
actually reached are labelled as not successful (label < 1) and the final plan gets a
return of 0. Following the original experiments, only episodes that are not a full
success (surviving all steps with a reward above `--min_reward`) are stored.

Examples (paper Supp. B.2):
    python scripts/collect_data.py --task flat --n_samples 100000 --out data/flat
    python scripts/collect_data.py --task platform --min_height 0.10 --out data/platform_10cm
    python scripts/collect_data.py --task hurdle --hurdle_heights 0.25 --out data/hurdle_25cm
"""

import argparse
import os

import numpy as np

from vfplan.controller import load_controller
from vfplan.envs import EPISODE_STEPS, make_env
from vfplan.geometry import (
    foot_near,
    glob_to_rel,
    hurdle_passed,
    policy_state,
    state_feet_heading_frame,
    state_feet_root_offset,
)
from vfplan.utils import make_dir, save_json, set_seed

GAMMA = 0.75
WAYPOINT_TOL = 0.55


def inside_heightmap(env, targets):
    e = env.unwrapped
    if np.any(targets[:, 0] < e.hm_x_window[0]) or np.any(targets[:, 0] > e.hm_x_window[1]):
        return False
    if np.any(targets[:, 1] < e.hm_y_window[0]) or np.any(targets[:, 1] > e.hm_y_window[1]):
        return False
    return True


def snapshot(env, task, state, target, hurdle_angle=None, zero_hurdle=False, k=4):
    """Records the next k trajectory footsteps and the state they are planned from.

    `target` is the current waypoint (flat/platform) or hurdle (hurdle task). Returns None
    if, for heightfield tasks, the footsteps fall outside the heightmap.
    """
    e = env.unwrapped
    record = {}
    if task != "hurdle":
        hm, _, heading = e.get_scandot()
        record["heightmap"] = (hm - e.robot.robot_body.pose().xyz()[2]).astype(np.float32)
        record["heading"] = heading

    # Right after a footstep is reached the trajectory index has not advanced yet.
    start = e.next_step_index + 1 if e.target_reached_count == 1 else e.next_step_index
    targets = e.terrain_info[start : start + k]
    if task != "hurdle" and not inside_heightmap(env, targets):
        return None

    body_xyz = e.robot.body_xyz
    record["end_index"] = start + k
    record["footsteps"] = np.concatenate((targets[:, :3], targets[:, 4:]), axis=1)
    record["body_xyz"] = body_xyz
    if task == "hurdle":
        record["humanoid_state"] = state_feet_root_offset(state, body_xyz)
        rel = target if zero_hurdle else target - body_xyz
        record["hurdle_state"] = np.concatenate((rel, [hurdle_angle]))
    else:
        record["humanoid_state"] = state_feet_heading_frame(state, body_xyz, heading)
        record["waypoint"] = target - body_xyz
    return record


def label_episode(env, task, records, success):
    """Relative plans with success labels, and discounted returns (paper Eq. 4)."""
    feet, _ = env.unwrapped.get_foot_placement()
    max_feet = len(feet) + 1

    confidence = 0.75
    if len(records) <= 3:
        confidence -= (4 - len(records)) * 0.25

    plans = []
    for r in records:
        label = np.ones((4, 1))
        if r["end_index"] > max_feet and success:
            break
        elif r["end_index"] > max_feet:
            label = np.zeros((4, 1)) + confidence
            confidence -= 0.25
        rel = np.copy(r["footsteps"])
        if task == "hurdle":
            rel[:, :3] = r["footsteps"][:, 0:3] - r["body_xyz"]
        else:
            rel[:, :3] = glob_to_rel(r["footsteps"], r["body_xyz"], r["heading"])
        plans.append(np.concatenate((rel, label), axis=1))

    discounts = np.flip(np.cumsum(np.cumprod(np.ones(len(plans) - 1) * GAMMA)))
    returns = np.ones(len(plans))
    if not success:
        returns[-1] = 0.0
        returns[:-2] = returns[:-2] + discounts[1:]
    else:
        returns[:-1] = returns[:-1] + discounts
    return plans, returns


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["flat", "platform", "hurdle"], required=True)
    parser.add_argument("--n_samples", type=int, default=100000, help="number of plans to collect")
    parser.add_argument("--out", required=True, help="output directory (data.npz is written there)")
    parser.add_argument(
        "--min_height", type=float, default=0.10, help="platform task: lowest platform height [m]"
    )
    parser.add_argument(
        "--max_height", type=float, default=0.65, help="platform task: highest platform height [m]"
    )
    parser.add_argument("--hurdle_heights", type=float, nargs="+", default=[0.25, 0.30, 0.35])
    parser.add_argument("--min_reward", type=float, default=5700, help="episode reward of a full success")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    make_dir(args.out)
    save_json(vars(args), os.path.join(args.out, "args.json"))

    env = make_env(
        args.task,
        render=args.render,
        max_episode_steps=EPISODE_STEPS,
        platform_height_range=(args.min_height, args.max_height),
        hurdle_heights=args.hurdle_heights,
    )
    e = env.unwrapped
    controller = load_controller()

    dataset = {
        k: [] for k in ["plans", "humanoid_state", "returns", "heightmaps", "waypoints", "hurdle_states"]
    }
    n_plans = 0
    n_episodes = 0
    fails = 0
    curriculum_freq = args.n_samples // (1.5 * e.max_curriculum)
    next_level = curriculum_freq

    while n_plans < args.n_samples:
        obs = env.reset(randomize=True)

        if args.task == "hurdle":
            hurdle_idx = 0
            final_hurdle = False
            target = np.array(e.hurdle_locations[0])
        else:
            waypoints = e.platform_locations if args.task == "platform" else e.waypoint_locations
            wpt_idx = 0
            target = np.array(waypoints[0])
        e.change_waypoint(target)
        if args.render:
            e.render_waypoint(target)

        hurdle_angle = e.hurdle_angles[0] if args.task == "hurdle" else None
        records = [snapshot(env, args.task, policy_state(env, obs), target, hurdle_angle)]

        swing_leg = 0
        ep_reward = 0
        step = 0
        done = False
        while not done:
            step += 1
            obs, reward, done, _ = env.step(controller.act(obs))
            state = policy_state(env, obs)
            ep_reward += reward

            if args.task == "hurdle":
                if not final_hurdle and hurdle_passed(env, target):
                    hurdle_idx += 1
                    if hurdle_idx >= len(e.hurdle_locations):
                        final_hurdle = True
                        target = np.zeros_like(target)
                        hurdle_angle = 0
                    else:
                        target = np.array(e.hurdle_locations[hurdle_idx])
                        hurdle_angle = e.hurdle_angles[hurdle_idx]
                    e.change_waypoint(target)
            elif foot_near(env, target, WAYPOINT_TOL):
                wpt_idx += 1
                target = e.terrain_info[-1, :3] if wpt_idx >= len(waypoints) else np.array(waypoints[wpt_idx])
                e.change_waypoint(target)
            if args.render:
                e.render_waypoint(target)

            if e.get_new_step_channel():
                e.set_new_step_channel(False)
                # Abort if the controller stepped with the wrong leg.
                if e.step_leg_idx != swing_leg:
                    done = True
                swing_leg = (swing_leg + 1) % 2
                zero_hurdle = args.task == "hurdle" and final_hurdle
                records.append(snapshot(env, args.task, state, target, hurdle_angle, zero_hurdle))

        records = [r for r in records if r is not None]
        n_episodes += 1
        success = step == EPISODE_STEPS and ep_reward > args.min_reward
        fails += not success
        progress = hurdle_idx if args.task == "hurdle" else wpt_idx
        if args.task == "flat":
            keep = progress > 1
        else:
            keep = (not success) and progress > 0

        if keep and records:
            plans, returns = label_episode(env, args.task, records, success)
            records = records[: len(plans)]
            dataset["plans"] += plans
            dataset["returns"] += list(returns)
            dataset["humanoid_state"] += [r["humanoid_state"] for r in records]
            if args.task == "hurdle":
                dataset["hurdle_states"] += [r["hurdle_state"] for r in records]
            else:
                dataset["heightmaps"] += [r["heightmap"] for r in records]
                dataset["waypoints"] += [r["waypoint"] for r in records]

        n_plans = len(dataset["plans"])
        if n_plans >= next_level:
            e.update_curriculum(e.curriculum + 1)
            next_level += curriculum_freq
        print(
            f"episodes: {n_episodes} | plans: {n_plans}/{args.n_samples} | "
            f"episode steps: {step} | fail rate: {fails / n_episodes:.3f}",
            flush=True,
        )

    out = {k: np.array(v) for k, v in dataset.items() if len(v)}
    if args.task == "flat":
        out.pop("returns")
    np.savez_compressed(os.path.join(args.out, "data.npz"), **out)
    print(f"Saved {n_plans} plans to {os.path.join(args.out, 'data.npz')}")


if __name__ == "__main__":
    main()
