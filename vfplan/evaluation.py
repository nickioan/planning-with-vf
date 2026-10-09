"""Evaluation protocols of the paper (Sec. 7).

Each protocol runs `trials` x `episodes_per_trial` episodes and returns one success
rate per trial:

* platform  (Fig. 7, Tab. 8): every platform has the same height; success = not falling
  within 350 steps.
* hurdle    (Tab. 1): all hurdles have the same height; success = not falling within
  350 steps.
* obstacle  (Tab. 2): all avoidance zones have the same radius; success = reaching the
  waypoint behind the first zone without entering it or falling.
* composed  (Sec. 7.4): platforms with a hurdle on top; success = not falling within
  350 steps.
"""

import numpy as np

from vfplan.envs import EPISODE_STEPS, SHORT_EPISODE_STEPS, make_env
from vfplan.rollout import run_planning_episode, run_procedural_episode
from vfplan.tasks import HurdleCourse, ObstacleCourse, PlatformCourse

DEFAULT_VALUES = {
    "platform": [round(h, 2) for h in np.linspace(0.1, 0.65, 12)],
    "hurdle": [0.25, 0.30, 0.35],
    "obstacle": [0.5, 1.0, 1.5],
}


def make_eval_env(task, value, render=False):
    """Environment for one evaluation setting (`value`: platform/hurdle height or zone radius)."""
    if task == "platform":
        return make_env("platform", render, SHORT_EPISODE_STEPS)
    if task == "hurdle":
        return make_env("hurdle", render, SHORT_EPISODE_STEPS, hurdle_heights=(value,))
    if task == "obstacle":
        return make_env("obstacle", render, EPISODE_STEPS, obstacle_radius_range=(value, value))
    if task == "composed":
        return make_env("composed", render, SHORT_EPISODE_STEPS)
    raise ValueError(task)


def make_eval_course(task, env):
    if task == "platform":
        return PlatformCourse(env, waypoint_tol=0.55)
    if task == "hurdle":
        return HurdleCourse(env, waypoint_tol=1.5)
    if task == "obstacle":
        return ObstacleCourse(env, waypoint_tol=0.55, end_at_first_waypoint=True)
    if task == "composed":
        return HurdleCourse(env, waypoint_tol=0.55)
    raise ValueError(task)


def run_eval_episode(task, env, value, controller, planner=None, selector=None, n_plans=None):
    """Runs one evaluation episode; returns True on success."""
    e = env.unwrapped
    obs = env.reset()
    if task == "platform":
        e.set_platform_heights(value)
        obs[-10:] = e.delta_to_k_targets(k=e.lookahead).flatten()

    if planner is None:  # procedural baseline
        result = run_procedural_episode(env, obs, controller)
        return result.steps == env._max_episode_steps

    course = make_eval_course(task, env)
    course.reset()
    result = run_planning_episode(env, obs, controller, planner, course, selector, n_plans)
    if task == "obstacle":
        return course.wpt_idx == 1
    return result.steps == env._max_episode_steps


def evaluate(
    task,
    value,
    controller,
    planner=None,
    selector=None,
    n_plans=None,
    trials=5,
    episodes_per_trial=20,
    render=False,
    env=None,
    verbose=True,
):
    """Returns a list with the success rate of each trial."""
    own_env = env is None
    env = make_eval_env(task, value, render) if own_env else env
    rates = []
    for trial in range(trials):
        successes = 0
        for episode in range(episodes_per_trial):
            successes += run_eval_episode(task, env, value, controller, planner, selector, n_plans)
            if verbose:
                print(
                    f"{task}={value} n_plans={n_plans} trial {trial + 1}/{trials} "
                    f"episode {episode + 1}/{episodes_per_trial}: success rate "
                    f"{successes / (episode + 1):.2f}",
                    flush=True,
                )
        rates.append(successes / episodes_per_trial)
    if own_env:
        env.close()
    return rates
