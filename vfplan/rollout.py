"""Receding-horizon execution of footstep plans (paper Sec. 4).

At every replanning step N plans are sampled and one is chosen; the controller is
then driven towards the plan's first two footsteps until a foot lands on the first
one, at which point the character replans.
"""

from dataclasses import dataclass

import numpy as np

from vfplan.geometry import controller_targets, foot_in_contact_near, policy_state
from vfplan.planner import combine_scores, score_plans
from vfplan.tasks import vf_inputs

FOOTSTEP_REACHED_TOL = 0.3


@dataclass
class EpisodeResult:
    steps: int  # number of control steps executed
    task_done: bool  # ended by the course (e.g. reached the goal or entered an avoidance zone)


class PlanSelector:
    """Chooses one plan out of a `PlanSet`; subclasses implement `select`.

    The `on_replan` / `on_episode_end` hooks let online VF training record transitions.
    """

    def select(self, planset, course):
        raise NotImplementedError

    def on_replan(self, planset, index, course):
        pass

    def on_episode_end(self, result):
        pass


class RandomSelector(PlanSelector):
    """Diffusion planner only: execute a random sample."""

    def select(self, planset, course):
        return np.random.randint(len(planset))


class VFSelector(PlanSelector):
    """Execute the plan with the highest (combined) viability.

    Args:
        vfs: {task: viability filter}; several VFs are combined with `combine_scores`.
    """

    def __init__(self, vfs, device, combine="product"):
        self.vfs = vfs
        self.device = device
        self.combine = combine

    def scores(self, planset, course):
        out = {}
        for task, vf in self.vfs.items():
            state, plans = vf_inputs(task, planset, course)
            out[task] = score_plans(vf, state, plans, self.device)
        return out

    def select(self, planset, course):
        scores = list(self.scores(planset, course).values())
        if len(scores) == 1:
            return int(np.argmax(scores[0]))
        return int(np.argmax(combine_scores(scores, self.combine)))


def run_planning_episode(env, obs, controller, planner, course, selector, n_plans, on_step=None):
    """Runs one episode starting from `obs`.

    The env must just have been reset and `course.reset()` called. `on_step()` is called
    after every env step (e.g. to record video frames).
    """
    e = env.unwrapped
    state = policy_state(env, obs)
    replan = True
    step = 0
    done = task_done = False
    while not done:
        step += 1
        if replan:
            planset = planner.plan(env, course.waypoint, state, n_plans)
            index = selector.select(planset, course)
            selector.on_replan(planset, index, course)
            plan = planset.plans[index]
            _, world = controller_targets(env, plan, planset.body_xyz, planset.heading)
            e.next_stone_target(world)
            replan = False

        # Drive the controller towards the plan's next two footsteps.
        targets, world = controller_targets(env, plan, planset.body_xyz, planset.heading)
        obs[-10:] = np.reshape(targets[0:2, :], 10)
        obs, _, done, _ = env.step(controller.act(obs))
        state = policy_state(env, obs)
        if on_step is not None:
            on_step()

        if course.update():
            done = task_done = True
        if foot_in_contact_near(env, world[0, :3], FOOTSTEP_REACHED_TOL):
            replan = True

    result = EpisodeResult(steps=step, task_done=task_done)
    selector.on_episode_end(result)
    return result


def run_procedural_episode(env, obs, controller):
    """Baseline "PG": the controller follows the environment's own procedural trajectory."""
    step = 0
    done = False
    while not done:
        step += 1
        obs, _, done, _ = env.step(controller.act(obs))
    return EpisodeResult(steps=step, task_done=False)
