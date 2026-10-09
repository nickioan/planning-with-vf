"""Watch the planner in the pybullet GUI, or record a video.

Pass one or more VFs as `task=path`; several VFs are combined by multiplying their
values (paper Eq. 9). Without `--vf`, random diffusion samples are executed.

Examples:
    python scripts/demo.py --task platform --diffusion_ckpt DP.pt --vf platform=VF_PLATFORM.pt
    python scripts/demo.py --task sequential --diffusion_ckpt DP.pt \
        --vf platform=VF_PLATFORM.pt --vf hurdle=VF_HURDLE.pt --vf obstacle=VF_OBSTACLE.pt
    python scripts/demo.py --task hurdle --diffusion_ckpt DP.pt --vf hurdle=VF_HURDLE.pt \
        --record demo.mp4 --episodes 1
"""

import argparse

import numpy as np

from vfplan.controller import load_controller
from vfplan.envs import EPISODE_STEPS, make_env
from vfplan.planner import DiffusionPlanner
from vfplan.rollout import RandomSelector, VFSelector, run_planning_episode
from vfplan.tasks import HurdleCourse, ObstacleCourse, PlatformCourse, SequentialCourse
from vfplan.utils import get_device, set_seed
from vfplan.vf import load_vf

COURSES = {
    "platform": lambda env: PlatformCourse(env, waypoint_tol=0.55),
    "hurdle": lambda env: HurdleCourse(env, waypoint_tol=1.5),
    "obstacle": lambda env: ObstacleCourse(env, waypoint_tol=0.55),
    "sequential": lambda env: SequentialCourse(env, waypoint_tol=0.55),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=list(COURSES), required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--vf", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--n_plans", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--platform_height", type=float, default=None, help="platform task: fixed height [m]")
    parser.add_argument("--record", default=None, help="write an mp4 here instead of opening the GUI")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    env = make_env(args.task, render=args.record is None, max_episode_steps=args.max_steps)
    e = env.unwrapped
    planner = DiffusionPlanner.load(args.diffusion_ckpt, device)
    controller = load_controller()

    vfs = {}
    for spec in args.vf:
        task, path = spec.split("=", 1)
        vfs[task] = load_vf(path, device)
    selector = VFSelector(vfs, device, combine="product") if vfs else RandomSelector()

    frames = []
    on_step = (lambda: frames.append(e.render())) if args.record else None
    course = COURSES[args.task](env)
    for episode in range(args.episodes):
        obs = env.reset()
        if args.task == "platform" and args.platform_height is not None:
            e.set_platform_heights(args.platform_height)
        course.reset()
        result = run_planning_episode(env, obs, controller, planner, course, selector, args.n_plans, on_step)
        print(f"episode {episode + 1}: {result.steps} steps, progress {course.progress()}")

    if args.record:
        from moviepy.editor import ImageSequenceClip

        ImageSequenceClip([np.asarray(f) for f in frames], fps=int(1 / e.control_step)).write_videofile(
            args.record
        )
    env.close()


if __name__ == "__main__":
    main()
