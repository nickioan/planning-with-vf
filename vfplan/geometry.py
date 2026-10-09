"""Coordinate conventions shared by data collection, planning and evaluation.

Plans are expressed relative to the character root in a frame whose "forward"
axis points from the root towards the current waypoint (yaw `heading`):
`x` is the lateral and `y` the forward offset, `z` the height offset.
"""

import numpy as np


def glob_to_rel(points, body_xyz, heading):
    """World points (N, >=3) -> (N, 3) offsets in the heading frame."""
    deltas = points[:, 0:3] - body_xyz
    angle = np.arctan2(deltas[:, 1], deltas[:, 0]) - heading
    distance = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)
    return np.stack(
        (np.sin(angle) * distance, np.cos(angle) * distance, deltas[:, 2]),
        axis=1,
    )


def rel_to_glob(plan, body_xyz, heading):
    """Inverse of `glob_to_rel` for a (N, >=3) plan; returns an array shaped like `plan`
    with world x, y, z filled in and all other columns set to zero."""
    world = np.zeros_like(plan)
    dr = np.linalg.norm(plan[:, 0:2], ord=2, axis=1)
    theta = np.arctan2(plan[:, 0], plan[:, 1]) + heading
    x_rel = dr * np.cos(theta)
    world[:, 0] = x_rel + body_xyz[0]
    world[:, 1] = x_rel * np.tan(theta) + body_xyz[1]
    world[:, 2] = plan[:, 2] + body_xyz[2]
    return world


def controller_targets(env, plan, body_xyz, heading):
    """Converts a plan made at (body_xyz, heading) into the controller's input format
    relative to the character's *current* root pose.

    Returns (targets (4, 5): x, y, z, x_tilt, y_tilt; world (4, 5) with world xyz).
    """
    world = rel_to_glob(plan, body_xyz, heading)
    robot = env.unwrapped.robot
    deltas = world[:, 0:3] - robot.body_xyz
    angle = np.arctan2(deltas[:, 1], deltas[:, 0]) - robot.body_rpy[2]
    distance = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)
    targets = np.stack(
        (
            np.sin(angle) * distance,
            np.cos(angle) * distance,
            deltas[:, 2],
            plan[:, 3],
            plan[:, 4],
        ),
        axis=1,
    )
    return targets, world


def policy_state(env, obs):
    """56-D character state used by the planner and the VFs.

    The controller observation without its footstep targets (50-D) followed by the
    world positions of both feet (6-D), with the first entry replaced by the root yaw
    and the contact flags refreshed after the step.
    """
    robot = env.unwrapped.robot
    feet = np.reshape(robot.feet_xyz, 6)
    state = np.concatenate((obs[:-10], feet))
    state[0] = robot.body_rpy[2]
    state[[48, 49]] = robot.feet_contact
    return state


def state_feet_heading_frame(state, body_xyz, heading):
    """Character state with the feet positions in the heading frame (planner / platform VF)."""
    state = np.copy(state)
    feet = np.reshape(state[-6:], (2, 3))
    state[-6:] = np.reshape(glob_to_rel(feet, body_xyz, heading), 6)
    return state


def state_feet_root_offset(state, body_xyz):
    """Character state with the feet positions as world-axis offsets from the root
    (hurdle / obstacle VFs)."""
    state = np.copy(state)
    feet = np.reshape(state[-6:], (2, 3))
    state[-6:] = np.reshape(feet - body_xyz, 6)
    return state


# ----------------------------------------------------------------------------- contacts


def foot_in_contact_near(env, target, tol):
    """True if a foot touching anything is within `tol` (3-D) of `target`."""
    for f in env.unwrapped.robot.feet:
        delta = f.pose().xyz() - target
        if (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5 < tol and f.contact_list():
            return True
    return False


def foot_near(env, point, tol):
    """True if a foot is within `tol` (3-D) of `point`, contact or not."""
    for f in env.unwrapped.robot.feet:
        delta = f.pose().xyz() - point
        if (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5 < tol:
            return True
    return False


def foot_inside_zone(env, center, radius):
    """True if a foot is inside the vertical cylinder of `radius` around `center`."""
    for f in env.unwrapped.robot.feet:
        delta = f.pose().xyz() - center
        if (delta[0] ** 2 + delta[1] ** 2) ** 0.5 < radius:
            return True
    return False


def hurdle_passed(env, hurdle):
    """True once both feet are further from the world origin than the hurdle, measured
    along the hurdle's direction from the origin (trajectories start at the origin)."""
    hurdle_dist = (hurdle[0] ** 2 + hurdle[1] ** 2) ** 0.5
    hurdle_theta = np.arctan2(hurdle[1], hurdle[0])
    for f in env.unwrapped.robot.feet:
        foot = f.pose().xyz()
        foot_dist = (foot[0] ** 2 + foot[1] ** 2) ** 0.5
        foot_proj = np.abs(np.cos(hurdle_theta - np.arctan2(foot[1], foot[0])) * foot_dist)
        if foot_proj < hurdle_dist:
            return False
    return True
