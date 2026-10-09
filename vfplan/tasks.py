"""Per-task bookkeeping during an episode: the waypoint sequence that conditions the
diffusion planner, the obstacle the VF currently attends to, and task-specific
termination.

Waypoint tolerances differ between data collection, training and evaluation in the
experiments of the paper; the defaults below reproduce them and are set by the
calling scripts.
"""

import numpy as np

from vfplan.geometry import foot_inside_zone, foot_near, hurdle_passed


class Course:
    """Follows a list of waypoints; after the last one, the end of the trajectory."""

    def __init__(self, env, waypoint_tol):
        self.env = env.unwrapped
        self.waypoint_tol = waypoint_tol

    def _waypoint_list(self):
        raise NotImplementedError

    def reset(self):
        self.waypoints = self._waypoint_list()
        self.wpt_idx = 0
        self.waypoint = np.array(self.waypoints[0])
        self.env.change_waypoint(self.waypoint)
        if self.env.is_render:
            self.env.render_waypoint(self.waypoint)

    def _advance_waypoint(self):
        self.wpt_idx += 1
        if self.wpt_idx >= len(self.waypoints):
            self.waypoint = self.env.terrain_info[-1, :3]
        else:
            self.waypoint = np.array(self.waypoints[self.wpt_idx])
        self.env.change_waypoint(self.waypoint)
        if self.env.is_render:
            self.env.render_waypoint(self.waypoint)

    def update(self):
        """Call after every env step. Returns True if the episode should end."""
        if foot_near(self.env, self.waypoint, self.waypoint_tol):
            self._advance_waypoint()
        return False

    def progress(self):
        return self.wpt_idx


class FlatCourse(Course):
    def _waypoint_list(self):
        return self.env.waypoint_locations


class PlatformCourse(Course):
    """Waypoints are the platform tops."""

    def _waypoint_list(self):
        return self.env.platform_locations


class HurdleCourse(Course):
    """Waypoints are the hurdle centers; the VF attends to the next hurdle not yet passed.

    `zero_after_last`: once every hurdle is passed, report an all-zero hurdle state
    (as during data collection) instead of the (zero) hurdle position minus the root.
    """

    def __init__(self, env, waypoint_tol, zero_after_last=False):
        super().__init__(env, waypoint_tol)
        self.zero_after_last = zero_after_last

    def _waypoint_list(self):
        return self.env.hurdle_locations

    def reset(self):
        super().reset()
        self.hurdle_idx = 0
        self.final_hurdle = False
        self.hurdle = np.array(self.env.hurdle_locations[0])
        self.hurdle_angle = self.env.hurdle_angles[0]

    def update(self):
        super().update()
        if not self.final_hurdle and hurdle_passed(self.env, self.hurdle):
            self.hurdle_idx += 1
            if self.hurdle_idx >= len(self.env.hurdle_locations):
                self.final_hurdle = True
                self.hurdle = np.zeros_like(self.hurdle)
                self.hurdle_angle = 0
            else:
                self.hurdle = np.array(self.env.hurdle_locations[self.hurdle_idx])
                self.hurdle_angle = self.env.hurdle_angles[self.hurdle_idx]
        return False

    def hurdle_state(self, body_xyz):
        if self.final_hurdle and self.zero_after_last:
            return np.concatenate((self.hurdle, [self.hurdle_angle]))
        return np.concatenate((self.hurdle - body_xyz, [self.hurdle_angle]))

    def progress(self):
        return self.hurdle_idx


class ObstacleCourse(Course):
    """Waypoints lie between consecutive avoidance zones; entering a zone ends the episode.

    With `end_at_first_waypoint` (evaluation), reaching the first waypoint, i.e. getting
    past the first zone, also ends the episode.
    """

    def __init__(self, env, waypoint_tol, end_at_first_waypoint=False):
        super().__init__(env, waypoint_tol)
        self.end_at_first_waypoint = end_at_first_waypoint

    def _waypoint_list(self):
        return self.env.waypoint_locations

    def reset(self):
        super().reset()
        self.zone_idx = 0
        self.zone = np.array(self.env.obstacle_locations[0])
        self.zone_radius = self.env.obstacle_radius[0]
        self.entered_zone = False

    def update(self):
        done = False
        if foot_near(self.env, self.waypoint, self.waypoint_tol):
            done = self.end_at_first_waypoint
            self.wpt_idx += 1
            self.waypoint = np.array(self.waypoints[min(self.wpt_idx, len(self.waypoints) - 1)])
            self.env.change_waypoint(self.waypoint)
            if self.env.is_render:
                self.env.render_waypoint(self.waypoint)
            self.zone_idx += 1
            if self.zone_idx >= len(self.env.obstacle_locations):
                self.zone = np.zeros_like(self.zone)
                self.zone_radius = 0
            else:
                self.zone = np.array(self.env.obstacle_locations[self.zone_idx])
                self.zone_radius = self.env.obstacle_radius[self.zone_idx]

        if foot_inside_zone(self.env, self.zone, self.zone_radius):
            self.entered_zone = True
            done = True
        return done

    def zone_state(self, body_xyz):
        return np.concatenate((self.zone - body_xyz, [self.zone_radius]))

    def progress(self):
        return self.zone_idx


class SequentialCourse(Course):
    """Demo course with platforms, hurdles and avoidance zones in sequence."""

    def _waypoint_list(self):
        return self.env.waypoints

    def reset(self):
        super().reset()
        self.final_hurdle = False
        self.hurdle_idx = 0
        self.hurdle = np.array(self.env.hurdle_locations[0])
        self.hurdle_angle = self.env.hurdle_angles[0]
        self.zone_idx = 0
        self.zone = np.array(self.env.obstacle_locations[0])
        self.zone_radius = self.env.obstacle_radius[0]

    def update(self):
        super().update()
        body_xyz = self.env.robot.body_xyz
        if body_xyz[0] > self.zone[0] and self.zone_idx + 1 < len(self.env.obstacle_locations):
            self.zone_idx += 1
            self.zone = np.array(self.env.obstacle_locations[self.zone_idx])
            self.zone_radius = self.env.obstacle_radius[self.zone_idx]
        if not self.final_hurdle and hurdle_passed(self.env, self.hurdle):
            self.hurdle_idx += 1
            if self.hurdle_idx >= len(self.env.hurdle_locations):
                self.final_hurdle = True
                self.hurdle = np.zeros_like(self.hurdle)
                self.hurdle_angle = 0
            else:
                self.hurdle = np.array(self.env.hurdle_locations[self.hurdle_idx])
                self.hurdle_angle = self.env.hurdle_angles[self.hurdle_idx]
        return False

    hurdle_state = HurdleCourse.hurdle_state
    zero_after_last = False
    zone_state = ObstacleCourse.zone_state


# ------------------------------------------------------------------- VF inputs


def vf_inputs(task, planset, course):
    """Builds the VF inputs for every plan in `planset`.

    Returns (state, plans): `state` is a tuple of numpy arrays with a leading batch
    dimension of 1, `plans` has shape (N, 4, 5).
    """
    if task == "platform":
        state = (planset.heightmap[None, None], planset.state_heading[None])
        return state, planset.plans
    if task == "hurdle":
        state = (course.hurdle_state(planset.body_xyz)[None], planset.state_root[None])
        return state, planset.plans_root
    if task == "obstacle":
        state = (
            course.zone_state(planset.body_xyz)[None],
            (course.waypoint - planset.body_xyz)[None],
            planset.state_root[None],
        )
        return state, planset.plans_root
    raise ValueError(f"No viability filter for task {task!r}")
